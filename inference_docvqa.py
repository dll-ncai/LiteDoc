# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
Batch inference + ANLS evaluation for DeepSeek-VL2 on the DocVQA dataset.

Expected annotation format (the official DocVQA json, e.g. train_v1.0.json /
val_v1.0.json), either as a single --annotation_file or as a --questions_dir
containing one or more such json files:

    {
        "data": [
            {
                "questionId": 1,
                "question": "What is the title?",
                "image": "documents/xxxxxxxx.png",
                "answers": ["Foo", "foo bar"]
            },
            ...
        ]
    }

("answers" is optional; if absent, ANLS is not computed for that sample.)

Example:
    python inference_docvqa.py \
        --model_path deepseek-ai/deepseek-vl2-tiny \
        --images_dir /data/docvqa/images \
        --annotation_file /data/docvqa/val_v1.0.json \
        --output_file docvqa_predictions.jsonl \
        --num_eval_samples 200
"""

import argparse
import glob
import json
import os
import sys

import PIL.Image
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_pil_images(conversations) -> list:
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
    return pil_images


# ---------------------------------------------------------------------------
# ANLS metric (Average Normalized Levenshtein Similarity), as defined for
# DocVQA (Biten et al., 2019). A prediction scores 0 against a gold answer
# once their normalized edit distance is >= 0.5; otherwise it scores
# 1 - normalized_edit_distance. The final per-question score is the best
# score across all gold answers.
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def anls_score(prediction: str, gold_answers, threshold: float = 0.5) -> float:
    if not gold_answers:
        return 0.0

    prediction = " ".join(prediction.strip().lower().split())
    best = 0.0
    for gold in gold_answers:
        gold = " ".join(str(gold).strip().lower().split())
        max_len = max(len(prediction), len(gold))
        if max_len == 0:
            similarity = 1.0
        else:
            nl = levenshtein_distance(prediction, gold) / max_len
            similarity = 1.0 - nl if nl < threshold else 0.0
        best = max(best, similarity)
    return best


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_docvqa_entries(annotation_file: str = None, questions_dir: str = None):
    if annotation_file:
        json_files = [annotation_file]
    else:
        json_files = sorted(glob.glob(os.path.join(questions_dir, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in --questions_dir '{questions_dir}'")

    entries = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        data = content["data"] if isinstance(content, dict) and "data" in content else content
        if not isinstance(data, list):
            raise ValueError(f"Unexpected annotation format in '{path}': expected a list of entries")
        entries.extend(data)

    if not entries:
        raise ValueError("No question entries were loaded; check --annotation_file / --questions_dir")

    return entries


def resolve_image_path(image_field: str, images_dir: str) -> str:
    candidates = [
        image_field,
        os.path.join(images_dir, image_field),
        os.path.join(images_dir, os.path.basename(image_field)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate image '{image_field}' under images_dir '{images_dir}' "
        f"(tried: {candidates})"
    )


def clean_answer(text: str) -> str:
    text = text.strip()
    # Model output is expected to be a short phrase; only keep the first line
    # in case it continues rambling past the answer.
    text = text.splitlines()[0] if text else text
    return text.strip().strip(".").strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if not os.path.isdir(args.images_dir):
        raise NotADirectoryError(f"--images_dir '{args.images_dir}' does not exist")

    entries = load_docvqa_entries(args.annotation_file, args.questions_dir)
    if args.num_eval_samples > 0:
        entries = entries[: args.num_eval_samples]
    print(f"Loaded {len(entries)} DocVQA questions for evaluation.")

    device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from: {args.model_path}")
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    vl_gpt = vl_gpt.to(device).eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)

    anls_scores = []
    num_errors = 0

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        progress = tqdm(entries, desc="DocVQA inference")
        for entry in progress:
            question_id = entry.get("questionId")
            question = entry.get("question")
            gold_answers = entry.get("answers")

            try:
                image_path = resolve_image_path(entry["image"], args.images_dir)

                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image>\n{question}\n{args.prompt_suffix}",
                        "images": [image_path],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]

                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True,
                    system_prompt="",
                ).to(vl_gpt.device, dtype=dtype)

                with torch.no_grad():
                    if args.chunk_size == -1:
                        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                        past_key_values = None
                    else:
                        inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                            input_ids=prepare_inputs.input_ids,
                            images=prepare_inputs.images,
                            images_seq_mask=prepare_inputs.images_seq_mask,
                            images_spatial_crop=prepare_inputs.images_spatial_crop,
                            attention_mask=prepare_inputs.attention_mask,
                            chunk_size=args.chunk_size,
                        )

                    generate_kwargs = dict(
                        inputs_embeds=inputs_embeds,
                        input_ids=prepare_inputs.input_ids,
                        images=prepare_inputs.images,
                        images_seq_mask=prepare_inputs.images_seq_mask,
                        images_spatial_crop=prepare_inputs.images_spatial_crop,
                        attention_mask=prepare_inputs.attention_mask,
                        past_key_values=past_key_values,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                    if args.do_sample:
                        generate_kwargs.update(
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                        )
                    else:
                        generate_kwargs.update(do_sample=False)

                    outputs = vl_gpt.generate(**generate_kwargs)

                answer = tokenizer.decode(
                    outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
                    skip_special_tokens=True,
                )
                answer = clean_answer(answer)

                score = anls_score(answer, gold_answers) if gold_answers else None
                if score is not None:
                    anls_scores.append(score)

                result = {
                    "questionId": question_id,
                    "question": question,
                    "image": entry["image"],
                    "answer": answer,
                    "gold_answers": gold_answers,
                    "anls": score,
                }

            except Exception as e:  # noqa: BLE001 - one bad sample must not kill the whole run
                num_errors += 1
                result = {
                    "questionId": question_id,
                    "question": question,
                    "image": entry.get("image"),
                    "error": str(e),
                }
                if args.verbose:
                    print(f"\n[warn] sample questionId={question_id} failed: {e}", file=sys.stderr)

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if args.verbose and "answer" in result:
                print(f"\nQ: {question}\nA: {result['answer']}  (gold: {gold_answers}, anls: {result['anls']})")

            if anls_scores:
                progress.set_postfix(avg_anls=sum(anls_scores) / len(anls_scores))

    avg_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
    summary = {
        "avg_anls": avg_anls,
        "scored_count": len(anls_scores),
        "total_count": len(entries),
        "errors": num_errors,
    }
    print("\n" + "=" * 60)
    print(f"Average ANLS: {avg_anls:.4f} over {len(anls_scores)}/{len(entries)} scored questions "
          f"({num_errors} errors)")
    print(f"Per-question predictions written to: {args.output_file}")
    print("=" * 60)

    summary_file = os.path.splitext(args.output_file)[0] + ".summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 batch inference + ANLS evaluation on DocVQA")

    parser.add_argument("--model_path", type=str, required=True,
                        help="model name or local path to the model")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="directory containing the DocVQA document images")

    ann_group = parser.add_mutually_exclusive_group(required=True)
    ann_group.add_argument("--annotation_file", type=str, default=None,
                            help="path to a single DocVQA-format json file (e.g. val_v1.0.json)")
    ann_group.add_argument("--questions_dir", type=str, default=None,
                            help="directory containing one or more DocVQA-format json files")

    parser.add_argument("--output_file", type=str, default="docvqa_predictions.jsonl",
                        help="path to write per-question predictions as jsonl")
    parser.add_argument("--num_eval_samples", type=int, default=-1,
                        help="evaluate only the first N questions; -1 means evaluate all")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="max new tokens to generate per answer (DocVQA answers are short)")
    parser.add_argument("--prompt_suffix", type=str,
                        default="Answer the question using a single word or phrase.",
                        help="instruction appended after the question to elicit a short answer")

    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for incremental prefilling (e.g. on a 40G GPU for vl2-small); "
                             "-1 disables incremental prefilling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to run the model on")

    parser.add_argument("--do_sample", action="store_true",
                        help="use sampling instead of greedy decoding (greedy is recommended for evaluation)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed, used only when --do_sample is set")

    parser.add_argument("--verbose", action="store_true",
                        help="print each question/answer pair as it is processed")

    args = parser.parse_args()
    main(args)
