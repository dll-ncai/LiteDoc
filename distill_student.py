"""
LiteDoc Stage 2 - Sparse-to-Sparse (S2S) knowledge distillation (paper Sec. 3.3).

The frozen, unpruned DeepSeek-VL2 teacher heals the pruned student by aligning
  (Phase 1) the hidden states of every decoder layer with an MSE loss, and
  (Phase 2) the next-token distributions with a KL-divergence loss,
combined as  L = alpha * L_KL + (1 - alpha) * L_hidden  (alpha = 0.5).

Each training sequence is  <image(s)> + prompt  followed by an assistant answer, which is the
gold answer of the sample (--answer_source gold), an answer generated once by the teacher
(--answer_source teacher, cached in <output_dir>/teacher_answers.jsonl) or the gold answer
with a teacher fallback for unlabeled samples (--answer_source auto, default).

Example:
    python distill_student.py \
        --teacher_path deepseek-ai/deepseek-vl2-tiny \
        --student_path checkpoints/litedoc-40-pruned \
        --train_data data/docvqa_val.jsonl data/funsd_test.jsonl data/cord_test.jsonl \
        --output_dir checkpoints/litedoc-40 \
        --epochs 1 --lr 1e-5 --grad_accum 8 --alpha 0.5
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import get_cosine_schedule_with_warmup

from litedoc import data, moe_utils
from litedoc.distill import answer_token_mask, build_loss_masks, s2s_distillation_loss
from litedoc.evaluation import format_summary, run_evaluation
from litedoc.modeling import (
    generate_answer,
    load_model,
    load_processor,
    resolve_dtype,
    save_litedoc_model,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="LiteDoc Stage 2: S2S knowledge distillation")
    parser.add_argument("--teacher_path", required=True, help="unpruned DeepSeek-VL2 checkpoint")
    parser.add_argument("--student_path", required=True, help="pruned checkpoint from prune_experts.py")
    parser.add_argument("--train_data", nargs="+", required=True, help="LiteDoc JSONL files")
    parser.add_argument("--num_train_samples", type=int, default=-1, help="-1 = use every sample")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--answer_source", choices=["auto", "gold", "teacher"], default="auto")
    parser.add_argument("--teacher_max_new_tokens", type=int, default=None,
                        help="override the per-sample generation budget when the teacher writes answers")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="skip longer sequences")

    # objective (paper Sec. 3.3 / ablation Sec. 5.3)
    parser.add_argument("--objective", choices=["combined", "output", "layer"], default="combined")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of L_KL; L_hidden gets 1 - alpha")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl_direction", choices=["reverse", "forward"], default="reverse",
                        help="reverse = KL(p_S || p_T) as in Eq. (8)")
    parser.add_argument("--kd_tokens", choices=["text", "answer"], default="text",
                        help="positions whose logits are distilled")
    parser.add_argument("--hidden_normalize", choices=["none", "rms"], default="none")
    parser.add_argument("--hidden_layers", type=int, nargs="*", default=None,
                        help="indices of decoder-layer outputs to align (default: all)")

    # optimization
    parser.add_argument("--train_modules", choices=["experts", "moe", "language", "all"], default="moe",
                        help="experts = routed experts; moe = routed + shared experts + routers; "
                             "language = whole LM; all = LM + vision encoder + projector")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1, help="optimizer steps; overrides --epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp32_master", action="store_true",
                        help="keep trainable weights in fp32 and run the forward pass in bf16 autocast")

    # devices
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="student device")
    parser.add_argument("--teacher_device", default=None, help="teacher device (default: --device)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    # logging / checkpoints / evaluation
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0, help="save a checkpoint every N optimizer steps")
    parser.add_argument("--eval_data", nargs="*", default=None, help="JSONL files evaluated after training")
    parser.add_argument("--num_eval_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_trainable(student, train_modules):
    for p in student.parameters():
        p.requires_grad_(False)
    language = moe_utils.get_language_model(student)
    if train_modules == "all":
        modules = [student]
    elif train_modules == "language":
        modules = [language]
    else:
        modules = []
        for _, moe in moe_utils.iter_moe_layers(student):
            if train_modules == "experts":
                modules.extend(e for e in moe.experts if e is not None)
            else:
                modules.append(moe)
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters selected")
    return trainable


def collect_answers(samples, teacher, processor, args):
    """Attach the assistant answer used for teacher forcing to every sample."""
    cache_path = os.path.join(args.output_dir, "teacher_answers.jsonl")
    cache = {}
    if os.path.isfile(cache_path):
        for row in data.read_jsonl(cache_path):
            cache[row["id"]] = row["answer"]

    prepared = []
    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for sample in tqdm(samples, desc="answers"):
            gold = data.reference_answer(sample)
            if args.answer_source == "gold" or (args.answer_source == "auto" and gold):
                if not gold:
                    continue
                answer = gold
            else:
                answer = cache.get(sample["id"])
                if answer is None:
                    try:
                        answer = generate_answer(
                            teacher, processor, sample,
                            max_new_tokens=args.teacher_max_new_tokens or sample.get("max_new_tokens", 512),
                            repetition_penalty=1.1, dtype=resolve_dtype(args.dtype),
                        )["text"].strip()
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] teacher generation failed for {sample['id']}: {e}", file=sys.stderr)
                        continue
                    cache[sample["id"]] = answer
                    cache_f.write(json.dumps({"id": sample["id"], "answer": answer}, ensure_ascii=False) + "\n")
                    cache_f.flush()
            if answer:
                prepared.append((sample, answer))
    return prepared


def make_batch(processor, items, max_seq_len):
    """
    Tokenize (sample, answer) pairs and right-pad them into one batch. Also returns the mask of
    the answer tokens (the processor may not mask the prompt in `labels`, see `mask_prompt`).
    """
    processed = []
    prompt_lengths = []
    for sample, answer in items:
        images = data.sample_images(sample)
        one = processor.process_one(
            conversations=data.build_conversation(sample, answer),
            images=images,
            inference_mode=False,
            system_prompt="",
        )
        if len(one.input_ids) > max_seq_len:
            continue
        prompt_only = processor.process_one(
            conversations=data.build_conversation(sample),
            images=images,
            inference_mode=True,
            system_prompt="",
        )
        processed.append(one)
        prompt_lengths.append(len(prompt_only.input_ids))
    if not processed:
        return None, None
    batch = processor.batchify(processed, padding="right")
    return batch, answer_token_mask(prompt_lengths, batch.attention_mask)


def forward(model, batch, device, dtype):
    batch = batch.to(device, dtype=dtype)
    return model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        images=batch.images,
        images_seq_mask=batch.images_seq_mask,
        images_spatial_crop=batch.images_spatial_crop,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    ), batch


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = resolve_dtype(args.dtype)
    teacher_device = args.teacher_device or args.device
    use_autocast = args.fp32_master and str(args.device).startswith("cuda")
    if args.fp32_master and not use_autocast:
        raise SystemExit("--fp32_master needs a CUDA student device (bf16 autocast)")

    processor = load_processor(args.student_path)

    print(f"loading frozen teacher from {args.teacher_path} on {teacher_device}")
    teacher = load_model(args.teacher_path, device=teacher_device, dtype=dtype)
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"loading pruned student from {args.student_path} on {args.device}")
    student = load_model(args.student_path, device=args.device, dtype=dtype)
    t_layers = moe_utils.num_decoder_layers(teacher)
    s_layers = moe_utils.num_decoder_layers(student)
    if t_layers != s_layers:
        raise SystemExit(f"teacher has {t_layers} decoder layers but student has {s_layers}")

    trainable = set_trainable(student, args.train_modules)
    if args.fp32_master:
        for p in trainable:
            p.data = p.data.float()
    print(f"student parameters: {moe_utils.count_parameters(student) / 1e9:.2f}B total, "
          f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable ({args.train_modules})")

    if args.gradient_checkpointing:
        moe_utils.get_language_model(student).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    samples = data.read_many_jsonl(args.train_data)
    random.Random(args.seed).shuffle(samples)
    if args.num_train_samples > 0:
        samples = samples[:args.num_train_samples]
    train_items = collect_answers(samples, teacher, processor, args)
    if not train_items:
        raise SystemExit("no training samples with an answer; check --train_data / --answer_source")
    print(f"distilling on {len(train_items)} samples")

    batches_per_epoch = math.ceil(len(train_items) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    epochs = args.epochs if args.max_steps <= 0 else math.ceil(total_steps / steps_per_epoch)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    teacher.eval()
    student.train()
    if args.train_modules != "all":
        # frozen vision tower / projector stay in eval mode
        student.vision.eval()
        student.projector.eval()

    step = 0
    micro = 0
    running = {"loss": 0.0, "kl": 0.0, "hidden_mse": 0.0, "n": 0}
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    with open(log_path, "a", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            order = list(range(len(train_items)))
            random.Random(args.seed + epoch).shuffle(order)
            for b in range(0, len(order), args.batch_size):
                if step >= total_steps:
                    break
                batch, answer_mask = make_batch(
                    processor, [train_items[i] for i in order[b:b + args.batch_size]], args.max_seq_len)
                if batch is None:
                    continue
                try:
                    with torch.no_grad():
                        t_out, _ = forward(teacher, batch, teacher_device, dtype)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                        s_out, s_batch = forward(student, batch, args.device, dtype)
                        logit_mask, hidden_mask = build_loss_masks(
                            s_batch.attention_mask, s_batch.images_seq_mask, args.kd_tokens, answer_mask)
                        losses = s2s_distillation_loss(
                            s_out, t_out, logit_mask, hidden_mask,
                            alpha=args.alpha, temperature=args.temperature, kl_direction=args.kl_direction,
                            hidden_layers=args.hidden_layers, hidden_normalize=args.hidden_normalize,
                            objective=args.objective,
                        )
                    (losses["loss"] / args.grad_accum).backward()
                except torch.cuda.OutOfMemoryError:
                    print("[warn] OOM on a batch, skipping it (lower --max_seq_len or enable "
                          "--gradient_checkpointing)", file=sys.stderr)
                    optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    torch.cuda.empty_cache()
                    continue
                finally:
                    t_out = None

                running["loss"] += float(losses["loss"].detach())
                running["kl"] += float(losses["kl"])
                running["hidden_mse"] += float(losses["hidden_mse"])
                running["n"] += 1
                micro += 1

                if micro % args.grad_accum != 0:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == total_steps:
                    n = max(running["n"], 1)
                    entry = {
                        "step": step, "epoch": epoch,
                        "loss": running["loss"] / n, "kl": running["kl"] / n,
                        "hidden_mse": running["hidden_mse"] / n,
                        "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": time.time() - start,
                    }
                    print(f"step {step}/{total_steps} loss={entry['loss']:.4f} kl={entry['kl']:.4f} "
                          f"hidden_mse={entry['hidden_mse']:.4f} lr={entry['lr']:.2e}")
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                    running = {"loss": 0.0, "kl": 0.0, "hidden_mse": 0.0, "n": 0}

                if args.save_every and step % args.save_every == 0:
                    save_litedoc_model(student, args.student_path, os.path.join(args.output_dir, f"step-{step}"),
                                       metadata={"stage": "distilling", "step": step})

    student.eval()
    if args.fp32_master:
        student.to(dtype)
    metadata = {"stage": "distilled", "teacher": args.teacher_path, "student_init": args.student_path,
                "steps": step, **{k: v for k, v in vars(args).items() if k not in ("teacher_path", "student_path")}}
    save_litedoc_model(student, args.student_path, args.output_dir, metadata=metadata)
    print(f"distilled LiteDoc checkpoint saved to {args.output_dir}")

    if args.eval_data:
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_samples = []
        for path in args.eval_data:
            eval_samples.extend(data.read_jsonl(path, args.num_eval_samples))
        summary = run_evaluation(student, processor, eval_samples,
                                 output_file=os.path.join(args.output_dir, "eval_predictions.jsonl"),
                                 dtype=dtype, seed=args.seed)
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(format_summary(summary))


if __name__ == "__main__":
    main()
