"""
Evaluate a DeepSeek-VL2 teacher or a LiteDoc student on LiteDoc JSONL benchmarks
(paper Sec. 4.3 / Table 7). Scores: ANLS (DocVQA, InfoVQA), F1 (CORD, SROIE, FUNSD,
DocILE, KLC), accuracy (WTQ, RVL-CDIP).

Example:
    python evaluate.py --model_path checkpoints/litedoc-40 \
        --data data/docvqa_val.jsonl data/funsd_test.jsonl data/cord_test.jsonl \
        --output_dir results/litedoc-40
"""

import argparse
import json
import os

import torch

from litedoc import data
from litedoc.evaluation import format_summary, run_evaluation
from litedoc.modeling import load_model, load_processor, resolve_dtype


def main():
    parser = argparse.ArgumentParser(description="LiteDoc benchmark evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="LiteDoc JSONL files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=-1, help="per file; -1 = all")

    # generation (paper defaults: greedy, repetition_penalty 1.1, max_new_tokens 512 / 128 for VQA)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="override the per-sample budget stored in the JSONL")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="incremental prefilling chunk size for long inputs (-1 = off)")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", default=None, help="e.g. 'auto' to shard over several GPUs")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = resolve_dtype(args.dtype)
    processor = load_processor(args.model_path)
    model = load_model(args.model_path, device=args.device, dtype=dtype, device_map=args.device_map)

    os.makedirs(args.output_dir, exist_ok=True)
    limit = args.num_samples if args.num_samples > 0 else None
    full_summary = {}
    for path in args.data:
        samples = data.read_jsonl(path, limit)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {name}: {len(samples)} samples ===")
        summary = run_evaluation(
            model, processor, samples,
            output_file=os.path.join(args.output_dir, f"{name}.predictions.jsonl"),
            max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty, chunk_size=args.chunk_size,
            dtype=dtype, seed=args.seed, verbose=args.verbose,
        )
        print(format_summary(summary))
        full_summary[name] = summary

    scores = [s[d]["score"] for s in full_summary.values() for d in s if not d.startswith("_") and d != "average"]
    full_summary["average"] = sum(scores) / len(scores) if scores else 0.0
    full_summary["model_path"] = args.model_path
    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\naverage score over {len(scores)} datasets: {full_summary['average']:.2f}")
    print(f"summary written to {out_path}")


if __name__ == "__main__":
    main()
