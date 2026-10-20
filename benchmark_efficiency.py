"""
Inference speed and memory of the teacher vs. LiteDoc students (paper Sec. 5.4 / Table 8):
total / active parameters, peak GPU memory and decoding throughput at batch size 1.

Example:
    python benchmark_efficiency.py \
        --model_paths deepseek-ai/deepseek-vl2-tiny checkpoints/litedoc-75 checkpoints/litedoc-50 \
        --data data/docvqa_val.jsonl --num_samples 50 --output results/efficiency.json
"""

import argparse
import gc
import json
import os
import time

import torch

from litedoc import data, moe_utils
from litedoc.modeling import generate_answer, load_model, load_processor, resolve_dtype


def benchmark(model_path, samples, args, dtype):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    processor = load_processor(model_path)
    model = load_model(model_path, device=args.device, dtype=dtype)
    load_memory = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")

    for sample in samples[:args.warmup]:
        generate_answer(model, processor, sample, max_new_tokens=8, dtype=dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    tokens = 0
    elapsed = 0.0
    for sample in samples:
        start = time.perf_counter()
        out = generate_answer(model, processor, sample, max_new_tokens=args.max_new_tokens,
                              repetition_penalty=1.0, dtype=dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        tokens += out["num_new_tokens"]
    peak_memory = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")

    result = {
        "model_path": model_path,
        "params_total_b": moe_utils.count_parameters(model) / 1e9,
        "params_active_b": moe_utils.count_active_parameters(model) / 1e9,
        "experts_per_layer": moe_utils.experts_per_layer(model),
        "weights_memory_gb": load_memory,
        "peak_inference_memory_gb": max(load_memory, peak_memory),
        "generated_tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
    }
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="LiteDoc memory / throughput benchmark")
    parser.add_argument("--model_paths", nargs="+", required=True, help="first entry is used as the baseline")
    parser.add_argument("--data", required=True, help="LiteDoc JSONL file with benchmark prompts")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output", default="results/efficiency.json")
    args = parser.parse_args()

    dtype = resolve_dtype(args.dtype)
    samples = data.read_jsonl(args.data, args.num_samples)
    results = [benchmark(path, samples, args, dtype) for path in args.model_paths]

    base = results[0]
    print(f"\n{'model':<45} {'params':>8} {'active':>8} {'mem GB':>8} {'tok/s':>8} {'mem red.':>9} {'speedup':>8}")
    for r in results:
        mem_red = 100 * (1 - r["peak_inference_memory_gb"] / base["peak_inference_memory_gb"]) \
            if base["peak_inference_memory_gb"] == base["peak_inference_memory_gb"] else float("nan")
        speedup = 100 * (r["tokens_per_second"] / base["tokens_per_second"] - 1) if base["tokens_per_second"] else 0.0
        r["memory_reduction_pct"] = mem_red
        r["throughput_gain_pct"] = speedup
        print(f"{os.path.basename(os.path.normpath(r['model_path'])):<45} {r['params_total_b']:>7.2f}B "
              f"{r['params_active_b']:>7.2f}B {r['peak_inference_memory_gb']:>8.2f} {r['tokens_per_second']:>8.1f} "
              f"{mem_red:>8.1f}% {speedup:>7.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {args.output}")


if __name__ == "__main__":
    main()
