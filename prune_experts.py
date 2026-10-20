"""
LiteDoc Stage 1 - task-aware expert pruning of DeepSeek-VL2 (paper Sec. 3.1-3.2).

1. Run the frozen teacher over document-understanding calibration samples (e.g. DocVQA + FUNSD)
   and record, for every MoE layer, the routing frequency of each routed expert and a random
   sample of the MoE-layer input tokens.
2. Evaluate all routed experts on the same tokens and compute the pairwise CKA matrix.
3. Build the redundancy graph (edges: CKA > tau) and drop one expert of every highly
   correlated pair (the less activated one), optionally up to a target prune ratio.
4. Slice the experts and router rows out of the model and save a loadable pruned checkpoint.

Examples:
    # LiteDoc-75 (tau = 0.7) / LiteDoc-50 (tau = 0.5)
    python prune_experts.py --model_path deepseek-ai/deepseek-vl2-tiny \
        --calib_data data/docvqa_val.jsonl data/funsd_test.jsonl --num_calib_samples 128 \
        --strategy cka --cka_threshold 0.7 --output_dir checkpoints/litedoc-75-pruned

    # fixed 40 % expert pruning (main results of the paper)
    python prune_experts.py --model_path deepseek-ai/deepseek-vl2-tiny \
        --calib_data data/docvqa_val.jsonl data/funsd_test.jsonl \
        --strategy cka --prune_ratio 0.4 --output_dir checkpoints/litedoc-40-pruned
"""

import argparse
import json
import os
import random
import sys
import time

import torch

from litedoc import cka as cka_lib
from litedoc import data
from litedoc import moe_utils, pruning
from litedoc.modeling import (
    load_model,
    load_processor,
    model_device,
    prepare_sample_inputs,
    resolve_dtype,
    save_litedoc_model,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="LiteDoc Stage 1: CKA-based MoE expert pruning")
    parser.add_argument("--model_path", required=True, help="teacher checkpoint (HF id or local dir)")
    parser.add_argument("--output_dir", required=True, help="where the pruned checkpoint is written")
    parser.add_argument("--calib_data", nargs="+", default=None, help="LiteDoc JSONL calibration files")
    parser.add_argument("--num_calib_samples", type=int, default=128,
                        help="total number of calibration samples drawn from --calib_data")
    parser.add_argument("--calib_with_answers", action="store_true",
                        help="append the gold answer as assistant turn during calibration")

    parser.add_argument("--strategy", choices=pruning.STRATEGIES, default="cka")
    parser.add_argument("--cka_threshold", type=float, default=None,
                        help="tau: prune one expert of every pair with similarity > tau (paper: 0.7 / 0.5)")
    parser.add_argument("--prune_ratio", type=float, default=None,
                        help="fraction of routed experts to remove (e.g. 0.25, 0.4, 0.5)")
    parser.add_argument("--budget", choices=["per_layer", "global"], default="per_layer",
                        help="apply --prune_ratio uniformly to every layer or to all layers jointly")
    parser.add_argument("--min_keep", type=int, default=None,
                        help="minimum experts kept per layer (default: router top-k)")
    parser.add_argument("--protect_top_fraction", type=float, default=0.0,
                        help="never prune the most-activated fraction of experts in a layer")
    parser.add_argument("--include_last_layer", action="store_true",
                        help="also prune the final decoder block (the paper keeps it intact)")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="explicit decoder layer indices to prune (default: all MoE layers)")

    parser.add_argument("--cka_tokens", type=int, default=1024,
                        help="calibration tokens per layer used for CKA (Gram matrices are n x n)")
    parser.add_argument("--cka_expert_chunk", type=int, default=None,
                        help="experts per block when computing CKA (lower if out of memory)")
    parser.add_argument("--stats_in", default=None,
                        help="reuse statistics saved by a previous run (skips calibration)")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", default=None, help="e.g. 'auto' to shard the teacher over GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_save_model", action="store_true",
                        help="only compute statistics and the pruning plan")
    return parser.parse_args()


def select_layers(model, args):
    moe_layers = [idx for idx, _ in moe_utils.iter_moe_layers(model)]
    if args.layers:
        missing = set(args.layers) - set(moe_layers)
        if missing:
            raise ValueError(f"layers {sorted(missing)} are not MoE layers (MoE layers: {moe_layers})")
        return sorted(args.layers)
    last = moe_utils.num_decoder_layers(model) - 1
    if not args.include_last_layer:
        moe_layers = [idx for idx in moe_layers if idx != last]
    return moe_layers


@torch.no_grad()
def calibrate(model, processor, samples, layers, args, dtype):
    collector = moe_utils.ExpertStatsCollector(
        model, layer_indices=layers, max_tokens_per_layer=args.cka_tokens,
        collect_inputs=args.strategy == "cka", seed=args.seed,
    )
    device = model_device(model)
    failures = 0
    try:
        for sample in tqdm(samples, desc="calibration"):
            try:
                answer = data.reference_answer(sample) if args.calib_with_answers else None
                inputs = prepare_sample_inputs(processor, sample, answer=answer, device=device, dtype=dtype)
                collector.set_token_mask(inputs.attention_mask)
                model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    images=inputs.images,
                    images_seq_mask=inputs.images_seq_mask,
                    images_spatial_crop=inputs.images_spatial_crop,
                    use_cache=False,
                    return_dict=True,
                )
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"[warn] calibration sample {sample.get('id')} failed: {e}", file=sys.stderr)
    finally:
        collector.remove()
    if failures == len(samples):
        raise RuntimeError("every calibration sample failed; check the data paths")
    return collector


@torch.no_grad()
def compute_statistics(model, processor, layers, args, dtype):
    stats = {"layers": layers, "strategy_inputs": args.strategy, "activation": {}, "token_counts": {},
             "gate_mass": {}, "similarity": {}, "magnitude": {}}
    layer_modules = dict(moe_utils.iter_moe_layers(model))

    needs_calibration = args.strategy in ("cka", "frequency", "weight_cosine")
    if needs_calibration:
        if not args.calib_data:
            raise ValueError(f"--calib_data is required for strategy '{args.strategy}'")
        samples = data.read_many_jsonl(args.calib_data)
        random.Random(args.seed).shuffle(samples)
        samples = samples[:args.num_calib_samples]
        print(f"calibrating on {len(samples)} samples from {args.calib_data}")
        collector = calibrate(model, processor, samples, layers, args, dtype)
        for idx in layers:
            stats["activation"][idx] = collector.routing_frequency(idx).float()
            stats["token_counts"][idx] = collector.token_counts[idx].clone()
            stats["gate_mass"][idx] = collector.gate_mass[idx].float()
    else:
        for idx in layers:
            stats["activation"][idx] = torch.ones(len(layer_modules[idx].experts))

    for idx in tqdm(layers, desc="expert similarity"):
        moe = layer_modules[idx]
        if args.strategy == "cka":
            inputs = collector.inputs[idx]
            if inputs is None or inputs.shape[0] < 2:
                raise RuntimeError(f"layer {idx}: not enough calibration tokens for CKA")
            features = moe_utils.compute_expert_outputs(moe, inputs)
            stats["similarity"][idx] = cka_lib.pairwise_linear_cka(
                features, expert_chunk=args.cka_expert_chunk).cpu()
            del features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif args.strategy == "weight_cosine":
            stats["similarity"][idx] = pruning.weight_cosine_similarity(moe)
        elif args.strategy == "weight_magnitude":
            stats["magnitude"][idx] = pruning.weight_magnitude(moe)
    return stats


def describe_stats(stats):
    for idx in stats["layers"]:
        line = f"  layer {idx:>2}:"
        counts = stats["token_counts"].get(idx)
        if counts is not None and counts.sum() > 0:
            line += f" tokens/expert max={int(counts.max()):,} min={int(counts.min()):,}"
        sim = stats["similarity"].get(idx)
        if sim is not None:
            off = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
            line += f" | similarity mean={off.mean():.3f} max={off.max():.3f}"
        print(line)


def main():
    args = parse_args()
    if args.cka_threshold is None and args.prune_ratio is None:
        raise SystemExit("set --cka_threshold and/or --prune_ratio")
    if args.strategy in ("frequency", "weight_magnitude") and args.prune_ratio is None:
        raise SystemExit(f"strategy '{args.strategy}' needs --prune_ratio")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = resolve_dtype(args.dtype)

    print(f"loading teacher from {args.model_path}")
    processor = load_processor(args.model_path)
    model = load_model(args.model_path, device=args.device, dtype=dtype, device_map=args.device_map)
    for p in model.parameters():
        p.requires_grad_(False)

    layers = select_layers(model, args)
    top_k = moe_utils.get_language_config(model).num_experts_per_tok
    min_keep = max(args.min_keep or top_k, top_k)
    params_before = moe_utils.count_parameters(model)
    active_before = moe_utils.count_active_parameters(model)
    print(f"MoE layers to prune: {layers} (top-k={top_k}, min_keep={min_keep})")

    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, "pruning_stats.pt")
    start = time.time()
    if args.stats_in:
        stats = torch.load(args.stats_in, map_location="cpu")
        if stats.get("strategy_inputs") != args.strategy and args.strategy in ("cka", "weight_cosine"):
            raise SystemExit(f"{args.stats_in} holds '{stats.get('strategy_inputs')}' statistics, "
                             f"not '{args.strategy}'")
        layers = [idx for idx in layers if idx in stats["activation"]]
        print(f"loaded statistics from {args.stats_in}")
    else:
        stats = compute_statistics(model, processor, layers, args, dtype)
        torch.save(stats, stats_path)
        print(f"statistics saved to {stats_path} ({time.time() - start:.0f}s)")
    describe_stats({**stats, "layers": layers})

    keep_map = pruning.select_experts(
        args.strategy, layers,
        activation_per_layer=stats["activation"],
        similarity_per_layer=stats["similarity"],
        magnitude_per_layer=stats["magnitude"],
        threshold=args.cka_threshold,
        prune_ratio=args.prune_ratio,
        min_keep=min_keep,
        budget=args.budget,
        protect_top_fraction=args.protect_top_fraction,
    )
    moe_utils.apply_expert_pruning(model, keep_map)

    params_after = moe_utils.count_parameters(model)
    active_after = moe_utils.count_active_parameters(model)
    experts_before = sum(len(stats["activation"][i]) for i in layers)
    experts_after = sum(len(k) for k in keep_map.values())
    report = {
        "teacher": args.model_path,
        "strategy": args.strategy,
        "cka_threshold": args.cka_threshold,
        "prune_ratio": args.prune_ratio,
        "budget": args.budget,
        "min_keep": min_keep,
        "num_calib_samples": args.num_calib_samples,
        "cka_tokens": args.cka_tokens,
        "pruned_layers": layers,
        "experts_per_layer": {str(i): len(keep_map[i]) for i in layers},
        "kept_experts": moe_utils.get_language_config(model).moe_layer_kept_experts,
        "experts_removed_fraction": 1.0 - experts_after / max(experts_before, 1),
        "params_total_before": params_before,
        "params_total_after": params_after,
        "params_active_before": active_before,
        "params_active_after": active_after,
    }
    print("\nexperts kept per layer: " + ", ".join(f"{i}:{len(keep_map[i])}" for i in layers))
    print(f"routed experts removed: {100 * report['experts_removed_fraction']:.1f}%")
    print(f"parameters: {params_before / 1e9:.2f}B -> {params_after / 1e9:.2f}B "
          f"(active {active_before / 1e9:.2f}B -> {active_after / 1e9:.2f}B)")

    with open(os.path.join(args.output_dir, "pruning_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.no_save_model:
        print("--no_save_model set: pruned checkpoint not written")
        return
    save_litedoc_model(model, args.model_path, args.output_dir, metadata={"stage": "pruned", **report})
    print(f"pruned checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
