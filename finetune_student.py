"""
Supervised fine-tuning (SFT) of a DeepSeek-VL2 / LiteDoc checkpoint on document-understanding data.

This is the "pruning + fine-tuning" baseline that LiteDoc's S2S distillation is compared against:
the pruned student is trained with the standard next-token cross-entropy on the gold answers
only (no teacher). It can also be used to fine-tune a distilled LiteDoc model further on a
target task.

Example:
    python finetune_student.py \
        --model_path checkpoints/litedoc-40-pruned \
        --train_data data/docvqa_train.jsonl data/funsd_train.jsonl \
        --output_dir checkpoints/litedoc-40-sft \
        --epochs 1 --lr 1e-5 --grad_accum 8
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from distill_student import make_batch, set_trainable
from litedoc import data, moe_utils
from litedoc.evaluation import format_summary, run_evaluation
from litedoc.modeling import load_model, load_processor, resolve_dtype, save_litedoc_model


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of a (pruned) DeepSeek-VL2")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", nargs="+", required=True, help="LiteDoc JSONL files with gold answers")
    parser.add_argument("--num_train_samples", type=int, default=-1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)

    parser.add_argument("--train_modules", choices=["experts", "moe", "language", "all"], default="moe")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp32_master", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--eval_data", nargs="*", default=None)
    parser.add_argument("--num_eval_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def answer_cross_entropy(logits, input_ids, answer_mask):
    """Next-token cross-entropy averaged over the answer tokens (prompt and padding are ignored)."""
    shift_logits = logits[:, :-1].float()
    shift_targets = input_ids[:, 1:]
    shift_mask = answer_mask[:, 1:].to(logits.device)
    loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_targets.reshape(-1),
                           reduction="none").view_as(shift_targets)
    return (loss * shift_mask).sum() / shift_mask.sum().clamp_min(1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = resolve_dtype(args.dtype)
    use_autocast = args.fp32_master and str(args.device).startswith("cuda")
    if args.fp32_master and not use_autocast:
        raise SystemExit("--fp32_master needs a CUDA device (bf16 autocast)")

    processor = load_processor(args.model_path)
    model = load_model(args.model_path, device=args.device, dtype=dtype)
    trainable = set_trainable(model, args.train_modules)
    if args.fp32_master:
        for p in trainable:
            p.data = p.data.float()
    print(f"parameters: {moe_utils.count_parameters(model) / 1e9:.2f}B total, "
          f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable ({args.train_modules})")
    if args.gradient_checkpointing:
        moe_utils.get_language_model(model).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    samples = data.read_many_jsonl(args.train_data)
    random.Random(args.seed).shuffle(samples)
    if args.num_train_samples > 0:
        samples = samples[:args.num_train_samples]
    items = [(s, data.reference_answer(s)) for s in samples]
    items = [(s, a) for s, a in items if a]
    if not items:
        raise SystemExit("no training samples with a gold answer")
    print(f"fine-tuning on {len(items)} samples")

    steps_per_epoch = math.ceil(math.ceil(len(items) / args.batch_size) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    epochs = args.epochs if args.max_steps <= 0 else math.ceil(total_steps / steps_per_epoch)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    model.train()
    if args.train_modules != "all":
        model.vision.eval()
        model.projector.eval()

    step = 0
    micro = 0
    running, n_running = 0.0, 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    with open(os.path.join(args.output_dir, "train_log.jsonl"), "a", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            order = list(range(len(items)))
            random.Random(args.seed + epoch).shuffle(order)
            for b in range(0, len(order), args.batch_size):
                if step >= total_steps:
                    break
                batch, answer_mask = make_batch(processor, [items[i] for i in order[b:b + args.batch_size]],
                                                args.max_seq_len)
                if batch is None:
                    continue
                try:
                    batch = batch.to(args.device, dtype=dtype)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                        out = model(
                            input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                            images=batch.images, images_seq_mask=batch.images_seq_mask,
                            images_spatial_crop=batch.images_spatial_crop, use_cache=False, return_dict=True,
                        )
                        loss = answer_cross_entropy(out.logits, batch.input_ids, answer_mask)
                    (loss / args.grad_accum).backward()
                except torch.cuda.OutOfMemoryError:
                    print("[warn] OOM on a batch, skipping it", file=sys.stderr)
                    optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    torch.cuda.empty_cache()
                    continue

                running += float(loss.detach())
                n_running += 1
                micro += 1
                if micro % args.grad_accum != 0:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == total_steps:
                    entry = {"step": step, "epoch": epoch, "loss": running / max(n_running, 1),
                             "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                             "elapsed_s": time.time() - start}
                    print(f"step {step}/{total_steps} loss={entry['loss']:.4f} lr={entry['lr']:.2e}")
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                    running, n_running = 0.0, 0
                if args.save_every and step % args.save_every == 0:
                    save_litedoc_model(model, args.model_path, os.path.join(args.output_dir, f"step-{step}"),
                                       metadata={"stage": "finetuning", "step": step})

    model.eval()
    if args.fp32_master:
        model.to(dtype)
    save_litedoc_model(model, args.model_path, args.output_dir,
                       metadata={"stage": "finetuned", "init": args.model_path, "steps": step, **vars(args)})
    print(f"fine-tuned checkpoint saved to {args.output_dir}")

    if args.eval_data:
        eval_samples = []
        for path in args.eval_data:
            eval_samples.extend(data.read_jsonl(path, args.num_eval_samples))
        summary = run_evaluation(model, processor, eval_samples,
                                 output_file=os.path.join(args.output_dir, "eval_predictions.jsonl"),
                                 dtype=dtype, seed=args.seed)
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(format_summary(summary))


if __name__ == "__main__":
    main()
