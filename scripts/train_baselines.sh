#!/usr/bin/env bash
# Baselines and ablations of the paper (Sec. 5.3):
#   1. pruning + supervised fine-tuning (no teacher)
#   2. pruning criteria: token-routing frequency / weight magnitude / weight cosine vs. CKA
#   3. distillation objective: output-only / layer-only vs. combined
#   4. CKA threshold sweep, tau in {0.4, 0.5, 0.6, 0.7}
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=${TEACHER:-deepseek-ai/deepseek-vl2-tiny}
DATA=${DATA:-data}
CKPT=${CKPT:-checkpoints}
CALIB_FILES=${CALIB_FILES:-"$DATA/docvqa_train.jsonl $DATA/funsd_train.jsonl"}
TRAIN_FILES=${TRAIN_FILES:-"$DATA/docvqa_train.jsonl $DATA/funsd_train.jsonl $DATA/cord_train.jsonl $DATA/sroie_train.jsonl"}
NUM_TRAIN=${NUM_TRAIN:-20000}
COMMON_TRAIN="--train_modules moe --epochs 1 --batch_size 1 --grad_accum 16 --lr 1e-5 --gradient_checkpointing --fp32_master"

# CKA pruning at 40% (its calibration statistics are reused below)
if [ ! -f "$CKPT/litedoc-40-pruned/pruning_stats.pt" ]; then
  python prune_experts.py --model_path "$TEACHER" --calib_data $CALIB_FILES --num_calib_samples 256 \
      --strategy cka --prune_ratio 0.4 --output_dir "$CKPT/litedoc-40-pruned"
fi

# 1. pruning + supervised fine-tuning baseline
python finetune_student.py --model_path "$CKPT/litedoc-40-pruned" --train_data $TRAIN_FILES \
    --num_train_samples "$NUM_TRAIN" $COMMON_TRAIN --output_dir "$CKPT/baseline-40-sft"

# 2. pruning criteria at the same 40% budget, each followed by the same distillation
python prune_experts.py --model_path "$TEACHER" --stats_in "$CKPT/litedoc-40-pruned/pruning_stats.pt" \
    --strategy frequency --prune_ratio 0.4 --output_dir "$CKPT/ablation-frequency-40-pruned"
python prune_experts.py --model_path "$TEACHER" --strategy weight_magnitude --prune_ratio 0.4 \
    --output_dir "$CKPT/ablation-magnitude-40-pruned"
python prune_experts.py --model_path "$TEACHER" --calib_data $CALIB_FILES --num_calib_samples 256 \
    --strategy weight_cosine --prune_ratio 0.4 --output_dir "$CKPT/ablation-cosine-40-pruned"
for crit in frequency magnitude cosine; do
  python distill_student.py --teacher_path "$TEACHER" --student_path "$CKPT/ablation-$crit-40-pruned" \
      --train_data $TRAIN_FILES --num_train_samples "$NUM_TRAIN" $COMMON_TRAIN \
      --output_dir "$CKPT/ablation-$crit-40"
done

# 3. distillation objective
for objective in output layer; do
  python distill_student.py --teacher_path "$TEACHER" --student_path "$CKPT/litedoc-40-pruned" \
      --train_data $TRAIN_FILES --num_train_samples "$NUM_TRAIN" $COMMON_TRAIN \
      --objective "$objective" --output_dir "$CKPT/ablation-kd-$objective-40"
done

# 4. CKA threshold sweep (pruning only, from cached statistics)
for tau in 0.4 0.5 0.6 0.7; do
  python prune_experts.py --model_path "$TEACHER" --stats_in "$CKPT/litedoc-40-pruned/pruning_stats.pt" \
      --strategy cka --cka_threshold "$tau" --output_dir "$CKPT/ablation-tau-$tau-pruned"
done
