#!/usr/bin/env bash
# Train a LiteDoc student: Stage 1 (CKA expert pruning) + Stage 2 (S2S distillation).
#   bash scripts/train_litedoc.sh        # LiteDoc-40 (main results: 40% of routed experts pruned)
#   bash scripts/train_litedoc.sh 75     # LiteDoc-75 (tau = 0.7, keeps ~75% of experts)
#   bash scripts/train_litedoc.sh 50     # LiteDoc-50 (tau = 0.5, keeps ~50% of experts)
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT=${1:-40}
TEACHER=${TEACHER:-deepseek-ai/deepseek-vl2-tiny}
DATA=${DATA:-data}
CKPT=${CKPT:-checkpoints}
CALIB_FILES=${CALIB_FILES:-"$DATA/docvqa_train.jsonl $DATA/funsd_train.jsonl"}
TRAIN_FILES=${TRAIN_FILES:-"$DATA/docvqa_train.jsonl $DATA/funsd_train.jsonl $DATA/cord_train.jsonl $DATA/sroie_train.jsonl"}
EVAL_FILES=${EVAL_FILES:-"$DATA/docvqa_val.jsonl"}
NUM_TRAIN=${NUM_TRAIN:-20000}
TEACHER_DEVICE=${TEACHER_DEVICE:-cuda:0}
STUDENT_DEVICE=${STUDENT_DEVICE:-cuda:0}

case "$VARIANT" in
  40) PRUNE_ARGS="--prune_ratio 0.4" ;;
  75) PRUNE_ARGS="--cka_threshold 0.7 --prune_ratio 0.25" ;;
  50) PRUNE_ARGS="--cka_threshold 0.5 --prune_ratio 0.5" ;;
  *)  echo "unknown variant: $VARIANT (use 40, 75 or 50)"; exit 1 ;;
esac

# ---- Stage 1: CKA-based expert pruning ----
python prune_experts.py \
    --model_path "$TEACHER" \
    --calib_data $CALIB_FILES \
    --num_calib_samples 256 \
    --strategy cka $PRUNE_ARGS \
    --cka_tokens 2048 \
    --output_dir "$CKPT/litedoc-$VARIANT-pruned"

# ---- Stage 2: Sparse-to-Sparse knowledge distillation ----
python distill_student.py \
    --teacher_path "$TEACHER" \
    --student_path "$CKPT/litedoc-$VARIANT-pruned" \
    --train_data $TRAIN_FILES \
    --num_train_samples "$NUM_TRAIN" \
    --objective combined --alpha 0.5 --temperature 1.0 --kl_direction reverse \
    --train_modules moe \
    --epochs 1 --batch_size 1 --grad_accum 16 --lr 1e-5 --warmup_ratio 0.03 \
    --gradient_checkpointing --fp32_master \
    --device "$STUDENT_DEVICE" --teacher_device "$TEACHER_DEVICE" \
    --save_every 500 \
    --eval_data $EVAL_FILES --num_eval_samples 500 \
    --output_dir "$CKPT/litedoc-$VARIANT"
