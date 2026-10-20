#!/usr/bin/env bash
# Evaluate the teacher and LiteDoc students on every prepared benchmark (paper Table 7)
# and measure memory / throughput (paper Table 8).
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=${TEACHER:-deepseek-ai/deepseek-vl2-tiny}
DATA=${DATA:-data}
CKPT=${CKPT:-checkpoints}
MODELS=${MODELS:-"$TEACHER $CKPT/litedoc-40-pruned $CKPT/litedoc-40"}
EVAL_FILES=$(ls "$DATA"/*_val.jsonl "$DATA"/*_test.jsonl "$DATA"/*_dev.jsonl 2>/dev/null || true)

for model in $MODELS; do
  name=$(basename "$model")
  python evaluate.py --model_path "$model" --data $EVAL_FILES --output_dir "results/$name"
done

python benchmark_efficiency.py --model_paths $MODELS --data "$DATA/docvqa_val.jsonl" \
    --num_samples 100 --output results/efficiency.json
