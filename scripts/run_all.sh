#!/usr/bin/env bash
# Full LiteDoc pipeline: data preparation -> pruning + distillation -> evaluation.
#   bash scripts/run_all.sh [40|75|50]
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT=${1:-40}
TEACHER=${TEACHER:-deepseek-ai/deepseek-vl2-tiny}
CKPT=${CKPT:-checkpoints}

bash scripts/prepare_data.sh
bash scripts/train_litedoc.sh "$VARIANT"
MODELS="$TEACHER $CKPT/litedoc-$VARIANT-pruned $CKPT/litedoc-$VARIANT" bash scripts/evaluate_all.sh
