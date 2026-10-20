#!/usr/bin/env bash
# Convert the raw benchmarks into LiteDoc JSONL files.
# Point the *_ROOT variables at your local copies; datasets whose folder is missing are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=${RAW:-/data}
OUT=${OUT:-data}
mkdir -p "$OUT"

DOCVQA_ROOT=${DOCVQA_ROOT:-$RAW/docvqa}
INFOVQA_ROOT=${INFOVQA_ROOT:-$RAW/infovqa}
SROIE_ROOT=${SROIE_ROOT:-$RAW/sroie}
FUNSD_ROOT=${FUNSD_ROOT:-$RAW/funsd/dataset}
DOCILE_ROOT=${DOCILE_ROOT:-$RAW/docile}
KLC_ROOT=${KLC_ROOT:-$RAW/kleister-charity}
WTQ_ROOT=${WTQ_ROOT:-$RAW/WikiTableQuestions}
RVL_ROOT=${RVL_ROOT:-$RAW/rvl-cdip}

# ---- training / calibration splits (used by Stage 1 and Stage 2) ----
if [ -f "$DOCVQA_ROOT/train_v1.0.json" ]; then
  python prepare_data.py docvqa --annotation_file "$DOCVQA_ROOT/train_v1.0.json" \
      --images_dir "$DOCVQA_ROOT" --output "$OUT/docvqa_train.jsonl"
fi
if [ -d "$FUNSD_ROOT/training_data" ]; then
  python prepare_data.py funsd --root "$FUNSD_ROOT/training_data" --output "$OUT/funsd_train.jsonl"
fi
if [ -d "$SROIE_ROOT/train" ]; then
  python prepare_data.py sroie --root "$SROIE_ROOT/train" --output "$OUT/sroie_train.jsonl"
fi
python prepare_data.py cord --source naver-clova-ix/cord-v2 --split train --output "$OUT/cord_train.jsonl" || true

# ---- evaluation splits (paper Table 6) ----
if [ -f "$DOCVQA_ROOT/val_v1.0.json" ]; then
  python prepare_data.py docvqa --annotation_file "$DOCVQA_ROOT/val_v1.0.json" \
      --images_dir "$DOCVQA_ROOT" --output "$OUT/docvqa_val.jsonl"
fi
if [ -f "$INFOVQA_ROOT/infographicsVQA_val_v1.0.json" ]; then
  python prepare_data.py infovqa --annotation_file "$INFOVQA_ROOT/infographicsVQA_val_v1.0.json" \
      --images_dir "$INFOVQA_ROOT" --output "$OUT/infovqa_val.jsonl"
fi
python prepare_data.py cord --source naver-clova-ix/cord-v2 --split test --output "$OUT/cord_test.jsonl" || true
if [ -d "$SROIE_ROOT/test" ]; then
  python prepare_data.py sroie --root "$SROIE_ROOT/test" --output "$OUT/sroie_test.jsonl"
fi
if [ -d "$FUNSD_ROOT/testing_data" ]; then
  python prepare_data.py funsd --root "$FUNSD_ROOT/testing_data" --output "$OUT/funsd_test.jsonl"
fi
if [ -d "$DOCILE_ROOT" ]; then
  python prepare_data.py docile --root "$DOCILE_ROOT" --split val --output "$OUT/docile_val.jsonl"
fi
if [ -d "$KLC_ROOT" ]; then
  python prepare_data.py klc --root "$KLC_ROOT" --split dev-0 --output "$OUT/klc_dev.jsonl"
fi
if [ -d "$WTQ_ROOT" ]; then
  python prepare_data.py wtq --root "$WTQ_ROOT" --output "$OUT/wtq_test.jsonl"
fi
if [ -d "$RVL_ROOT" ]; then
  python prepare_data.py rvl_cdip --root "$RVL_ROOT" --output "$OUT/rvl_cdip_test.jsonl"
fi

ls -l "$OUT"
