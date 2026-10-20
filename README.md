# LiteDoc: Distilling Large Document Models into Efficient Task-Specific Encoders

LiteDoc compresses the **DeepSeek-VL2 Tiny** Mixture-of-Experts vision-language model into a smaller,
document-specialized model in two stages:

1. **Stage 1: task-aware expert pruning.** The frozen teacher is run over document-understanding (DU)
   data. For every MoE layer, all routed experts are evaluated on the same tokens and their pairwise
   functional similarity is measured with **Centered Kernel Alignment (CKA)**. A redundancy graph
   (edges = expert pairs with CKA > τ) is built, and one expert of every highly correlated pair is
   removed: the one with the lower token-routing activation. The experts and the matching router rows
   are physically sliced out, so the pruned checkpoint really is smaller in memory.
2. **Stage 2: Sparse-to-Sparse (S2S) knowledge distillation.** The pruned student is healed by the
   frozen, unpruned teacher with
   `L = α · KL(p_S ‖ p_T) + (1 − α) · 1/M Σ_i MSE(H_S^i, H_T^i)` (α = 0.5). The first term aligns the output
   logits and the second aligns the hidden states of every decoder layer.

The repository also contains the baselines and ablations from the paper (frequency / weight-cosine /
weight-magnitude pruning, pruning + fine-tuning, output-only / layer-only distillation, τ sweep), the
evaluation suite for 9 DU benchmarks, and a memory/throughput benchmark.

---

## Repository layout

```
LiteDoc_/
├── deepseek_vl2/               DeepSeek-VL2 model code (extended for pruned checkpoints, see below)
├── litedoc/                    LiteDoc library
│   ├── cka.py                  HSIC / CKA (Eq. 3-4) + fast pairwise linear CKA
│   ├── moe_utils.py            MoE layer discovery, calibration hooks, structural expert pruning
│   ├── pruning.py              pruning criteria: cka | frequency | weight_cosine | weight_magnitude
│   ├── distill.py              S2S losses: logit KL (Eq. 8), hidden-state MSE (Eq. 6-7)
│   ├── data.py                 unified JSONL format, prompts, converters for all benchmarks
│   ├── render.py               PDF -> PNG (PyMuPDF), HTML/CSV table -> PNG (Selenium or PIL)
│   ├── metrics.py              ANLS, token F1, CORD per-field F1, KLC key=value F1, EM, CER/WER
│   ├── evaluation.py           generation + scoring loop
│   └── modeling.py             load / save / generate helpers
├── prepare_data.py             raw benchmark -> LiteDoc JSONL
├── prune_experts.py            Stage 1: expert pruning (training stage 1)
├── distill_student.py          Stage 2: S2S distillation training (training stage 2)
├── finetune_student.py         supervised fine-tuning trainer (pruning + fine-tuning baseline)
├── evaluate.py                 benchmark evaluation (Table 7)
├── benchmark_efficiency.py     parameters, memory, tokens/s (Table 8)
├── inference.py                original DeepSeek-VL2 single-example inference
├── inference_docvqa.py         DocVQA batch inference + ANLS
├── scripts/
│   ├── prepare_data.sh         prepare all datasets
│   ├── train_litedoc.sh        train LiteDoc-40 / LiteDoc-75 / LiteDoc-50 (pruning + distillation)
│   ├── train_baselines.sh      baselines and ablations
│   ├── evaluate_all.sh         evaluate models and benchmark efficiency
│   └── run_all.sh              everything, end to end
```

### Changes to the DeepSeek-VL2 model code

* `DeepseekV2Config` has two new optional fields. `moe_layer_num_experts` (`{layer: #experts}`) lets
  every MoE layer have its own number of routed experts. `moe_layer_kept_experts` records which original
  experts were kept. Pruned checkpoints therefore load with a plain `from_pretrained`.
* `DeepseekV2MoE` / `MoEGate` read the per-layer expert count. In a pruned layer, group-limited routing
  falls back to a single group over the kept experts.
* `noaux_tc` routing is allowed in training mode, which distilling students of larger VL2 variants
  requires.
* The SigLIP attention falls back to `torch.nn.functional.scaled_dot_product_attention` when `xformers`
  is not installed. The math is identical.

Unpruned checkpoints behave exactly as before.

---

## 1. Installation

Python ≥ 3.8 and a CUDA GPU. The paper's experiments use NVIDIA A100 (40 GB / 80 GB) in bfloat16.

```bash
git clone <this repository> LiteDoc_
cd LiteDoc_
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional, faster attention on Ampere+ GPUs
pip install flash-attn --no-build-isolation
```

`transformers==4.38.2` is required, because the DeepSeek-VL2 code targets this version.

The teacher `deepseek-ai/deepseek-vl2-tiny` is downloaded automatically from the HuggingFace Hub the
first time it is used. To use a local copy, pass its directory as `--model_path` / `--teacher_path`.

Memory guideline: distillation keeps the bf16 teacher (~7 GB) and the trainable student on the GPU. One
80 GB A100 is enough with `--gradient_checkpointing --fp32_master`. With two GPUs, put the teacher on the
second one with `--teacher_device cuda:1`.

---

## 2. Data preparation

Every benchmark is converted once into a JSON-lines file in the following format. Pruning, distillation,
fine-tuning and evaluation all read this same format:

```json
{"id": "docvqa-49153", "dataset": "docvqa", "task": "vqa", "metric": "anls",
 "images": ["/abs/path/page.png"], "prompt": "<image>\nWhat is the date?\nAnswer the question using a single word or phrase.",
 "answers": ["12/03/1998"], "max_new_tokens": 128}
```

Download the raw datasets from their official sources, then run one command per dataset:

| Dataset | Command | Raw layout expected |
|---|---|---|
| DocVQA | `python prepare_data.py docvqa --annotation_file DocVQA/val_v1.0.json --images_dir DocVQA --output data/docvqa_val.jsonl` | official json + `documents/` |
| InfographicsVQA | `python prepare_data.py infovqa --annotation_file infographicsVQA_val_v1.0.json --images_dir infovqa/images --output data/infovqa_val.jsonl` | official json + images |
| CORD | `python prepare_data.py cord --source naver-clova-ix/cord-v2 --split test --output data/cord_test.jsonl` | HF dataset id or local copy |
| SROIE | `python prepare_data.py sroie --root SROIE/test --output data/sroie_test.jsonl` | `img/*.jpg`, `entities/*.txt` |
| FUNSD | `python prepare_data.py funsd --root FUNSD/dataset/testing_data --output data/funsd_test.jsonl` | `images/`, `annotations/` |
| DocILE | `python prepare_data.py docile --root docile --split val --output data/docile_val.jsonl` | `val.json`, `annotations/`, `pdfs/` (rendered @150 DPI) |
| Kleister Charity | `python prepare_data.py klc --root kleister-charity --split dev-0 --output data/klc_dev.jsonl` | `dev-0/in.tsv`, `dev-0/expected.tsv`, `documents/*.pdf` (run `git annex get documents` first; rendered @300 DPI, all pages as multiple `<image>` tags) |
| WikiTableQuestions | `python prepare_data.py wtq --root WikiTableQuestions --output data/wtq_test.jsonl` | `data/pristine-unseen-tables.tsv`, `csv/` (tables rendered with Selenium/Chrome; `--renderer pil` for the built-in renderer) |
| RVL-CDIP | `python prepare_data.py rvl_cdip --root rvl-cdip --output data/rvl_cdip_test.jsonl` | `labels/test.txt`, `images/` |

Useful options are `--max_samples N` and `--shuffle --seed 42`. Training splits are produced the same
way, for example DocVQA `train_v1.0.json`, FUNSD `training_data`, CORD `--split train` and SROIE `train/`.

To convert everything at once, set the dataset roots and run:

```bash
RAW=/path/to/raw/datasets bash scripts/prepare_data.sh      # writes data/*.jsonl
```

The prompts follow the paper (Sec. 4.3):

* CORD: JSON-enforcing extraction prompt.
* SROIE: natural-language entity prompt.
* FUNSD and DocILE: `<|grounding|>Extract all form fields and their labels from this document`.
* KLC: `key=value` extraction over all pages.
* RVL-CDIP: classification prompt that lists all 16 classes.
* DocVQA, InfoVQA and WTQ: `<image>\n` + question.

---

## 3. Training

### Quick start: train LiteDoc

```bash
bash scripts/train_litedoc.sh 40     # LiteDoc-40: 40% of routed experts pruned (main results)
bash scripts/train_litedoc.sh 75     # LiteDoc-75: tau = 0.7, keeps ~75% of experts (48 / layer)
bash scripts/train_litedoc.sh 50     # LiteDoc-50: tau = 0.5, keeps ~50% of experts (32 / layer)
```

Each run writes `checkpoints/litedoc-<V>-pruned` (after Stage 1) and `checkpoints/litedoc-<V>`
(after Stage 2). The data files, devices and sample counts can be overridden with environment variables
such as `TRAIN_FILES`, `CALIB_FILES`, `EVAL_FILES`, `TEACHER_DEVICE`, `STUDENT_DEVICE` and `NUM_TRAIN`.

The two stages can also be run by hand, as described below.

### Stage 1: CKA-based expert pruning

```bash
python prune_experts.py \
    --model_path deepseek-ai/deepseek-vl2-tiny \
    --calib_data data/docvqa_train.jsonl data/funsd_train.jsonl \
    --num_calib_samples 256 \
    --strategy cka --prune_ratio 0.4 \
    --cka_tokens 2048 \
    --output_dir checkpoints/litedoc-40-pruned
```

| Option | Meaning |
|---|---|
| `--strategy` | `cka` (LiteDoc), `frequency` (token-routing frequency), `weight_cosine` (Eq. 2), `weight_magnitude` |
| `--cka_threshold τ` | prune one expert of every pair with CKA > τ (paper: 0.7 → LiteDoc-75, 0.5 → LiteDoc-50) |
| `--prune_ratio r` | cap and target: remove a fraction `r` of the routed experts |
| `--budget per_layer \| global` | uniform pruning per layer, or a joint budget over all layers (layer-dependent retention, as in Fig. 9) |
| `--protect_top_fraction f` | never prune the most-activated fraction `f` of a layer's experts |
| `--min_keep` | minimum experts kept per layer (default: router top-k = 6) |
| `--include_last_layer` | also prune the final decoder block. By default it is kept intact, as in the paper |
| `--cka_tokens n` | calibration tokens per layer for CKA (Gram matrices are n × n) |
| `--stats_in file` | reuse `pruning_stats.pt` from an earlier run, which skips calibration (handy for τ sweeps) |

Outputs in `--output_dir`:

* the pruned checkpoint (weights, config with per-layer expert counts, tokenizer and processor files);
* `pruning_report.json` with the experts kept per layer, their original indices and the parameter
  counts before and after;
* `pruning_stats.pt` with the CKA matrices, routing frequencies and token counts per layer, which are
  the data behind Fig. 2, 3 and 9.

### Stage 2: Sparse-to-Sparse knowledge distillation

```bash
python distill_student.py \
    --teacher_path deepseek-ai/deepseek-vl2-tiny \
    --student_path checkpoints/litedoc-40-pruned \
    --train_data data/docvqa_train.jsonl data/funsd_train.jsonl data/cord_train.jsonl data/sroie_train.jsonl \
    --num_train_samples 20000 \
    --objective combined --alpha 0.5 --kl_direction reverse \
    --train_modules moe \
    --epochs 1 --batch_size 1 --grad_accum 16 --lr 1e-5 \
    --gradient_checkpointing --fp32_master \
    --teacher_device cuda:0 --device cuda:0 \
    --eval_data data/docvqa_val.jsonl --num_eval_samples 500 \
    --output_dir checkpoints/litedoc-40
```

| Option | Meaning |
|---|---|
| `--objective combined \| output \| layer` | α·KL + (1−α)·MSE (paper), KL only, or MSE only (ablation, Sec. 5.3) |
| `--alpha` | weight of the logit KL term (paper: 0.5) |
| `--kl_direction reverse \| forward` | `reverse` = KL(p_S ‖ p_T) exactly as in Eq. 8, `forward` = Hinton KD |
| `--temperature` | KD softmax temperature |
| `--kd_tokens text \| answer` | distill the logits of all text positions, or only of the answer tokens |
| `--hidden_normalize none \| rms` | plain MSE (paper), or MSE on teacher-RMS-normalized states |
| `--answer_source auto \| gold \| teacher` | assistant turn = gold answer, teacher-generated answer (cached in `teacher_answers.jsonl`), or gold with teacher fallback |
| `--train_modules experts \| moe \| language \| all` | what the student updates (default: routed + shared experts and routers) |
| `--fp32_master` | fp32 trainable weights with bf16 autocast |
| `--gradient_checkpointing` | trade compute for activation memory |
| `--save_every N` | intermediate checkpoints `step-N/` |

Training metrics (loss, KL, hidden MSE, grad norm, lr) are written to `train_log.jsonl`. The final
student is saved to `--output_dir`, which can be loaded like any DeepSeek-VL2 checkpoint.

### Baseline: pruning + supervised fine-tuning

This is the same data and optimizer as distillation, but the loss is plain cross-entropy on the gold
answers and there is no teacher:

```bash
python finetune_student.py \
    --model_path checkpoints/litedoc-40-pruned \
    --train_data data/docvqa_train.jsonl data/funsd_train.jsonl data/cord_train.jsonl data/sroie_train.jsonl \
    --num_train_samples 20000 --train_modules moe \
    --epochs 1 --grad_accum 16 --lr 1e-5 --gradient_checkpointing --fp32_master \
    --output_dir checkpoints/baseline-40-sft
```

`finetune_student.py` can also fine-tune a distilled LiteDoc model further on a single target task.

### Ablations (Sec. 5.3)

```bash
bash scripts/train_baselines.sh
```

This script runs, in order:

* pruning + fine-tuning;
* the frequency, weight-magnitude and weight-cosine pruning criteria at the same 40% budget, each
  followed by the same distillation;
* output-only and layer-only distillation;
* a CKA threshold sweep with τ ∈ {0.4, 0.5, 0.6, 0.7}.

---

## 4. Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/litedoc-40 \
    --data data/docvqa_val.jsonl data/infovqa_val.jsonl data/cord_test.jsonl data/sroie_test.jsonl \
           data/funsd_test.jsonl data/docile_val.jsonl data/klc_dev.jsonl data/wtq_test.jsonl data/rvl_cdip_test.jsonl \
    --output_dir results/litedoc-40
```

Generation follows the paper: greedy decoding, `repetition_penalty=1.1`, `max_new_tokens=512`, and 128
for VQA (stored per sample). For long multi-page inputs, add `--chunk_size 512` (incremental
prefilling). For multi-GPU sharding, add `--device_map auto`.

| Metric | Datasets |
|---|---|
| ANLS (threshold 0.5, max over gold answers) | DocVQA, InfoVQA (WTQ also reports it) |
| bag-of-tokens F1 | SROIE, FUNSD, DocILE |
| per-field macro F1 on the parsed JSON (+ per-field CER/WER) | CORD |
| upper-cased key=value F1 | Kleister Charity |
| exact match / accuracy | WikiTableQuestions, RVL-CDIP |

Outputs:

* `<dataset>.predictions.jsonl` with the per-sample prediction, raw output and scores;
* `summary.json` with the per-dataset scores (×100) and their average.

Evaluate the teacher, the pruned student and the distilled student together, and measure efficiency:

```bash
MODELS="deepseek-ai/deepseek-vl2-tiny checkpoints/litedoc-40-pruned checkpoints/litedoc-40" \
    bash scripts/evaluate_all.sh
```

### Inference speed and memory (Table 8)

```bash
python benchmark_efficiency.py \
    --model_paths deepseek-ai/deepseek-vl2-tiny checkpoints/litedoc-75 checkpoints/litedoc-50 \
    --data data/docvqa_val.jsonl --num_samples 100 --output results/efficiency.json
```

For each model, this reports total and active parameters, experts per layer, weight memory, peak
inference memory and decoding tokens/s at batch size 1. Memory reduction and speed-up are given
relative to the first model.

### Single-document inference with a LiteDoc model

LiteDoc checkpoints work with the original scripts:

```bash
python inference_docvqa.py --model_path checkpoints/litedoc-40 \
    --images_dir /data/docvqa --annotation_file /data/docvqa/val_v1.0.json --num_eval_samples 200
```

They also load from Python:

```python
from litedoc.modeling import load_model, load_processor, generate_answer

processor = load_processor("checkpoints/litedoc-40")
model = load_model("checkpoints/litedoc-40", device="cuda")
sample = {"images": ["invoice.png"], "prompt": "<image>\nWhat is the invoice total?"}
print(generate_answer(model, processor, sample, max_new_tokens=64)["text"])
```

---

## 5. Everything end to end

```bash
RAW=/path/to/raw/datasets bash scripts/run_all.sh 40
```

This prepares the data, prunes and distils LiteDoc-40, and evaluates the teacher, the pruned model and
LiteDoc-40 on all benchmarks, followed by the efficiency benchmark.

---

## Acknowledgements

The model code is based on [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2). Use of the
DeepSeek-VL2 weights is subject to the DeepSeek model license.
