"""
Unified LiteDoc sample format and benchmark converters (paper Table 6 / Sec. 4.3).

Every benchmark is converted once into a JSON-lines file in which each line is:

    {
      "id":        "docvqa-49153",
      "dataset":   "docvqa",
      "task":      "vqa" | "kie" | "cls" | "tableqa",
      "metric":    "anls" | "token_f1" | "cord_f1" | "key_value_f1" | "accuracy" | "wtq",
      "images":    ["/abs/path/page.png", ...],
      "prompt":    "<image>\nWhat is the date?",       # one <image> tag per image
      "answers":   ["12/03/1998", ...],                # gold strings (VQA / CLS / WTQ / bag-of-tokens KIE)
      "target":    {...},                              # optional structured gold (CORD / SROIE / KLC)
      "label_set": [...],                              # optional, classification labels
      "max_new_tokens": 128                            # optional per-sample generation budget
    }

The same file feeds Stage 1 (CKA calibration), Stage 2 (distillation) and the evaluation script.
"""

import csv
import glob
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .render import load_image, render_pdf, render_table, table_rows_from_csv

# ---------------------------------------------------------------------------
# Prompts (paper Sec. 4.3)
# ---------------------------------------------------------------------------

VQA_SUFFIX = "Answer the question using a single word or phrase."

CORD_PROMPT = (
    "Extract receipt fields as JSON with keys:\n"
    '{"menu": [{"nm": "...", "cnt": "...", "unitprice": "...", "price": "..."}],\n'
    ' "sub_total": {"subtotal_price": "...", "tax_price": "..."},\n'
    ' "total": {"total_price": "...", "cashprice": "...", "changeprice": "..."}}'
)

SROIE_PROMPT = "Extract the company name, date, address, and total amount from this receipt."

FORM_PROMPT = "<|grounding|>Extract all form fields and their labels from this document"

KLC_KEYS = [
    "address__post_town",
    "address__postcode",
    "address__street_line",
    "charity_name",
    "charity_number",
    "income_annually_in_british_pounds",
    "report_date",
    "spending_annually_in_british_pounds",
]

KLC_PROMPT = (
    "Extract the following information from this charity annual report and answer with one "
    "`key=value` line per field: " + ", ".join(KLC_KEYS) + "."
)

RVL_CDIP_CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement", "scientific report",
    "scientific publication", "specification", "file folder", "news article", "budget",
    "invoice", "presentation", "questionnaire", "resume", "memo",
]

RVL_CDIP_PROMPT = (
    "Classify this document into exactly one category from: "
    + ", ".join(RVL_CDIP_CLASSES)
    + ". Respond with only the category name."
)

DATASETS = ("docvqa", "infovqa", "cord", "sroie", "funsd", "docile", "klc", "wtq", "rvl_cdip")


def image_prompt(text: str, num_images: int = 1) -> str:
    return "<image>\n" * num_images + text


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and 0 < limit <= len(samples):
                break
    return samples


def write_jsonl(samples: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_many_jsonl(paths: Iterable[str], limit_per_file: Optional[int] = None) -> List[Dict[str, Any]]:
    samples = []
    for path in paths:
        samples.extend(read_jsonl(path, limit_per_file))
    return samples


def sample_images(sample: Dict[str, Any]):
    return [load_image(p) for p in sample.get("images", [])]


def build_conversation(sample: Dict[str, Any], answer: Optional[str] = None) -> List[Dict[str, Any]]:
    """DeepSeek-VL2 chat format; `answer` is appended as the assistant turn when given."""
    return [
        {"role": "<|User|>", "content": sample["prompt"], "images": sample.get("images", [])},
        {"role": "<|Assistant|>", "content": answer or ""},
    ]


def reference_answer(sample: Dict[str, Any]) -> Optional[str]:
    """Gold text used for teacher forcing during distillation (None if the sample has no label)."""
    if sample.get("answers"):
        return str(sample["answers"][0])
    target = sample.get("target")
    if target is None:
        return None
    if sample.get("metric") == "key_value_f1":
        return "\n".join(f"{k}={v[0] if isinstance(v, list) else v}" for k, v in target.items())
    return json.dumps(target, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _resolve(path_field: str, roots: Iterable[str]) -> str:
    candidates = [path_field]
    for root in roots:
        candidates += [os.path.join(root, path_field), os.path.join(root, os.path.basename(path_field))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"could not locate '{path_field}' (tried {candidates})")


def _load_docvqa_json(annotation_file: str) -> List[Dict[str, Any]]:
    with open(annotation_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content["data"] if isinstance(content, dict) and "data" in content else content


def convert_docvqa(annotation_file: str, images_dir: str, dataset: str = "docvqa",
                   suffix: str = VQA_SUFFIX) -> Iterator[Dict[str, Any]]:
    """DocVQA (SP-DocVQA) / InfographicsVQA official json files (`image` or `image_local_name`)."""
    for entry in _load_docvqa_json(annotation_file):
        image_field = entry.get("image") or entry.get("image_local_name")
        try:
            image_path = _resolve(image_field, [images_dir, os.path.join(images_dir, "documents"),
                                                os.path.join(images_dir, "infographicsvqa_images")])
        except FileNotFoundError:
            continue
        question = entry["question"].strip()
        yield {
            "id": f"{dataset}-{entry.get('questionId')}",
            "dataset": dataset,
            "task": "vqa",
            "metric": "anls",
            "images": [image_path],
            "prompt": image_prompt(f"{question}\n{suffix}".strip()),
            "answers": entry.get("answers") or [],
            "max_new_tokens": 128,
        }


def convert_cord(source: str, out_dir: str, split: str = "test") -> Iterator[Dict[str, Any]]:
    """CORD v2 (HuggingFace `naver-clova-ix/cord-v2` or a local copy of it): structured JSON targets."""
    from datasets import load_dataset

    dataset = load_dataset(source, split=split)
    image_dir = os.path.join(out_dir, "images", "cord")
    os.makedirs(image_dir, exist_ok=True)
    for idx, row in enumerate(dataset):
        ground_truth = json.loads(row["ground_truth"])
        gt_parse = ground_truth.get("gt_parse", ground_truth)
        target = {key: gt_parse[key] for key in ("menu", "sub_total", "total") if key in gt_parse}
        image_path = os.path.join(image_dir, f"cord_{split}_{idx:04d}.png")
        if not os.path.isfile(image_path):
            row["image"].convert("RGB").save(image_path)
        yield {
            "id": f"cord-{split}-{idx}",
            "dataset": "cord",
            "task": "kie",
            "metric": "cord_f1",
            "images": [os.path.abspath(image_path)],
            "prompt": image_prompt(CORD_PROMPT),
            "target": target,
            "max_new_tokens": 512,
        }


def convert_sroie(root: str) -> Iterator[Dict[str, Any]]:
    """
    SROIE (ICDAR 2019 task 3) test split laid out as `<root>/img/*.jpg` + `<root>/entities/*.txt`
    (each entities file is a JSON object with company / date / address / total).
    Samples whose entity file is missing or malformed are skipped (paper: 18 erroneous files).
    """
    for image_path in sorted(glob.glob(os.path.join(root, "img", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        entity_path = os.path.join(root, "entities", stem + ".txt")
        try:
            with open(entity_path, "r", encoding="utf-8", errors="ignore") as f:
                entities = json.loads(f.read())
        except (OSError, json.JSONDecodeError):
            continue
        target = {k: str(entities.get(k, "")).strip() for k in ("company", "date", "address", "total")}
        if not all(target.values()):
            continue
        yield {
            "id": f"sroie-{stem}",
            "dataset": "sroie",
            "task": "kie",
            "metric": "token_f1",
            "images": [os.path.abspath(image_path)],
            "prompt": image_prompt(SROIE_PROMPT),
            "answers": [f"{target['company']}\n{target['date']}\n{target['address']}\n{target['total']}"],
            "target": target,
            "max_new_tokens": 256,
        }


def convert_funsd(root: str) -> Iterator[Dict[str, Any]]:
    """FUNSD `testing_data` folder: `<root>/images/*.png` + `<root>/annotations/*.json`."""
    for ann_path in sorted(glob.glob(os.path.join(root, "annotations", "*.json"))):
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        image_path = os.path.join(root, "images", stem + ".png")
        if not os.path.isfile(image_path):
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            form = json.load(f).get("form", [])
        entities = [
            {"label": e.get("label", "other"), "text": e.get("text", "").strip()}
            for e in form if e.get("text", "").strip()
        ]
        yield {
            "id": f"funsd-{stem}",
            "dataset": "funsd",
            "task": "kie",
            "metric": "token_f1",
            "images": [os.path.abspath(image_path)],
            "prompt": image_prompt(FORM_PROMPT),
            "answers": ["\n".join(e["text"] for e in entities)],
            "target": {"entities": entities},
            "max_new_tokens": 1024,
        }


def convert_docile(root: str, out_dir: str, split: str = "val", dpi: int = 150,
                   max_pages: Optional[int] = 1) -> Iterator[Dict[str, Any]]:
    """
    DocILE: `<root>/<split>.json` (list of document ids), `<root>/annotations/<id>.json`,
    `<root>/pdfs/<id>.pdf`. PDFs are rendered with PyMuPDF at 150 DPI; the gold values are the
    texts of all KILE fields and line-item fields on the rendered pages.
    """
    with open(os.path.join(root, f"{split}.json"), "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    image_dir = os.path.join(out_dir, "images", "docile")
    for doc_id in doc_ids:
        pdf_path = os.path.join(root, "pdfs", f"{doc_id}.pdf")
        ann_path = os.path.join(root, "annotations", f"{doc_id}.json")
        if not (os.path.isfile(pdf_path) and os.path.isfile(ann_path)):
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        pages = render_pdf(pdf_path, image_dir, dpi=dpi, max_pages=max_pages)
        fields = annotation.get("field_extractions", []) + annotation.get("line_item_extractions", [])
        texts = [
            fld.get("text", "").strip() for fld in fields
            if fld.get("text", "").strip() and fld.get("page", 0) < len(pages)
        ]
        if not texts:
            continue
        yield {
            "id": f"docile-{doc_id}",
            "dataset": "docile",
            "task": "kie",
            "metric": "token_f1",
            "images": pages,
            "prompt": image_prompt(FORM_PROMPT, len(pages)),
            "answers": ["\n".join(texts)],
            "max_new_tokens": 1024,
        }


def _parse_klc_expected(line: str) -> Dict[str, List[str]]:
    target: Dict[str, List[str]] = {}
    for item in line.strip().split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        target.setdefault(key, []).append(value.replace("_", " "))
    return target


def convert_klc(root: str, out_dir: str, split: str = "dev-0", dpi: int = 300,
                max_pages: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Kleister Charity: `<root>/<split>/in.tsv` (first column = PDF file name),
    `<root>/<split>/expected.tsv` (space separated key=value, '_' for spaces) and
    `<root>/documents/<file>.pdf` (fetched with `git annex get`). All pages are rendered at
    300 DPI and passed jointly with one <image> tag per page.
    """
    csv.field_size_limit(1 << 30)
    split_dir = os.path.join(root, split)
    with open(os.path.join(split_dir, "in.tsv"), "r", encoding="utf-8") as f:
        inputs = [row for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    with open(os.path.join(split_dir, "expected.tsv"), "r", encoding="utf-8") as f:
        expected = [line.rstrip("\n") for line in f]
    image_dir = os.path.join(out_dir, "images", "klc")
    for row, gold_line in zip(inputs, expected):
        filename = row[0]
        pdf_path = os.path.join(root, "documents", filename)
        if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) < 1024:
            # git-annex placeholders are tiny symlink/pointer files
            continue
        pages = render_pdf(pdf_path, image_dir, dpi=dpi, max_pages=max_pages)
        yield {
            "id": f"klc-{os.path.splitext(filename)[0]}",
            "dataset": "klc",
            "task": "kie",
            "metric": "key_value_f1",
            "images": pages,
            "prompt": image_prompt(KLC_PROMPT, len(pages)),
            "target": _parse_klc_expected(gold_line),
            "max_new_tokens": 256,
        }


def convert_wtq(root: str, out_dir: str, split_file: str = "data/pristine-unseen-tables.tsv",
                renderer: str = "auto", suffix: str = VQA_SUFFIX) -> Iterator[Dict[str, Any]]:
    """
    WikiTableQuestions: `<root>/<split_file>` with columns id / utterance / context / targetValue,
    tables in `<root>/csv/*-csv/*.csv`. Tables are rendered once to PNG and cached.
    """
    image_dir = os.path.join(out_dir, "images", "wtq")
    with open(os.path.join(root, split_file), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            context = row["context"]
            image_path = os.path.join(image_dir, context.replace("/", "_").replace(".csv", ".png"))
            if not os.path.isfile(image_path):
                rows = table_rows_from_csv(os.path.join(root, context))
                render_table(rows, image_path, renderer=renderer)
            answers = [a.replace("\\n", "\n").replace("\\p", "|") for a in row["targetValue"].split("|")]
            yield {
                "id": f"wtq-{row['id']}",
                "dataset": "wtq",
                "task": "tableqa",
                "metric": "wtq",
                "images": [os.path.abspath(image_path)],
                "prompt": image_prompt(f"{row['utterance'].strip()}\n{suffix}".strip()),
                "answers": [", ".join(answers)] + (answers if len(answers) > 1 else []),
                "max_new_tokens": 128,
            }


def convert_rvl_cdip(root: str, labels_file: str = "labels/test.txt",
                     images_subdir: str = "images") -> Iterator[Dict[str, Any]]:
    """RVL-CDIP: `<root>/<labels_file>` lines `<relative/path.tif> <label_id>`; TIFFs use their first page."""
    with open(os.path.join(root, labels_file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) != 2:
                continue
            rel_path, label_id = parts
            image_path = os.path.join(root, images_subdir, rel_path)
            if not os.path.isfile(image_path):
                continue
            label = RVL_CDIP_CLASSES[int(label_id)]
            yield {
                "id": f"rvl_cdip-{os.path.splitext(os.path.basename(rel_path))[0]}",
                "dataset": "rvl_cdip",
                "task": "cls",
                "metric": "accuracy",
                "images": [os.path.abspath(image_path)],
                "prompt": image_prompt(RVL_CDIP_PROMPT),
                "answers": [label],
                "label_set": RVL_CDIP_CLASSES,
                "max_new_tokens": 16,
            }
