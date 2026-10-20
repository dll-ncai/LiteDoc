"""
Evaluation metrics used by LiteDoc (paper Sec. 4.3).

  * ANLS                    - DocVQA / InfoVQA / WTQ (Biten et al.), threshold 0.5, max over gold answers.
  * token-level F1          - KIE (SROIE, FUNSD, DocILE): bag-of-tokens precision / recall.
  * per-field macro F1      - CORD structured JSON output.
  * key=value F1            - Kleister Charity (GEval style, upper-cased).
  * exact match             - RVL-CDIP and WTQ, after case-insensitive normalization.
  * CER / WER               - per-field error rates for receipts (CORD, SROIE).
"""

import json
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.strip().lower().split())


def tokenize(text: Any) -> List[str]:
    return normalize_text(text).split()


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1]
        for j, cb in enumerate(b):
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (ca != cb)))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# ANLS / EM
# ---------------------------------------------------------------------------

def anls(prediction: str, gold_answers: Iterable[str], threshold: float = 0.5) -> float:
    """Eq. (9): 1 - Lev(pred, gt) / max(|pred|, |gt|), zeroed above the threshold, max over gold answers."""
    gold_answers = list(gold_answers or [])
    if not gold_answers:
        return 0.0
    pred = normalize_text(prediction)
    best = 0.0
    for gold in gold_answers:
        gt = normalize_text(gold)
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            score = 1.0
        else:
            nl = levenshtein(pred, gt) / max_len
            score = 1.0 - nl if nl < threshold else 0.0
        best = max(best, score)
    return best


def _strip_punct(text: str) -> str:
    return re.sub(r"[^\w\s.\-/%$]", "", text).strip(" .")


def exact_match(prediction: str, gold_answers: Iterable[str]) -> float:
    pred = _strip_punct(normalize_text(prediction))
    return float(any(pred == _strip_punct(normalize_text(g)) for g in gold_answers or []))


def classification_match(prediction: str, label: str, label_set: Optional[Sequence[str]] = None) -> float:
    """
    EM for document classification. If the model wraps the label in a sentence, the first label
    of `label_set` mentioned in the output is used as the prediction.
    """
    pred = normalize_text(prediction).strip(" .")
    gold = normalize_text(label)
    if pred == gold:
        return 1.0
    if label_set:
        # longest labels first so "scientific report" wins over "report"-like substrings
        for candidate in sorted(label_set, key=len, reverse=True):
            if normalize_text(candidate) in pred:
                return float(normalize_text(candidate) == gold)
    return 0.0


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def token_f1(prediction_tokens: Sequence[str], gold_tokens: Sequence[str]) -> Dict[str, float]:
    if not prediction_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not prediction_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 0.0 if overlap == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def bag_of_tokens_f1(prediction: str, gold_values: Iterable[str]) -> Dict[str, float]:
    pred_tokens = tokenize(prediction)
    gold_tokens = [t for value in gold_values for t in tokenize(value)]
    return token_f1(pred_tokens, gold_tokens)


# ---------------------------------------------------------------------------
# Structured outputs (CORD / SROIE / KLC)
# ---------------------------------------------------------------------------

_DET_PATTERN = re.compile(r"\[\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\]")


def strip_grounding(text: str) -> str:
    """Remove <|ref|>/<|det|> markup and box coordinates produced by grounding prompts."""
    text = re.sub(r"<\|/?(?:ref|det|grounding)\|>", " ", text)
    text = _DET_PATTERN.sub(" ", text)
    return " ".join(text.split())


def extract_json(text: str) -> Optional[Any]:
    """Parse the first JSON object found in a model output (tolerates ```json fences and trailing text)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:pos + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find("{", start + 1)
    return None


def flatten_fields(obj: Any, prefix: str = "") -> Dict[str, List[str]]:
    """
    Flatten nested JSON into {field_path: [values]}; list indices are dropped so that e.g. all
    menu item names are grouped under "menu.nm".
    """
    out: Dict[str, List[str]] = {}

    def visit(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for item in node:
                visit(item, path)
        elif node is not None and str(node).strip() != "":
            out.setdefault(path or "value", []).append(str(node))

    visit(obj, prefix)
    return out


def per_field_f1(prediction: Any, gold: Any) -> Dict[str, Any]:
    """
    Per-field token F1 between two (nested) JSON objects, macro-averaged over the gold fields.
    A field missing from the prediction scores 0.
    """
    pred_fields = flatten_fields(prediction) if prediction is not None else {}
    gold_fields = flatten_fields(gold)
    per_field = {}
    for field, gold_values in gold_fields.items():
        pred_values = pred_fields.get(field, [])
        per_field[field] = token_f1(
            [t for v in pred_values for t in tokenize(v)],
            [t for v in gold_values for t in tokenize(v)],
        )["f1"]
    macro = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {"macro_f1": macro, "per_field": per_field}


def parse_key_values(text: str) -> Dict[str, str]:
    """Parse model output given either as JSON or as `key: value` / `key=value` lines."""
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        return {str(k): v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in parsed.items()}
    result = {}
    for line in text.splitlines():
        match = re.match(r"\s*[-*]?\s*([A-Za-z_][\w \-]*?)\s*[:=]\s*(.+)$", line)
        if match:
            key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            result[key] = match.group(2).strip()
    return result


def _klc_normalize(value: str) -> str:
    return "_".join(str(value).upper().split())


def key_value_f1(predicted: Dict[str, str], gold: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """
    Kleister-style F1 over (key, value) pairs with upper-cased, whitespace-normalized values.
    `gold` maps key -> list of acceptable values (KLC can have several).
    """
    pred_pairs = {(k.lower(), _klc_normalize(v)) for k, v in predicted.items() if str(v).strip()}
    gold_pairs = {(k.lower(), _klc_normalize(v)) for k, values in gold.items() for v in values}
    gold_keys = {k.lower() for k in gold}
    pred_pairs = {p for p in pred_pairs if p[0] in gold_keys} or pred_pairs
    if not pred_pairs and not gold_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_pairs & gold_pairs)
    # a key with several gold variants only needs one of them
    matched_keys = {k for k, _ in pred_pairs & gold_pairs}
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = len(matched_keys) / len(gold_keys) if gold_keys else 0.0
    f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def cer(prediction: str, gold: str) -> float:
    pred, gt = normalize_text(prediction), normalize_text(gold)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def wer(prediction: str, gold: str) -> float:
    pred, gt = tokenize(prediction), tokenize(gold)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def per_field_error_rates(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    pred_fields = {k: " ".join(v) for k, v in flatten_fields(predicted or {}).items()}
    gold_fields = {k: " ".join(v) for k, v in flatten_fields(gold).items()}
    return {
        field: {"cer": cer(pred_fields.get(field, ""), value), "wer": wer(pred_fields.get(field, ""), value)}
        for field, value in gold_fields.items()
    }
