"""Generation-based evaluation on LiteDoc JSONL benchmarks (paper Sec. 4.3)."""

import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from . import metrics
from .modeling import generate_answer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

# metric that is reported as THE score of each dataset (x100 in the summary)
PRIMARY_METRIC = {
    "anls": "anls",
    "wtq": "accuracy",
    "token_f1": "f1",
    "cord_f1": "f1",
    "key_value_f1": "f1",
    "accuracy": "accuracy",
}


def postprocess(prediction: str, sample: Dict[str, Any]) -> str:
    """Regex-based artifact removal before scoring."""
    text = prediction.replace("<｜end▁of▁sentence｜>", "").strip()
    task = sample.get("task")
    if task in ("vqa", "tableqa", "cls"):
        text = metrics.strip_grounding(text)
        text = text.splitlines()[0] if text else text
        text = re.sub(r"^(the )?answer( is)?\s*[:：]?\s*", "", text, flags=re.IGNORECASE)
        return text.strip().strip(".").strip()
    if sample.get("metric") in ("token_f1",):
        return metrics.strip_grounding(text)
    return text


def score_sample(sample: Dict[str, Any], prediction: str) -> Dict[str, Any]:
    metric = sample.get("metric", "anls")
    answers = sample.get("answers") or []
    if metric == "anls":
        return {"anls": metrics.anls(prediction, answers)}
    if metric == "wtq":
        return {"accuracy": metrics.exact_match(prediction, answers), "anls": metrics.anls(prediction, answers)}
    if metric == "accuracy":
        return {"accuracy": metrics.classification_match(prediction, answers[0], sample.get("label_set"))}
    if metric == "token_f1":
        result = metrics.bag_of_tokens_f1(prediction, answers)
        target = sample.get("target")
        if isinstance(target, dict) and sample.get("dataset") == "sroie":
            result["field_errors"] = metrics.per_field_error_rates(metrics.parse_key_values(prediction), target)
        return result
    if metric == "cord_f1":
        parsed = metrics.extract_json(prediction)
        field = metrics.per_field_f1(parsed, sample["target"])
        return {
            "f1": field["macro_f1"],
            "per_field": field["per_field"],
            "json_valid": float(parsed is not None),
            "field_errors": metrics.per_field_error_rates(parsed if isinstance(parsed, dict) else {}, sample["target"]),
        }
    if metric == "key_value_f1":
        return metrics.key_value_f1(metrics.parse_key_values(prediction), sample["target"])
    raise ValueError(f"unknown metric '{metric}'")


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        if "scores" in result:
            by_dataset[result["dataset"]].append(result)
    summary: Dict[str, Any] = {}
    for dataset, rows in by_dataset.items():
        metric = rows[0]["metric"]
        primary = PRIMARY_METRIC[metric]
        entry = {
            "metric": metric,
            "primary": primary,
            "score": 100.0 * sum(r["scores"][primary] for r in rows) / len(rows),
            "num_scored": len(rows),
            "num_errors": sum(1 for r in results if r["dataset"] == dataset and "error" in r),
        }
        if metric == "wtq":
            entry["anls"] = 100.0 * sum(r["scores"]["anls"] for r in rows) / len(rows)
        if metric == "cord_f1":
            fields = defaultdict(list)
            for r in rows:
                for field, value in r["scores"]["per_field"].items():
                    fields[field].append(value)
            entry["per_field_f1"] = {f: 100.0 * sum(v) / len(v) for f, v in sorted(fields.items())}
        field_errors = defaultdict(lambda: {"cer": [], "wer": []})
        for r in rows:
            for field, err in r["scores"].get("field_errors", {}).items():
                field_errors[field]["cer"].append(err["cer"])
                field_errors[field]["wer"].append(err["wer"])
        if field_errors:
            entry["per_field_error_rates"] = {
                f: {k: sum(v) / len(v) for k, v in e.items()} for f, e in sorted(field_errors.items())
            }
        summary[dataset] = entry
    if summary:
        summary["average"] = sum(v["score"] for v in summary.values()) / len(summary)
    return summary


def run_evaluation(
    model,
    processor,
    samples: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: bool = False,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    chunk_size: int = -1,
    dtype: torch.dtype = torch.bfloat16,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate an answer for every sample, score it and return per-dataset scores."""
    if seed is not None:
        torch.manual_seed(seed)
    results = []
    out_f = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
        out_f = open(output_file, "w", encoding="utf-8")
    generated_tokens = 0
    start = time.time()
    try:
        for sample in tqdm(samples, desc="evaluating"):
            record = {"id": sample.get("id"), "dataset": sample.get("dataset"), "metric": sample.get("metric")}
            try:
                output = generate_answer(
                    model, processor, sample,
                    max_new_tokens=max_new_tokens or sample.get("max_new_tokens", 512),
                    do_sample=do_sample, temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty, chunk_size=chunk_size, dtype=dtype,
                )
                generated_tokens += output["num_new_tokens"]
                prediction = postprocess(output["text"], sample)
                record.update(prediction=prediction, raw_prediction=output["text"],
                              scores=score_sample(sample, prediction))
                if verbose:
                    print(f"\n[{record['id']}] pred={prediction!r} gold={sample.get('answers') or sample.get('target')} "
                          f"scores={ {k: v for k, v in record['scores'].items() if isinstance(v, float)} }")
            except Exception as e:  # noqa: BLE001 - one bad sample must not kill the whole run
                record["error"] = f"{type(e).__name__}: {e}"
                print(f"[warn] sample {record['id']} failed: {record['error']}", file=sys.stderr)
            results.append(record)
            if out_f is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()
    summary = summarize(results)
    elapsed = time.time() - start
    summary["_runtime"] = {"seconds": elapsed, "generated_tokens": generated_tokens,
                           "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0}
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    lines = [f"{'dataset':<12} {'metric':<10} {'score':>8} {'n':>6}"]
    for dataset, entry in summary.items():
        if dataset.startswith("_") or dataset == "average":
            continue
        lines.append(f"{dataset:<12} {entry['primary']:<10} {entry['score']:>8.2f} {entry['num_scored']:>6}")
    if "average" in summary:
        lines.append(f"{'average':<12} {'':<10} {summary['average']:>8.2f}")
    return "\n".join(lines)
