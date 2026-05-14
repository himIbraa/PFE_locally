"""Evaluate ``akn_rlm.rlm.classifier`` against the AlgerianLegalBench gold
``query_type`` labels.

Phase B task 3 (see HANDOFF.md §3 Phase B): produces a confusion matrix
(8x8) plus per-class precision/recall/F1 and overall accuracy so we know
how much of the locked SOTA Cite F1 we lose when the dispatcher routes
via the classifier instead of the benchmark's gold ``query_type``.

Usage:
    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"
    & $py scripts\\eval_classifier_accuracy.py \\
          --out eval_results\\classifier_accuracy_v1

Output files (under {--out}/):
    classifier_predictions.jsonl   one line per question with gold + pred
    confusion_matrix.json          {gold_type: {pred_type: count}}
    classifier_report.md           Markdown summary table
    classifier_report.txt          Plain-text summary (mirrors stdout)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.config import get_benchmark_path  # noqa: E402
from akn_rlm.rlm.classifier import (  # noqa: E402
    DEFAULT_LLM_CLASSIFIER_MODEL,
    classify,
    llm_classify,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_classifier_accuracy")


QUERY_TYPES: list[str] = [
    "rule_application",
    "exact_article",
    "multi_hop",
    "unanswerable",
    "layman",
    "long_context",
    "conceptual_definitional",
    "temporal_factual",
]


def _load_benchmark(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("questions", data) if isinstance(data, dict) else data
    return list(raw)


def _classify_one(
    query: str,
    *,
    use_llm: bool = False,
    llm_pool=None,
    model: str = DEFAULT_LLM_CLASSIFIER_MODEL,
) -> tuple[str, float]:
    try:
        if use_llm:
            if llm_pool is None:
                raise RuntimeError("use_llm=True but llm_pool is None")
            result = llm_classify(query, llm_pool, model=model)
        else:
            result = classify(query)
        return result.query_type, float(result.confidence)
    except Exception as exc:
        log.warning("classify() raised %s on query=%r", exc, query[:80])
        return "rule_application", 0.0


def _per_class_metrics(
    confusion: dict[str, dict[str, int]],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute precision / recall / F1 per class from a confusion dict.

    confusion[gold][pred] = count
    """
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(v for k, v in confusion.get(label, {}).items() if k != label)
        fp = sum(
            confusion.get(g, {}).get(label, 0)
            for g in labels if g != label
        )
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return out


def _format_confusion_md(
    confusion: dict[str, dict[str, int]],
    labels: list[str],
) -> str:
    short = {
        "rule_application":         "RA",
        "exact_article":            "EA",
        "multi_hop":                "MH",
        "unanswerable":             "UA",
        "layman":                   "Lay",
        "long_context":             "LC",
        "conceptual_definitional":  "CD",
        "temporal_factual":         "TF",
    }
    header = "| gold \\ pred | " + " | ".join(short[l] for l in labels) + " | total |"
    sep = "|---" * (len(labels) + 2) + "|"
    rows = [header, sep]
    for g in labels:
        row_total = sum(confusion.get(g, {}).get(p, 0) for p in labels)
        vals = [str(confusion.get(g, {}).get(p, 0)) for p in labels]
        rows.append(f"| **{short[g]}** | " + " | ".join(vals) + f" | {row_total} |")
    col_totals = []
    for p in labels:
        col_totals.append(str(sum(confusion.get(g, {}).get(p, 0) for g in labels)))
    grand = sum(int(c) for c in col_totals)
    rows.append("| **total** | " + " | ".join(col_totals) + f" | {grand} |")
    return "\n".join(rows)


def _format_metrics_md(metrics: dict[str, dict[str, float]]) -> str:
    rows = [
        "| Query type | n | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in QUERY_TYPES:
        m = metrics[label]
        rows.append(
            f"| {label} | {int(m['support'])} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate akn_rlm.rlm.classifier against ALB v3.0 gold labels."
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Path to AlgerianLegalBench JSON (default: auto-detect)",
    )
    parser.add_argument(
        "--out", default="eval_results/classifier_accuracy_v1",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of questions (smoke test)",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use the LLM-backed classifier (Gemma-4-31B by default) "
             "instead of the regex one.",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_LLM_CLASSIFIER_MODEL,
        help=f"Model name for the LLM classifier "
             f"(default: {DEFAULT_LLM_CLASSIFIER_MODEL}).",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Benchmark: %s", benchmark_path)
    log.info("Output   : %s/", out_dir)

    questions = _load_benchmark(benchmark_path)
    if args.limit:
        questions = questions[: args.limit]
    log.info("Loaded %d questions", len(questions))

    llm_pool = None
    if args.llm:
        log.info("Loading LLMPool for Gemma-backed classifier (model=%s) …",
                 args.llm_model)
        from akn_rlm.llm.client import LLMPool  # noqa: PLC0415
        llm_pool = LLMPool.default()

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows: list[dict] = []
    correct = 0

    for idx, q in enumerate(questions):
        gold = q.get("query_type", "rule_application")
        query = q.get("question", q.get("query", ""))
        pred, conf = _classify_one(
            query,
            use_llm=args.llm,
            llm_pool=llm_pool,
            model=args.llm_model,
        )
        if args.llm and (idx + 1) % 20 == 0:
            log.info("Classified %d / %d", idx + 1, len(questions))
        confusion[gold][pred] += 1
        is_correct = (pred == gold)
        if is_correct:
            correct += 1
        rows.append({
            "id":              q.get("id", ""),
            "query":           query,
            "gold_query_type": gold,
            "pred_query_type": pred,
            "confidence":      conf,
            "correct":         is_correct,
        })

    overall_acc = correct / len(questions) if questions else 0.0

    # Persist predictions
    pred_path = out_dir / "classifier_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Predictions saved %s", pred_path)

    # Confusion matrix as plain dict for JSON
    confusion_plain: dict[str, dict[str, int]] = {
        g: dict(preds) for g, preds in confusion.items()
    }
    cm_path = out_dir / "confusion_matrix.json"
    with cm_path.open("w", encoding="utf-8") as fh:
        json.dump(confusion_plain, fh, ensure_ascii=False, indent=2)
    log.info("Confusion matrix saved %s", cm_path)

    metrics = _per_class_metrics(confusion_plain, QUERY_TYPES)
    macro_f1 = (
        sum(m["f1"] for m in metrics.values()) / len(metrics)
        if metrics else 0.0
    )
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in metrics.values())
        / sum(m["support"] for m in metrics.values())
        if any(m["support"] for m in metrics.values()) else 0.0
    )

    classifier_kind = (
        f"LLM ({args.llm_model})" if args.llm else "regex"
    )

    md_lines = [
        "# Classifier accuracy report",
        "",
        f"- Classifier: **{classifier_kind}**",
        f"- Questions evaluated: **{len(questions)}**",
        f"- Overall accuracy: **{overall_acc:.4f}** ({correct}/{len(questions)})",
        f"- Macro F1: **{macro_f1:.4f}**",
        f"- Weighted F1: **{weighted_f1:.4f}**",
        "",
        "## Per-class metrics",
        "",
        _format_metrics_md(metrics),
        "",
        "## Confusion matrix (rows = gold, cols = predicted)",
        "",
        _format_confusion_md(confusion_plain, QUERY_TYPES),
        "",
        "Short codes: RA=rule_application, EA=exact_article, MH=multi_hop, "
        "UA=unanswerable, Lay=layman, LC=long_context, "
        "CD=conceptual_definitional, TF=temporal_factual.",
        "",
    ]
    md_text = "\n".join(md_lines)
    (out_dir / "classifier_report.md").write_text(md_text, encoding="utf-8")
    (out_dir / "classifier_report.txt").write_text(md_text, encoding="utf-8")
    log.info("Reports saved %s/", out_dir)

    # Stdout summary
    print(md_text)


if __name__ == "__main__":
    main()
