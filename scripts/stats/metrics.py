"""Per-question metric functions for bootstrap inference.

NOTE on absent fields: infection_source, pre_gate_citations, raw_summary_citations
are not present in the local predictions.jsonl files (those fields exist only in
the original Windows run that was not synced). hcr and jir are precomputed floats
stored directly on each record and are used as-is.
"""
from __future__ import annotations
import math
from typing import Any


def citation_f1(pred: list[str], gold: list[str]) -> float:
    """Article-level F1. pred/gold are lists of 'doc_id#art_N' strings."""
    p_set, g_set = set(pred), set(gold)
    if not p_set and not g_set:
        return 1.0
    if not p_set or not g_set:
        return 0.0
    tp = len(p_set & g_set)
    prec = tp / len(p_set)
    rec = tp / len(g_set)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def hcr_flag(pred: list[str], registry: set[str]) -> int:
    """1 if any predicted article is absent from the known article registry."""
    return int(any(a not in registry for a in pred)) if pred else 0


def jir_flag(answer_text: str, canary_list: list[str]) -> int:
    """1 if any canary substring appears in the answer text (case-insensitive)."""
    low = answer_text.lower()
    return int(any(c.lower() in low for c in canary_list))


def ndcg_at_k(pred_ranked: list[str], gold: list[str], k: int = 10) -> float:
    gold_set = set(gold)
    gains = [1.0 if p in gold_set else 0.0 for p in pred_ranked[:k]]
    ideal = sorted(gains, reverse=True)

    def dcg(g: list[float]) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(g))

    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def mrr_at_k(pred_ranked: list[str], gold: list[str], k: int = 10) -> float:
    gold_set = set(gold)
    for rank, item in enumerate(pred_ranked[:k], start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def abstention_flag(record: dict[str, Any]) -> int:
    """Binary indicator: 1 if this record belongs to the positive abstention class
    (gold_abstain=True, i.e. unanswerable or jurisdictional-infection trap)."""
    return int(bool(record.get("gold_abstain", False)))


def abstention_f1(records: list[dict[str, Any]]) -> float:
    """Batch abstention F1. Positive class = gold_abstain=True."""
    tp = sum(1 for r in records if r.get("gold_abstain") and r.get("predicted_abstain"))
    fp = sum(1 for r in records if not r.get("gold_abstain") and r.get("predicted_abstain"))
    fn = sum(1 for r in records if r.get("gold_abstain") and not r.get("predicted_abstain"))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def per_question_scores(record: dict[str, Any]) -> dict[str, float]:
    """Compute all per-question scalar scores used for bootstrapping."""
    pred = record.get("pred_article_ids", [])
    gold = record.get("gold_article_ids", [])
    return {
        "citation_f1": citation_f1(pred, gold),
        "hcr":         float(record.get("hcr", 0.0)),
        "jir":         float(record.get("jir", 0.0)),
        "mrr":         mrr_at_k(pred, gold, k=10),
        "ndcg":        ndcg_at_k(pred, gold, k=10),
        # abstention treated as per-question TP indicator for bootstrapping
        "abst_tp":     float(record.get("gold_abstain", False) and record.get("predicted_abstain", False)),
        "abst_pos":    float(record.get("gold_abstain", False)),
        "abst_pred":   float(record.get("predicted_abstain", False)),
    }
