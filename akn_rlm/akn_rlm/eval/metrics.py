"""Retrieval and answer-quality metrics for AlgerianLegalBench — Phase I.

Per-query metrics (pure functions, no side effects):
  Retrieval    : mrr_at_k, recall_at_k, precision_at_k, ndcg_at_k, map_at_k
  Citation     : citation_precision, citation_recall, citation_f1,
                 doc_level_citation_f1, citation_ordering_kendall_tau
  Generation   : exact_match, token_f1, rouge_l, sacrebleu_score,
                 bertscore_f1, reasoning_chain_score
  Faithfulness : hcr, jir, answer_faithfulness, citation_groundedness
  Abstention   : abstention_accuracy

Batch metrics (computed over a list of result dicts):
  abstention_recall, abstention_precision, abstention_f1
  telemetry_metrics (mean depth, tokens, latency, retry rate)

aggregate() computes macro-averages of all per-query metrics plus batch metrics.

Optional heavy dependencies (sacrebleu, bert_score, scipy) fall back to 0.0
when not installed so the module is always importable.
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank(predicted: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(predicted, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def _bigrams(tokens: list[str]) -> set[tuple]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _mean(results: list[dict], key: str, default: float = 0.0) -> float:
    if not results:
        return default
    return sum(r.get(key, default) for r in results) / len(results)


def _citation_key(c: dict) -> tuple:
    return (c.get("doc_id", ""), str(c.get("article_ref", "")))


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def mrr_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    return _reciprocal_rank(predicted[:k], relevant)


def recall_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(1 for p in predicted[:k] if p in relevant) / len(relevant)


def precision_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    if k == 0:
        return 0.0
    return sum(1 for p in predicted[:k] if p in relevant) / k


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    gains = [1.0 if p in relevant else 0.0 for p in predicted[:k]]
    ideal = sorted(gains, reverse=True)
    dcg = _dcg(gains)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    """Mean Average Precision @ K for a single query."""
    if not relevant:
        return 0.0
    num_rel = 0
    prec_sum = 0.0
    for i, item in enumerate(predicted[:k], start=1):
        if item in relevant:
            num_rel += 1
            prec_sum += num_rel / i
    return prec_sum / len(relevant) if num_rel else 0.0


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------

def citation_precision(predicted_citations: list[dict], gold_citations: list[dict]) -> float:
    pred_set = {_citation_key(c) for c in predicted_citations}
    gold_set = {_citation_key(c) for c in gold_citations}
    if not pred_set:
        return 0.0
    return len(pred_set & gold_set) / len(pred_set)


def citation_recall(predicted_citations: list[dict], gold_citations: list[dict]) -> float:
    pred_set = {_citation_key(c) for c in predicted_citations}
    gold_set = {_citation_key(c) for c in gold_citations}
    if not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(gold_set)


def citation_f1(predicted_citations: list[dict], gold_citations: list[dict]) -> float:
    p = citation_precision(predicted_citations, gold_citations)
    r = citation_recall(predicted_citations, gold_citations)
    if not predicted_citations and not gold_citations:
        return 1.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def doc_level_citation_f1(predicted_citations: list[dict], gold_citations: list[dict]) -> float:
    pred_docs = {c.get("doc_id", "") for c in predicted_citations}
    gold_docs = {c.get("doc_id", "") for c in gold_citations}
    if not gold_docs and not pred_docs:
        return 1.0
    if not gold_docs or not pred_docs:
        return 0.0
    tp = len(pred_docs & gold_docs)
    p = tp / len(pred_docs)
    r = tp / len(gold_docs)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def citation_ordering_kendall_tau(
    predicted_citations: list[dict],
    gold_citations: list[dict],
) -> float:
    """Kendall tau correlation between predicted and gold citation rank order."""
    gold_keys = [_citation_key(c) for c in gold_citations]
    pred_keys = [_citation_key(c) for c in predicted_citations]
    gold_rank = {k: i for i, k in enumerate(gold_keys)}
    common = [k for k in pred_keys if k in gold_rank]
    if len(common) < 2:
        return 0.0
    gold_order = [gold_rank[k] for k in common]
    try:
        from scipy.stats import kendalltau  # type: ignore
        tau, _ = kendalltau(list(range(len(common))), gold_order)
        return float(tau) if tau == tau else 0.0  # guard NaN
    except ImportError:
        log.debug("scipy not available; citation_ordering_kendall_tau returns 0.0")
        return 0.0
    except Exception as exc:
        log.warning("kendalltau failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def exact_match(predicted_text: str, gold_text: str) -> float:
    return 1.0 if predicted_text.strip() == gold_text.strip() else 0.0


def token_f1(predicted_text: str, gold_text: str) -> float:
    pred = set(predicted_text.lower().split())
    gold = set(gold_text.lower().split())
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rouge_l(predicted_text: str, gold_text: str) -> float:
    """ROUGE-L F1 computed via LCS (no external dependency)."""
    if not predicted_text or not gold_text:
        return 0.0
    pred_tokens = predicted_text.split()
    gold_tokens = gold_text.split()
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred_tokens)
    r = lcs / len(gold_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def sacrebleu_score(predicted_text: str, gold_text: str) -> float:
    """SacreBLEU-4; returns 0.0 if sacrebleu is not installed."""
    try:
        from sacrebleu.metrics import BLEU  # type: ignore
        bleu = BLEU(effective_order=True)
        result = bleu.sentence_score(predicted_text, [gold_text])
        return result.score / 100.0
    except ImportError:
        return 0.0
    except Exception as exc:
        log.warning("sacrebleu failed: %s", exc)
        return 0.0


def bertscore_f1(predicted_text: str, gold_text: str, lang: str = "ar") -> float:
    """BERTScore F1; returns 0.0 if bert_score is not installed."""
    if not predicted_text or not gold_text:
        return 0.0
    try:
        from bert_score import score  # type: ignore
        _, _, F = score([predicted_text], [gold_text], lang=lang, verbose=False)
        return float(F[0])
    except ImportError:
        return 0.0
    except Exception as exc:
        log.warning("bertscore failed: %s", exc)
        return 0.0


def reasoning_chain_score(predicted_chain: list[str], gold_chain: list[str]) -> float:
    """Average ROUGE-L of each predicted step vs. the corresponding gold step."""
    if not gold_chain:
        return 0.0
    scores = []
    for i, gold_step in enumerate(gold_chain):
        pred_step = predicted_chain[i] if i < len(predicted_chain) else ""
        scores.append(rouge_l(pred_step, gold_step))
    return sum(scores) / len(gold_chain)


# ---------------------------------------------------------------------------
# Faithfulness metrics (lightweight — no NLI model required at metric time)
# ---------------------------------------------------------------------------

def hcr(total_cited: int, hallucinated_count: int) -> float:
    """Hallucinated Citation Rate = hallucinated / total_cited."""
    return hallucinated_count / total_cited if total_cited > 0 else 0.0


def jir(is_answerable: bool, has_foreign_signals: bool) -> float:
    """Jurisdictional Infection Rate (1.0 if answerable query has foreign signal)."""
    return 1.0 if (is_answerable and has_foreign_signals) else 0.0


def answer_faithfulness(faithfulness_score: float) -> float:
    """Pass-through for pre-computed NLI faithfulness coverage (0..1)."""
    return float(faithfulness_score)


def citation_groundedness(answer_text: str, citations: list[dict]) -> float:
    """Fraction of cited articles whose content appears (bigram-level) in answer."""
    if not citations or not answer_text:
        return 0.0
    ans_bigrams = _bigrams(answer_text.lower().split())
    grounded = 0
    texts_with_content = 0
    for c in citations:
        art_text = c.get("text", "") or c.get("supporting_span", "")
        if not art_text:
            continue
        texts_with_content += 1
        art_bigrams = _bigrams(art_text.lower().split())
        if art_bigrams & ans_bigrams:
            grounded += 1
    return grounded / texts_with_content if texts_with_content > 0 else 0.0


def _citation_ground(citation: dict) -> str:
    """Pick the Toulmin *ground* text out of a citation, accepting both
    Phase-C canonical ``argumentation.ground`` and the legacy CD
    ``adu.ground``. Returns empty string when neither key is populated.
    """
    arg = citation.get("argumentation") if isinstance(citation, dict) else None
    if isinstance(arg, dict):
        g = arg.get("ground")
        if isinstance(g, str) and g.strip():
            return g.strip()
    legacy = citation.get("adu") if isinstance(citation, dict) else None
    if isinstance(legacy, dict):
        g = legacy.get("ground")
        if isinstance(g, str) and g.strip():
            return g.strip()
    return ""


def am_faithfulness_score(
    answer_text: str,
    citations: list[dict],
    *,
    claim_threshold: float = 0.5,
) -> float:
    """Phase C — Argument-Mining faithfulness.

    For each sentence-level claim in ``answer_text``, score
    ``NLI(entailment | premise=ground, hypothesis=claim)`` against the
    Toulmin *ground* of every citation that carries one, and take the
    max. Returns the average of per-claim max scores in ``[0, 1]``.

    Conventions:
      - Empty answer text → ``0.0`` (nothing to verify, treat as
        unverified — abstention paths short-circuit upstream in
        ``_answer_to_result`` and assign ``1.0`` for trivially-faithful
        empty answers).
      - No citations OR no citation carries a Toulmin ground → ``0.0``
        (metric is not applicable; ADU was off or extraction failed).
        This is the deliberate "unverified" semantic — falling back to
        a neutral 1.0 inflates the metric on ADU-off ablations.
      - NLI model unavailable → ``0.5`` (genuine neutral).
    """
    if not answer_text or not citations:
        return 0.0
    grounds: list[str] = []
    for c in citations:
        g = _citation_ground(c)
        if g:
            grounds.append(g)
    if not grounds:
        return 0.0
    try:
        from akn_rlm.gates.faithfulness_nli import (
            _split_claims, entailment_score,
        )
    except Exception as exc:
        log.debug("am_faithfulness_score: faithfulness_nli unavailable (%s)", exc)
        return 0.5

    claims = _split_claims(answer_text)
    if not claims:
        return 0.0

    per_claim_max: list[float] = []
    for claim in claims:
        best = 0.0
        for ground in grounds:
            s = entailment_score(ground, claim)
            if s > best:
                best = s
        per_claim_max.append(best)
    return sum(per_claim_max) / len(per_claim_max)


# ---------------------------------------------------------------------------
# Abstention metrics (batch — over a full result list)
# ---------------------------------------------------------------------------

def abstention_recall(results: list[dict]) -> float:
    unanswerable = [r for r in results if r.get("gold_abstain", False)]
    if not unanswerable:
        return 0.0
    return sum(1 for r in unanswerable if r.get("predicted_abstain", False)) / len(unanswerable)


def abstention_precision(results: list[dict]) -> float:
    abstained = [r for r in results if r.get("predicted_abstain", False)]
    if not abstained:
        return 0.0
    return sum(1 for r in abstained if r.get("gold_abstain", False)) / len(abstained)


def abstention_f1(results: list[dict]) -> float:
    r = abstention_recall(results)
    p = abstention_precision(results)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def abstention_accuracy(predicted_abstain: bool, gold_abstain: bool) -> float:
    return 1.0 if predicted_abstain == gold_abstain else 0.0


# ---------------------------------------------------------------------------
# Telemetry (batch)
# ---------------------------------------------------------------------------

def telemetry_metrics(results: list[dict]) -> dict[str, float]:
    return {
        "mean_tokens_per_query":    _mean(results, "tokens_used"),
        "mean_latency_s":           _mean(results, "latency_s"),
        "corrective_retry_rate":    _mean(results, "retry_count"),
        "mean_hcr":                 _mean(results, "hcr"),
        "mean_jir":                 _mean(results, "jir"),
        "mean_faithfulness":        _mean(results, "answer_faithfulness"),
        "mean_citation_groundedness": _mean(results, "citation_groundedness"),
        "mean_am_faithfulness":     _mean(results, "am_faithfulness_score"),
    }


# ---------------------------------------------------------------------------
# Aggregate over a list of per-query result dicts
# ---------------------------------------------------------------------------

def aggregate(results: list[dict[str, Any]], k: int = 10) -> dict[str, float]:
    """Compute macro-averaged metrics over a list of per-query result dicts.

    Required keys in each result dict (existing):
      pred_doc_ids, pred_article_ids, gold_doc_ids, gold_article_ids,
      predicted_citations, gold_citations, predicted_abstain, gold_abstain

    Optional keys (Phase I extensions — default to 0.0 when absent):
      answer_text, gold_answer_text, gold_reasoning_chain, reasoning_chain,
      hcr, jir, answer_faithfulness, citation_groundedness,
      tokens_used, latency_s, retry_count
    """
    if not results:
        return {}

    totals: dict[str, float] = {
        # Retrieval
        "mrr_doc": 0.0, "mrr_article": 0.0,
        "map_doc": 0.0, "map_article": 0.0,
        "recall_doc": 0.0, "recall_article": 0.0,
        "precision_doc": 0.0, "precision_article": 0.0,
        "ndcg_doc": 0.0, "ndcg_article": 0.0,
        # Citation
        "citation_f1": 0.0, "citation_precision": 0.0, "citation_recall": 0.0,
        "doc_citation_f1": 0.0,
        # Generation
        "exact_match": 0.0, "token_f1": 0.0, "rouge_l": 0.0,
        "sacrebleu": 0.0, "bertscore_f1": 0.0, "reasoning_chain_score": 0.0,
        # Faithfulness
        "hcr": 0.0, "jir": 0.0,
        "answer_faithfulness": 0.0, "citation_groundedness": 0.0,
        "am_faithfulness_score": 0.0,
        # Abstention (per-query accuracy)
        "abstention_acc": 0.0,
    }

    for r in results:
        pred_docs    = r.get("pred_doc_ids", [])
        pred_arts    = r.get("pred_article_ids", [])
        gold_docs    = r.get("gold_doc_ids", set())
        gold_arts    = r.get("gold_article_ids", set())
        pred_cites   = r.get("predicted_citations", [])
        gold_cites   = r.get("gold_citations", [])
        pred_text    = r.get("answer_text", "")
        gold_text    = r.get("gold_answer_text", "")
        pred_chain   = r.get("reasoning_chain", [])
        gold_chain   = r.get("gold_reasoning_chain", [])

        totals["mrr_doc"]       += mrr_at_k(pred_docs, gold_docs, k)
        totals["mrr_article"]   += mrr_at_k(pred_arts, gold_arts, k)
        totals["map_doc"]       += map_at_k(pred_docs, gold_docs, k)
        totals["map_article"]   += map_at_k(pred_arts, gold_arts, k)
        totals["recall_doc"]    += recall_at_k(pred_docs, gold_docs, k)
        totals["recall_article"] += recall_at_k(pred_arts, gold_arts, k)
        totals["precision_doc"] += precision_at_k(pred_docs, gold_docs, k)
        totals["precision_article"] += precision_at_k(pred_arts, gold_arts, k)
        totals["ndcg_doc"]      += ndcg_at_k(pred_docs, gold_docs, k)
        totals["ndcg_article"]  += ndcg_at_k(pred_arts, gold_arts, k)

        totals["citation_f1"]       += citation_f1(pred_cites, gold_cites)
        totals["citation_precision"] += citation_precision(pred_cites, gold_cites)
        totals["citation_recall"]   += citation_recall(pred_cites, gold_cites)
        totals["doc_citation_f1"]   += doc_level_citation_f1(pred_cites, gold_cites)

        totals["exact_match"]           += exact_match(pred_text, gold_text)
        totals["token_f1"]              += token_f1(pred_text, gold_text)
        totals["rouge_l"]               += rouge_l(pred_text, gold_text)
        totals["sacrebleu"]             += sacrebleu_score(pred_text, gold_text)
        totals["reasoning_chain_score"] += reasoning_chain_score(pred_chain, gold_chain)

        totals["hcr"]                    += r.get("hcr", 0.0)
        totals["jir"]                    += r.get("jir", 0.0)
        totals["answer_faithfulness"]    += r.get("answer_faithfulness", 0.0)
        totals["citation_groundedness"]  += r.get("citation_groundedness", 0.0)
        totals["am_faithfulness_score"]  += r.get("am_faithfulness_score", 0.0)

        totals["abstention_acc"] += abstention_accuracy(
            r.get("predicted_abstain", False), r.get("gold_abstain", False)
        )

    n = len(results)
    out: dict[str, float] = {k_: v / n for k_, v in totals.items()}

    # Batch metrics
    out["abstention_recall"]    = abstention_recall(results)
    out["abstention_precision"] = abstention_precision(results)
    out["abstention_f1"]        = abstention_f1(results)
    out["mean_tokens_per_query"] = _mean(results, "tokens_used")
    out["mean_latency_s"]        = _mean(results, "latency_s")
    out["corrective_retry_rate"] = _mean(results, "retry_count")

    return out
