"""Benchmark runner — Phase I.

Loads AlgerianLegalBench questions, runs the pipeline, collects per-query
results, computes metrics, and optionally saves predictions + metrics to disk.

Saved files (when output_dir provided):
  {output_dir}/{run_id}/predictions.jsonl   — one JSON object per line
  {output_dir}/{run_id}/metrics.json        — full stratified metrics tree
  {output_dir}/{run_id}/metrics.md          — markdown table for quick review

Benchmark JSONL schema:
  {"id": "q001", "query": "...", "query_type": "rule_application",
   "legal_category": "family_law", "difficulty": "medium",
   "language": "ar", "split": "test",
   "temporal_note": null,
   "gold_doc_ids": ["84-11_1984-06-09"],
   "gold_article_ids": ["84-11_1984-06-09#art_54"],
   "gold_citations": [{"doc_id": "84-11_1984-06-09", "article_ref": "54"}],
   "gold_abstain": false,
   "gold_answer": "...",
   "gold_reasoning_chain": ["step 1", "step 2", "step 3", "step 4"]}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-query result builder
# ---------------------------------------------------------------------------

def _answer_to_result(
    question: dict,
    answer: dict,
) -> dict[str, Any]:
    """Convert a pipeline answer + gold question into a metrics-ready result dict."""
    from akn_rlm.normalizers import canonical_article_ref

    citations = answer.get("citations", []) or []
    telemetry = answer.get("_telemetry", {})
    gate_results = telemetry.get("gate_results", {})

    # Build ranked doc / article ID lists from citations. Canonicalise
    # article_ref so prediction art_keys match gold art_keys regardless of
    # the surface form the LLM emitted.
    seen_docs: list[str] = []
    seen_arts: list[str] = []
    for c in citations:
        doc_id = c.get("doc_id", "")
        art_ref = canonical_article_ref(str(c.get("article_ref", "")))
        art_key = f"{doc_id}#art_{art_ref}"
        if doc_id not in seen_docs:
            seen_docs.append(doc_id)
        if art_key not in seen_arts:
            seen_arts.append(art_key)

    # HCR: hallucinated citation rate (rejected / total_tried)
    cit_gate = gate_results.get("citation", {})
    rejected = cit_gate.get("rejected", [])
    total_tried = len(citations) + len(rejected)
    hcr_val = len(rejected) / total_tried if total_tried > 0 else 0.0

    # JIR: 1.0 if answerable query has foreign-law signal in answer
    from akn_rlm.gates.jurisdiction import is_infected
    answer_text = answer.get("answer_text") or ""
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
    gold_abstain = bool(question.get("gold_abstain", False))
    jir_val = 1.0 if (not gold_abstain and is_infected(answer_text)) else 0.0

    # Faithfulness score from NLI gate (stored in telemetry)
    faith_gate = gate_results.get("faithfulness", {})
    faith_score = faith_gate.get("score", 0.0) if not answer.get("abstention") else 1.0

    # Citation groundedness (bigram overlap proxy — no NLI required)
    from akn_rlm.eval.metrics import am_faithfulness_score, citation_groundedness
    groundedness = citation_groundedness(answer_text, citations)
    # Phase C — Toulmin Argument-Mining faithfulness. Skipped (treated as
    # 1.0) when the question was abstained, in line with how the legacy
    # faithfulness signal handles abstention.
    if answer.get("abstention"):
        am_faith = 1.0
    else:
        am_faith = am_faithfulness_score(answer_text, citations)

    # Store raw (pre-gate) citations when AKN_NO_CITATION_GATE is active so
    # offline HCR computation can access the unfiltered LLM output.
    import os as _os_runner
    raw_summary_citations = answer.get("_raw_citations", []) if _os_runner.getenv("AKN_NO_CITATION_GATE") else []

    return {
        # ── Retrieval fields (for metrics.aggregate) ──────────────────────
        "pred_doc_ids":        seen_docs,
        "pred_article_ids":    seen_arts,
        "gold_doc_ids":        sorted(set(question.get("gold_doc_ids", []))),
        "gold_article_ids":    sorted(set(question.get("gold_article_ids", []))),
        "predicted_citations": citations,
        "gold_citations":      question.get("gold_citations", []),
        "predicted_abstain":   bool(answer.get("abstention", False)),
        "gold_abstain":        gold_abstain,
        # ── Generation fields ─────────────────────────────────────────────
        "answer_text":         answer_text,
        "gold_answer_text":    question.get("gold_answer", ""),
        "reasoning_chain":     answer.get("reasoning_chain", []),
        "gold_reasoning_chain": question.get("gold_reasoning_chain", []),
        "trajectory":          answer.get("trajectory", []),
        # ── Faithfulness fields ──────────────────────────────────────────
        "hcr":                 hcr_val,
        "jir":                 jir_val,
        "answer_faithfulness": faith_score,
        "citation_groundedness": groundedness,
        "am_faithfulness_score": am_faith,
        # ── Stratification keys ───────────────────────────────────────────
        "query_type":          question.get("query_type", "rule_application"),
        "legal_category":      question.get("legal_category", "unknown"),
        "difficulty":          question.get("difficulty", "unknown"),
        "language":            question.get("language", "ar"),
        "split":               question.get("split", "test"),
        "has_temporal_note":   bool(question.get("temporal_note")),
        # ── Telemetry ─────────────────────────────────────────────────────
        "question_id":         question.get("id", ""),
        "query":               question.get("query", ""),
        "abstention_reason":   answer.get("abstention_reason"),
        "tokens_used":         answer.get("tokens_used", 0),
        "latency_s":           answer.get("_latency_s", 0.0),
        "retry_count":         telemetry.get("retry_count", 0),
        # ── R9.7 per-handler-run telemetry ────────────────────────────────
        "sub_call_count":      telemetry.get("sub_call_count", 0),
        "calls_by_model":      telemetry.get("calls_by_model", {}),
        "supervisor_used":     telemetry.get("supervisor_used", False),
        "dispatched_handler":  telemetry.get("dispatched_handler"),
        # ── Gate bypass (only populated when AKN_NO_CITATION_GATE=1) ─────
        "raw_summary_citations": raw_summary_citations,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _format_markdown(strata: dict[str, Any]) -> str:
    overall = strata.get("overall", {})
    counts = strata.get("counts", {})
    lines = [
        "# Evaluation Metrics\n",
        f"**Total questions:** {counts.get('total', '?')}\n",
        "## Overall\n",
        "| Metric | Score |",
        "|--------|------:|",
    ]
    for key in sorted(overall):
        val = overall[key]
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.4f} |")
    lines.append("")

    for stratum_name, stratum_data in [
        ("By Query Type", strata.get("by_query_type", {})),
        ("By Difficulty", strata.get("by_difficulty", {})),
        ("By Language", strata.get("by_language", {})),
    ]:
        if not stratum_data:
            continue
        lines.append(f"## {stratum_name}\n")
        rows = sorted(stratum_data.items())
        if rows:
            header_keys = ["mrr_doc", "mrr_article", "citation_f1",
                           "abstention_recall", "abstention_f1"]
            lines.append("| Stratum | " + " | ".join(header_keys) + " |")
            lines.append("|---------|" + "|".join(["------:"] * len(header_keys)) + "|")
            for label, m in rows:
                vals = " | ".join(f"{m.get(k, 0.0):.4f}" for k in header_keys)
                lines.append(f"| {label} | {vals} |")
        lines.append("")

    return "\n".join(lines)


def _save_run(
    results: list[dict],
    strata: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # predictions.jsonl
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # metrics.json
    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(strata, fh, ensure_ascii=False, indent=2, default=_default)

    # metrics.md
    md_path = out_dir / "metrics.md"
    md_path.write_text(_format_markdown(strata), encoding="utf-8")

    log.info("Saved run to %s  (%d questions)", out_dir, len(results))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    pipeline,
    benchmark_path: str | Path,
    *,
    limit: int | None = None,
    query_types: list[str] | None = None,
    difficulty: str | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline against every matching question in benchmark_path.

    Args:
        pipeline      : compiled LangGraph pipeline with .run(query) method.
        benchmark_path: path to JSONL benchmark file.
        limit         : stop after N questions (useful for quick iteration).
        query_types   : if set, only questions with matching query_type.
        difficulty    : if set, only questions with matching difficulty.
        output_dir    : if set, write predictions + metrics to this directory.
        run_id        : subdirectory under output_dir (defaults to timestamp).

    Returns:
        list of per-query result dicts (passable to metrics.aggregate / stratify).
    """
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    questions: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if query_types:
        questions = [q for q in questions if q.get("query_type") in query_types]
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    if limit:
        questions = questions[:limit]

    log.info("Benchmark: %d questions from %s", len(questions), path.name)

    results: list[dict] = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            answer = pipeline.run(q["query"])
        except Exception as exc:
            log.error("Question %s failed: %s", q.get("id", i), exc)
            answer = {
                "answer_text": "",
                "abstention": True,
                "abstention_reason": "pipeline_error",
                "citations": [],
            }
        answer["_latency_s"] = time.time() - t0
        results.append(_answer_to_result(q, answer))

        if (i + 1) % 10 == 0:
            log.info("Progress: %d/%d", i + 1, len(questions))

    if output_dir:
        from akn_rlm.eval.stratified import stratify
        strata = stratify(results)
        _run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        _save_run(results, strata, Path(output_dir) / _run_id)

    return results
