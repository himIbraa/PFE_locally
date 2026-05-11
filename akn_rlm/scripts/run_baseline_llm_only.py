"""Run AlgerianLegalBench v3.0 against the LLM-only baseline pipeline.

This is the **floor** baseline: sends the query directly to a configured
LLM and asks it to emit citations. No retrieval (raw) or minimal RAG
(--with-context).

Usage:
    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"

    # Raw gpt-oss-120b, no retrieval (~30 min for full 244 — LLM bound)
    & $py scripts\\run_baseline_llm_only.py --model gpt-oss-120b \\
        --run-id baseline_gpt_oss_raw_full

    # Raw Qwen3
    & $py scripts\\run_baseline_llm_only.py --model Qwen3-30B-A3B-Thinking \\
        --run-id baseline_qwen3_raw_full

    # With dense top-5 retrieval context
    & $py scripts\\run_baseline_llm_only.py --model gpt-oss-120b --with-dense \\
        --run-id baseline_gpt_oss_with_dense5_full

    # With dense + KG amendment chain context for any cited article
    & $py scripts\\run_baseline_llm_only.py --model gpt-oss-120b --with-dense --with-kg \\
        --run-id baseline_gpt_oss_with_dense5_kg_full
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

# UTF-8 console on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.baselines.llm_only_pipeline import build_llm_only_pipeline  # noqa: E402
from akn_rlm.config import (  # noqa: E402
    BM25_INDEX_PATH, DENSE_FAISS_PATH, DENSE_META_PATH, get_benchmark_path,
)
from akn_rlm.corpus.akn_parser import parse_all  # noqa: E402
from akn_rlm.corpus.article_registry import ArticleRegistry  # noqa: E402
from akn_rlm.eval.report import format_report, print_report  # noqa: E402
from akn_rlm.eval.runner import _answer_to_result, _format_markdown  # noqa: E402
from akn_rlm.eval.stratified import stratify  # noqa: E402
from akn_rlm.indexers.bm25 import BM25Index  # noqa: E402
from akn_rlm.indexers.dense import DenseIndex  # noqa: E402
from akn_rlm.llm.client import LLMPool  # noqa: E402
from akn_rlm.normalizers import canonical_article_ref  # noqa: E402

from scripts.run_benchmark import (  # noqa: E402
    _benchmark_to_records,
    _stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_baseline_llm_only")


# ---------------------------------------------------------------------------
# Context providers
# ---------------------------------------------------------------------------

def _make_dense_context_provider(dense: DenseIndex, registry: ArticleRegistry, top_k: int = 5):
    """Return ctx(query) -> list[{doc_id, article_ref, text}]."""

    def _provider(query: str):
        try:
            hits = dense.search(query, k=top_k)
        except Exception as exc:
            log.warning("dense.search failed: %s", exc)
            return []
        out = []
        seen: set[tuple[str, str]] = set()
        for h in hits:
            ref = canonical_article_ref(h.article_ref) or h.article_ref
            key = (h.doc_id, ref)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "doc_id":      h.doc_id,
                "article_ref": ref,
                "text":        h.text or "",
            })
        return out

    return _provider


def _make_dense_plus_kg_context_provider(
    dense: DenseIndex, registry: ArticleRegistry, kg, top_k: int = 5,
):
    """Dense top-K, then enrich each candidate with its KG amendment chain text."""
    base = _make_dense_context_provider(dense, registry, top_k=top_k)

    def _provider(query: str):
        items = base(query)
        if not items:
            return items
        try:
            from rdflib import Literal  # type: ignore
        except Exception:
            return items
        for it in items:
            # Append the latest version text from the amendment chain.
            doc_id = it.get("doc_id", "")
            ref = it.get("article_ref", "")
            extra = _kg_amendment_text(kg, doc_id, ref)
            if extra:
                it["text"] = (it.get("text", "") or "") + "\n\n" + extra
        return items

    return _provider


_KG_LATEST_VERSION_SPARQL = """\
PREFIX dzdoc: <https://legal.dz/ontology/document#>

SELECT ?text ?inForceFrom WHERE {
  ?article dzdoc:hasVersion ?v .
  ?v dzdoc:versionText ?text ;
     dzdoc:inForceFrom ?inForceFrom .
  FILTER(CONTAINS(STR(?article), CONCAT("/", "%s", "/")))
  FILTER(CONTAINS(STR(?article), CONCAT("art_", "%s")))
}
ORDER BY DESC(?inForceFrom)
LIMIT 1
"""


def _kg_amendment_text(kg, doc_id: str, ref: str) -> str:
    """Best-effort: ask the KG for the latest version text of (doc_id, ref).
    Returns empty string on any error / no match.
    """
    if kg is None or not doc_id or not ref:
        return ""
    # doc_id pattern: "84-11_1984-06-09" — extract the num + date
    parts = doc_id.split("_", 1)
    if len(parts) != 2:
        return ""
    num, date = parts
    safe_ref = ref.replace('"', '\\"')
    sparql = _KG_LATEST_VERSION_SPARQL % (date + "/" + num, safe_ref)
    try:
        rows = list(kg.query(sparql))
    except Exception as exc:
        log.debug("KG enrichment SPARQL failed: %s", exc)
        return ""
    if not rows:
        return ""
    try:
        return str(rows[0][0])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def _run(
    pipeline,
    records: list[dict],
    *,
    limit: int | None,
    query_types: list[str] | None,
    difficulty: str | None,
    stratified: int | None,
    output_dir: Path,
    run_id: str,
) -> list[dict]:
    questions = records
    if query_types:
        questions = [q for q in questions if q.get("query_type") in query_types]
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    if stratified:
        questions = _stratified_sample(questions, stratified)
    if limit:
        questions = questions[:limit]

    log.info("Running %d questions through LLM-only baseline ...", len(questions))

    results: list[dict] = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            answer = pipeline.run(q["query"])
        except Exception as exc:
            log.error("Q%s failed: %s", q.get("id", i), exc)
            answer = {
                "answer_text": "",
                "abstention": True,
                "abstention_reason": "pipeline_error",
                "citations": [],
            }
        answer["_latency_s"] = time.time() - t0
        results.append(_answer_to_result(q, answer))
        log.info(
            "Q%-15s type=%-22s n_cit=%d  lat=%.2fs",
            q.get("id", "?"), q.get("query_type", ""),
            len(answer.get("citations") or []), answer["_latency_s"],
        )

    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log.info("Predictions -> %s", pred_path)

    strata = stratify(results)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(strata, fh, ensure_ascii=False, indent=2, default=str)
    log.info("Metrics -> %s", metrics_path)

    md_path = out_dir / "metrics.md"
    md_path.write_text(_format_markdown(strata), encoding="utf-8")

    report_path = out_dir / "report.txt"
    report_path.write_text(
        format_report(strata, title=f"LLM-only baseline  {run_id}"),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlgerianLegalBench v3.0 against an LLM-only baseline"
    )
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified", type=int, default=None)
    parser.add_argument(
        "--query-types", nargs="*",
        choices=[
            "rule_application", "exact_article", "multi_hop", "unanswerable",
            "layman", "long_context", "conceptual_definitional", "temporal_factual",
        ],
        default=None,
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="LLM model name (gpt-oss-120b | Qwen3-30B-A3B-Thinking | google/gemma-4-31B)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--with-dense", action="store_true",
                        help="Prepend dense top-5 article texts to the prompt (minimal RAG).")
    parser.add_argument("--with-kg", action="store_true",
                        help="If --with-dense: append each context article's latest KG version text. Needs KG.")
    parser.add_argument("--top-k-context", type=int, default=5)
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("baseline_llm_only_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("LLM-only baseline  run_id=%s", run_id)
    log.info("Model       : %s", args.model)
    log.info("With dense  : %s", args.with_dense)
    log.info("With KG     : %s", args.with_kg)
    log.info("=" * 64)

    log.info("Parsing corpus to build registry ...")
    registry = ArticleRegistry()
    registry.build(parse_all())

    records = _benchmark_to_records(benchmark_path, registry)

    llm_pool = LLMPool.default()

    context_provider = None
    telemetry_suffix = "raw"
    if args.with_dense:
        if not DENSE_FAISS_PATH.exists() or not DENSE_META_PATH.exists():
            raise FileNotFoundError(
                "Dense index missing — run scripts/build_indices.py first"
            )
        log.info("Loading dense index ...")
        dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)
        if args.with_kg:
            log.info("Loading KG for context enrichment (~26s) ...")
            from akn_rlm.corpus.kg_loader import load_kg
            kg = load_kg()
            context_provider = _make_dense_plus_kg_context_provider(
                dense, registry, kg, top_k=args.top_k_context,
            )
            telemetry_suffix = f"dense{args.top_k_context}_kg"
        else:
            context_provider = _make_dense_context_provider(
                dense, registry, top_k=args.top_k_context,
            )
            telemetry_suffix = f"dense{args.top_k_context}"

    pipeline = build_llm_only_pipeline(
        llm_pool=llm_pool, registry=registry,
        model=args.model, max_tokens=args.max_tokens,
        temperature=args.temperature,
        context_provider=context_provider,
        telemetry_suffix=telemetry_suffix,
    )

    _run(
        pipeline, records,
        limit=args.limit,
        query_types=args.query_types,
        difficulty=args.difficulty,
        stratified=args.stratified,
        output_dir=output_dir,
        run_id=run_id,
    )

    log.info("Done. Results in %s/%s/", output_dir, run_id)


if __name__ == "__main__":
    main()
