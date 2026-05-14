"""Run AlgerianLegalBench v3.0 through the Phase-2 / R7 dispatcher.

The dispatcher routes each question to its native typed handler
(R2-R6) based on the benchmark's ``query_type`` field — the same field
already produced by ``_benchmark_to_records``. Records carry the
``query_type`` directly, so no classifier call is needed on the smoke
path; the classifier is only used as a safety net if the field is
missing.

Output layout matches ``run_baseline_*.py`` / ``run_handler_*.py`` so
``scripts/compare_baselines.py`` can read it directly:

    eval_results/{run_id}/predictions.jsonl
    eval_results/{run_id}/metrics.json
    eval_results/{run_id}/metrics.md
    eval_results/{run_id}/report.txt

Usage:
    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"

    # Stratified-2 smoke (16 q across all 8 types — the R7 gate)
    & $py scripts\\run_dispatcher.py --stratified 2 \\
          --run-id rlm_dispatched_smoke

    # Full 244-q benchmark
    & $py scripts\\run_dispatcher.py --run-id rlm_dispatched_full

    # Force a specific query_type (everything routes to that handler)
    & $py scripts\\run_dispatcher.py --query-types multi_hop \\
          --run-id rlm_dispatched_mh

    # Disable KG (skip the ~26 s parse — temporal/conceptual then
    # error out and abstain via dispatch_build_error)
    & $py scripts\\run_dispatcher.py --no-kg --stratified 2 \\
          --run-id rlm_dispatched_no_kg
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.config import (  # noqa: E402
    BM25_INDEX_PATH,
    DENSE_FAISS_PATH,
    DENSE_META_PATH,
    SUB_LLM_MODEL,
    get_benchmark_path,
)
from akn_rlm.corpus.akn_parser import parse_all  # noqa: E402
from akn_rlm.corpus.article_registry import ArticleRegistry  # noqa: E402
from akn_rlm.eval.report import format_report, print_report  # noqa: E402
from akn_rlm.eval.runner import _answer_to_result, _format_markdown  # noqa: E402
from akn_rlm.eval.stratified import stratify  # noqa: E402
from akn_rlm.indexers.bm25 import BM25Index  # noqa: E402
from akn_rlm.indexers.dense import DenseIndex  # noqa: E402
from akn_rlm.llm.client import LLMPool  # noqa: E402
from akn_rlm.rlm.dispatcher import (  # noqa: E402
    DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S,
    build_dispatcher,
)
from akn_rlm.rlm.handlers import LAYMAN_DEFAULT_REWRITE_MODEL  # noqa: E402
from akn_rlm.rlm.routing import build_doc_router  # noqa: E402

# Reuse the benchmark-format converter + stratified sampler from run_benchmark
from scripts.run_benchmark import (  # noqa: E402
    _benchmark_to_records,
    _stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_dispatcher")


def _run(
    dispatcher,
    records: list[dict],
    *,
    limit: int | None,
    query_types: list[str] | None,
    difficulty: str | None,
    stratified: int | None,
    output_dir: Path,
    run_id: str,
    no_gold_type: bool = False,
    show_trajectory: bool = False,
    show_trajectory_n: int = 3,
) -> list[dict]:
    questions = records
    if query_types:
        questions = [q for q in questions if q.get("query_type") in query_types]
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    if stratified:
        questions = _stratified_sample(questions, stratified)
        log.info(
            "Stratified sample: %d questions across %d types",
            len(questions),
            len({q.get("query_type") for q in questions}),
        )
    if limit:
        questions = questions[:limit]

    log.info("Running %d questions through RLM dispatcher …", len(questions))

    results: list[dict] = []
    for i, q in enumerate(questions):
        t0 = time.time()
        # Phase B: when --no-gold-type is set we deliberately *do not*
        # pass the benchmark's gold query_type so the dispatcher must
        # fall back to its classifier_fn. The record itself keeps its
        # gold query_type so _answer_to_result can still stratify by
        # gold type in the metrics.
        dispatched_type = None if no_gold_type else q.get("query_type")
        try:
            answer = dispatcher.run(
                q["query"], query_type=dispatched_type,
            )
        except Exception as exc:
            log.error("Q%s failed: %s", q.get("id", i), exc)
            answer = {
                "answer_text": "",
                "abstention": True,
                "abstention_reason": "dispatcher_runner_error",
                "citations": [],
                "_telemetry": {"baseline": "rlm_dispatched"},
            }
        answer["_latency_s"] = time.time() - t0
        results.append(_answer_to_result(q, answer))

        # Per-question line so a hung handler is visible immediately.
        tel = answer.get("_telemetry", {})
        log.info(
            "  Q%s  type=%s  handler=%s  abstain=%s  lat=%.2fs  depth=%d",
            q.get("id", i),
            q.get("query_type", "?"),
            tel.get("dispatched_handler", "?"),
            answer.get("abstention", False),
            answer["_latency_s"],
            tel.get("recursion_depth_max", answer.get("depth_max_reached", 1)),
        )
        # Phase D — log the first few trajectories for sanity inspection.
        # Use log.info (routed through whatever stdout wandb is capturing)
        # rather than bare print, which crashes when wandb has closed
        # the underlying file descriptor.
        if show_trajectory and i < show_trajectory_n:
            trj = answer.get("trajectory") or []
            log.info(
                "  -- Trajectory for Q%s (handler=%s) --",
                q.get("id", i), tel.get("dispatched_handler"),
            )
            for step in trj:
                step_name = step.get("step", "?")
                depth = step.get("depth", 0)
                detail = {k: v for k, v in step.items() if k not in {"step", "depth", "sub_question"}}
                sub_q = step.get("sub_question") or ""
                sub_q_short = (sub_q[:60] + "...") if len(sub_q) > 60 else sub_q
                log.info("    [d%d] %s  sub_q=%r  %s", depth, step_name, sub_q_short, detail)

    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log.info("Predictions saved → %s", pred_path)

    strata = stratify(results)

    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(strata, fh, ensure_ascii=False, indent=2, default=_default)

    md_path = out_dir / "metrics.md"
    md_path.write_text(_format_markdown(strata), encoding="utf-8")

    report_path = out_dir / "report.txt"
    report_path.write_text(
        format_report(strata, title=f"RLM dispatcher  {run_id}"),
        encoding="utf-8",
    )

    log.info("All results saved to %s/", out_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlgerianLegalBench v3.0 through the Phase-2 dispatcher"
    )
    parser.add_argument("--benchmark", default=None,
                        help="Path to AlgerianLegalBench JSON file (default: auto-detect)")
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
        help="Restrict the run to these query types (default: all 8)",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--sub-model",     default=SUB_LLM_MODEL,
                        help=f"Sub-LM model name (default: {SUB_LLM_MODEL}).")
    parser.add_argument("--rewrite-model", default=LAYMAN_DEFAULT_REWRITE_MODEL,
                        help=f"Gemma Darja→MSA rewrite model for the layman "
                             f"handler (default: {LAYMAN_DEFAULT_REWRITE_MODEL}).")
    parser.add_argument("--long-context-timeout", type=float,
                        default=DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S,
                        help=f"Wall-clock seconds before the long_context "
                             f"summariser is aborted and the handler falls "
                             f"back to the deterministic Arabic template "
                             f"(default: {DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S:.1f}s).")
    parser.add_argument("--no-kg", action="store_true",
                        help="Skip KG load. temporal_factual / conceptual_definitional "
                             "queries will then abstain with `dispatch_build_error`.")
    parser.add_argument("--ceiling-breakers", action="store_true",
                        help="Enable the 5 HPC-grade ceiling-breaker upgrades: "
                             "BGE-m3 dense + BGE-reranker-v2-m3 (via env), "
                             "per-citation NLI verifier, doc-router LLM tie-breaker, "
                             "and concept->amendment SPARQL helper for CD. "
                             "Equivalent to AKN_CEILING_BREAKERS=1 env var.")
    parser.add_argument("--e1", action="store_true",
                        help="E1: union concept->amendment SPARQL hits into the "
                             "CD handler's KG-bias set (AKN_E1_CONCEPT_AMENDMENT=1).")
    parser.add_argument("--e2", action="store_true",
                        help="E2: reverse-direction NLI verifier (Gemma rewrites "
                             "question as declarative; NLI scores entailment of "
                             "article -> claim). Replaces F5 LLM verifier in RA/MH/EA. "
                             "(AKN_E2_NLI_REVERSE=1).")
    parser.add_argument("--e3", action="store_true",
                        help="E3: Gemma paraphrase pre-retrieval; BM25 + Dense run "
                             "over [original, paraphrases] and RRF-merge "
                             "(AKN_E3_PARAPHRASE=1).")
    parser.add_argument("--e4", action="store_true",
                        help="E4: HyDE retrieval. Qwen drafts a hypothetical answer; "
                             "dense embeds query+answer. (AKN_E4_HYDE=1).")
    parser.add_argument("--e5", action="store_true",
                        help="E5: KG topology disambiguator (Phase E.2). Multi-hop "
                             "candidates with similar consensus-boost scores in the "
                             "same doc are re-ranked using chapter/section title "
                             "concept match. Falls back to F5 ranking if KG is "
                             "unavailable. (AKN_E5_KG_TOPOLOGY=1).")
    parser.add_argument("--e6", action="store_true",
                        help="E6: concept-KG retrieval channel for MH/RA (Phase E.3). "
                             "Adds a SPARQL-CONTAINS channel that surfaces articles "
                             "whose any-version text contains a query phrase, alongside "
                             "BM25/Dense in the per-sub-q (MH) or per-query (RA) RRF "
                             "pool. Falls back to hybrid-only when KG is unavailable. "
                             "(AKN_E6_CONCEPT_KG=1).")
    parser.add_argument("--e7", action="store_true",
                        help="E7: KG-derived doc-router channel (Phase E.4). The doc-"
                             "router gains a 4th channel that queries the KG for docs "
                             "containing articles with the query's concept phrases, "
                             "then adds a 0.5 bonus per matched doc. Lazy-loads KG on "
                             "first dispatch. (AKN_E7_KG_DOC_ROUTER=1).")
    parser.add_argument("--enhancers-all", action="store_true",
                        help="Enable E1+E2+E3+E4 simultaneously (AKN_ENHANCERS=all).")
    parser.add_argument("--no-gold-type", action="store_true",
                        help="Phase B: strip the benchmark's gold query_type from each "
                             "record before dispatch. Forces the dispatcher to fall "
                             "back to its classifier_fn (LLM by default — see "
                             "--classifier and --classifier-model). Measures the "
                             "real-world drop when query_type isn't known a priori.")
    parser.add_argument("--classifier", choices=["llm", "regex"], default="llm",
                        help="Classifier used when --no-gold-type is set. 'llm' uses "
                             "Gemma via classifier.llm_classify (Phase B target); "
                             "'regex' uses the deterministic akn_rlm.rlm.classifier.classify "
                             "(default: llm).")
    parser.add_argument("--classifier-model", default=None,
                        help="Override model name for the LLM classifier (default: "
                             "google/gemma-4-31B routed via the LLMPool 'gemma' slot).")
    parser.add_argument("--adu", dest="adu", action="store_true", default=True,
                        help="Phase C: enable pervasive Toulmin ADU extraction "
                             "on every citation-emitting handler (default: ON).")
    parser.add_argument("--no-adu", dest="adu", action="store_false",
                        help="Phase C ablation: disable pervasive ADU on the "
                             "dispatched path. CD still runs its own ADU.")
    parser.add_argument("--adu-top-n", type=int, default=5,
                        help="Phase C: number of top citations per query that "
                             "receive ADU extraction (default: 5).")
    parser.add_argument("--recursion", dest="recursion", action="store_true", default=False,
                        help="Phase D: enable gap-driven recursion in RA / MH / "
                             "TF / CD handlers. The gap-probe (gpt-oss-120b) "
                             "decides whether to issue depth-2/3 retrieval "
                             "passes; new candidates merge additively. (Default OFF.)")
    parser.add_argument("--no-recursion", dest="recursion", action="store_false",
                        help="Phase D ablation: force depth-1-only behaviour.")
    parser.add_argument("--recursion-max-depth", type=int, default=3,
                        help="Phase D: maximum recursion depth (default: 3 = original "
                             "+ 2 re-retrieval passes).")
    parser.add_argument("--mh-ra-coverage-min", type=int, default=None,
                        help="Phase E: override recursion_coverage_min for the "
                             "multi_hop and rule_application handlers. Default None "
                             "(handlers use DEFAULT_COVERAGE_MIN=2). Phase D measured "
                             "MH/RA regressing despite high recursion firing rates; "
                             "raising this to 4 only fires the gap-probe on genuinely "
                             "thin pools while leaving TF/CD untouched.")
    parser.add_argument("--corrective-retry", dest="corrective_retry", action="store_true", default=False,
                        help="Phase D: enable corrective retry on faithfulness gate "
                             "failure. After the summariser, run gates.faithfulness_nli; "
                             "on fail, regenerate ONCE with feedback. (Default OFF.)")
    parser.add_argument("--no-corrective-retry", dest="corrective_retry", action="store_false",
                        help="Phase D ablation: skip the post-summary faithfulness gate "
                             "+ retry.")
    parser.add_argument("--show-trajectory", action="store_true",
                        help="Phase D: print the per-step trajectory of the first 3 "
                             "questions to stdout for sanity inspection.")
    args = parser.parse_args()
    # Convert per-flag CLI toggles to env vars so the enhancers module
    # picks them up uniformly regardless of activation path.
    import os as _os
    if args.enhancers_all: _os.environ["AKN_ENHANCERS"] = "all"
    if args.e1: _os.environ["AKN_E1_CONCEPT_AMENDMENT"] = "1"
    if args.e2: _os.environ["AKN_E2_NLI_REVERSE"] = "1"
    if args.e3: _os.environ["AKN_E3_PARAPHRASE"] = "1"
    if args.e4: _os.environ["AKN_E4_HYDE"] = "1"
    if args.e5: _os.environ["AKN_E5_KG_TOPOLOGY"] = "1"
    if args.e6: _os.environ["AKN_E6_CONCEPT_KG"] = "1"
    if args.e7: _os.environ["AKN_E7_KG_DOC_ROUTER"] = "1"

    run_id = args.run_id or time.strftime("rlm_dispatched_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("RLM dispatcher  run_id=%s", run_id)
    log.info("Benchmark      : %s", benchmark_path)
    log.info("Output         : %s/%s/", output_dir, run_id)
    log.info("Sub-LM model   : %s", args.sub_model)
    log.info("Rewrite model  : %s", args.rewrite_model)
    log.info("LC timeout     : %.1fs", args.long_context_timeout)
    log.info("KG             : %s", "OFF" if args.no_kg else "lazy-load on first KG dispatch")
    log.info("Gold query_type: %s",
             "OFF (classifier=%s)" % args.classifier if args.no_gold_type else "ON")
    log.info("=" * 64)

    log.info("Parsing corpus to build registry …")
    registry = ArticleRegistry()
    registry.build(parse_all())

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {BM25_INDEX_PATH} — run scripts/build_indices.py first"
        )
    if not DENSE_FAISS_PATH.exists() or not DENSE_META_PATH.exists():
        raise FileNotFoundError(
            "Dense index missing — run scripts/build_indices.py first"
        )

    log.info("Loading BM25 index …")
    bm25 = BM25Index.load(BM25_INDEX_PATH)
    log.info("Loading dense index …")
    dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)

    log.info("Connecting to LLM pool …")
    llm_pool = LLMPool.default()

    # Phase B: optionally build a classifier_fn so dispatcher.run() with
    # query_type=None falls back to it. Default is the Gemma-backed
    # llm_classify (Phase B target); --classifier=regex keeps the
    # deterministic regex path for ablation.
    classifier_fn = None
    if args.no_gold_type:
        if args.classifier == "llm":
            from akn_rlm.rlm.classifier import (  # noqa: PLC0415
                DEFAULT_LLM_CLASSIFIER_MODEL,
                make_llm_classifier_fn,
            )
            cls_model = args.classifier_model or DEFAULT_LLM_CLASSIFIER_MODEL
            classifier_fn = make_llm_classifier_fn(llm_pool, model=cls_model)
            log.info("--no-gold-type: classifier=LLM (model=%s)", cls_model)
        else:
            from akn_rlm.rlm.classifier import classify as _regex_classify  # noqa: PLC0415
            classifier_fn = lambda q: _regex_classify(q).query_type  # noqa: E731
            log.info("--no-gold-type: classifier=regex (deterministic)")
    else:
        log.info("Using benchmark gold query_type for dispatch routing.")

    # When ceiling-breakers are on, the dispatcher builds its own router
    # with the LLM tie-breaker channel wired in. Pass router=None below
    # so the dispatcher constructs one internally.
    if args.ceiling_breakers:
        log.info("Ceiling-breakers ENABLED — dispatcher will build router with LLM tie-breaker.")
        router = None
    else:
        log.info("Building doc-router (deterministic; pass --ceiling-breakers to enable LLM channel) …")
        router = build_doc_router(registry=registry, bm25=bm25)

    kg_loader = None
    if not args.no_kg:
        def _load_kg():
            log.info("Loading KG (rdflib parse, ~26 s on first KG dispatch) …")
            from akn_rlm.corpus.kg_loader import load_kg
            return load_kg()
        kg_loader = _load_kg

    log.info("Pervasive ADU: %s (top-n=%d)",
             "ON" if args.adu else "OFF", args.adu_top_n)
    log.info("Recursion     : %s (max-depth=%d)",
             "ON" if args.recursion else "OFF", args.recursion_max_depth)
    log.info("Corrective    : %s",
             "ON" if args.corrective_retry else "OFF")

    coverage_overrides: dict[str, int] | None = None
    if args.mh_ra_coverage_min is not None:
        coverage_overrides = {
            "multi_hop":        args.mh_ra_coverage_min,
            "rule_application": args.mh_ra_coverage_min,
        }
        log.info("MH/RA coverage_min override: %d (TF/CD keep default 2)",
                 args.mh_ra_coverage_min)

    dispatcher = build_dispatcher(
        bm25=bm25, dense=dense, registry=registry,
        llm_pool=llm_pool, router=router,
        kg=None,
        kg_loader=kg_loader,
        sub_model=args.sub_model,
        rewrite_model=args.rewrite_model,
        long_context_timeout_s=args.long_context_timeout,
        enable_ceiling_breakers=args.ceiling_breakers or None,
        classifier_fn=classifier_fn,
        enable_pervasive_adu=args.adu,
        adu_extract_top_n=args.adu_top_n,
        enable_recursion=args.recursion,
        recursion_max_depth=args.recursion_max_depth,
        enable_corrective_retry=args.corrective_retry,
        recursion_coverage_min_overrides=coverage_overrides,
    )

    results = _run(
        dispatcher,
        records=_benchmark_to_records(benchmark_path, registry),
        limit=args.limit,
        query_types=args.query_types,
        difficulty=args.difficulty,
        stratified=args.stratified,
        output_dir=output_dir,
        run_id=run_id,
        no_gold_type=args.no_gold_type,
        show_trajectory=args.show_trajectory,
    )

    strata = stratify(results)
    try:
        print_report(strata, title=f"RLM dispatcher  {run_id}")
    except (ValueError, OSError) as exc:
        # wandb's stdout capture occasionally closes the file descriptor
        # before this point; the report is already written to report.txt
        # so we just log a notice and continue.
        log.warning("print_report stdout failed (%s); see report.txt for full table", exc)
    log.info("Done. Results in %s/%s/", output_dir, run_id)


if __name__ == "__main__":
    main()
