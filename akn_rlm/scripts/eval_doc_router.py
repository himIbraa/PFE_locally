"""Doc-router smoke evaluation against AlgerianLegalBench v3.0.

Measures the R1 gate: ``top-3 doc recall >= 80%`` on the stratified
sample.  For each question the router predicts up to N=3 doc_ids; a
question is counted as covered when at least one gold doc_id appears in
the prediction set.

Usage:
    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\eval_doc_router.py --stratified 5
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\eval_doc_router.py            # full 244 q
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\eval_doc_router.py --top-n 1   # tighter

Output: a markdown report with per-query-type recall@K.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# UTF-8 console on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.config import BM25_INDEX_PATH, get_benchmark_path  # noqa: E402
from akn_rlm.corpus.akn_parser import parse_all  # noqa: E402
from akn_rlm.corpus.article_registry import ArticleRegistry  # noqa: E402
from akn_rlm.indexers.bm25 import BM25Index  # noqa: E402
from akn_rlm.rlm.routing.doc_router import DEFAULT_TOP_N, build_doc_router  # noqa: E402

from scripts.run_benchmark import (  # noqa: E402
    _benchmark_to_records,
    _stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_doc_router")


def _eval(
    router,
    records: list[dict],
    *,
    top_n: int,
    output_path: Path,
) -> dict:
    log.info("Evaluating %d questions (top_n=%d) …", len(records), top_n)

    by_type_total: dict[str, int] = defaultdict(int)
    by_type_hit: dict[str, int] = defaultdict(int)
    by_type_alias_hit: dict[str, int] = defaultdict(int)
    confusion: list[dict] = []
    latencies: list[float] = []

    overall_total = 0
    overall_hit = 0
    overall_alias_hit = 0

    for q in records:
        gold = set(q.get("gold_doc_ids", []))
        if not gold:
            continue
        qt = q.get("query_type", "rule_application")
        t0 = time.time()
        result = router.route(q["query"], top_n=top_n)
        latencies.append(time.time() - t0)
        pred = set(result.doc_ids)
        hit = bool(gold & pred)
        alias_hit = any(
            "alias" in result.sources.get(d, []) for d in (gold & pred)
        )

        overall_total += 1
        by_type_total[qt] += 1
        if hit:
            overall_hit += 1
            by_type_hit[qt] += 1
            if alias_hit:
                overall_alias_hit += 1
                by_type_alias_hit[qt] += 1
        else:
            confusion.append({
                "id":       q.get("id", ""),
                "type":     qt,
                "query":    q["query"][:140],
                "gold":     sorted(gold),
                "pred":     result.doc_ids,
                "scores":   {d: round(s, 3) for d, s in result.scores.items()},
            })

    def _pct(num: int, denom: int) -> float:
        return 100.0 * num / denom if denom else 0.0

    summary = {
        "overall": {
            "n":                   overall_total,
            "recall_at_top_n":     _pct(overall_hit, overall_total) / 100.0,
            "alias_recall_at_top_n": _pct(overall_alias_hit, overall_total) / 100.0,
            "mean_latency_s":      sum(latencies) / len(latencies) if latencies else 0.0,
        },
        "by_query_type": {
            qt: {
                "n":       by_type_total[qt],
                "hit":     by_type_hit[qt],
                "alias":   by_type_alias_hit[qt],
                "recall":  _pct(by_type_hit[qt], by_type_total[qt]) / 100.0,
            }
            for qt in sorted(by_type_total)
        },
        "top_n": top_n,
        "misses": confusion,
    }

    # Markdown
    lines: list[str] = []
    lines.append(f"# Doc-router smoke (top_n={top_n})")
    lines.append("")
    lines.append(
        f"**Overall:** {overall_hit}/{overall_total} hits  "
        f"= **{summary['overall']['recall_at_top_n']*100:.1f}%** recall@{top_n}  "
        f"(alias-only-hit subset: {overall_alias_hit}/{overall_total} = "
        f"{summary['overall']['alias_recall_at_top_n']*100:.1f}%)  "
        f"Latency: {summary['overall']['mean_latency_s']*1000:.1f} ms/q"
    )
    lines.append("")
    lines.append("## Per query type")
    lines.append("")
    lines.append(f"| Query type | n | hit | recall@{top_n} | alias |")
    lines.append("|---|---:|---:|---:|---:|")
    for qt, s in summary["by_query_type"].items():
        lines.append(
            f"| {qt} | {s['n']} | {s['hit']} | "
            f"{s['recall']*100:.1f}% | {s['alias']} |"
        )
    lines.append("")
    if confusion:
        lines.append(f"## Misses ({len(confusion)})")
        lines.append("")
        for m in confusion[:30]:
            lines.append(
                f"- **{m['id']}** ({m['type']}) — gold={m['gold']} "
                f"pred={m['pred']} — `{m['query']}`"
            )
        if len(confusion) > 30:
            lines.append(f"- … and {len(confusion) - 30} more")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report saved -> %s", output_path)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    log.info("JSON summary saved   -> %s", json_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-evaluate the doc_router against AlgerianLegalBench"
    )
    parser.add_argument("--benchmark", default=None,
                        help="Override benchmark path (default: auto-detect)")
    parser.add_argument("--stratified", type=int, default=None,
                        help="Pick N questions per query_type")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N questions")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Router top-N (default {DEFAULT_TOP_N})")
    parser.add_argument("--out", default="eval_results/doc_router_smoke.md",
                        help="Output markdown path")
    parser.add_argument("--e7", action="store_true",
                        help="Phase E.4: enable the KG-derived doc-router "
                             "channel. Loads the KG (~26s) on startup and "
                             "adds a 0.5 bonus per matched doc.")
    args = parser.parse_args()

    bm_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("Doc-router smoke")
    log.info("Benchmark : %s", bm_path)
    log.info("Top-N     : %d", args.top_n)
    log.info("=" * 64)

    log.info("Parsing corpus to build registry …")
    registry = ArticleRegistry()
    registry.build(parse_all())

    if BM25_INDEX_PATH.exists():
        log.info("Loading BM25 index …")
        bm25 = BM25Index.load(BM25_INDEX_PATH)
    else:
        log.warning("BM25 index not found — alias-only routing")
        bm25 = None

    router_kwargs: dict = {}
    if args.e7:
        log.info("E7 enabled — loading KG and building KG doc-router call …")
        from akn_rlm.corpus.kg_loader import load_kg  # noqa: PLC0415
        from akn_rlm.rlm.enhancers import make_kg_doc_router_call  # noqa: PLC0415

        kg = load_kg()

        def _sparql_fn(q: str):
            try:
                res = kg.query(q)
            except Exception:
                return []
            if q.strip().lower().startswith("ask"):
                try:
                    return [{"_ask": bool(res)}]
                except Exception:
                    return []
            rows = []
            try:
                vars_list = list(res.vars or [])
            except Exception:
                vars_list = []
            for row in res:
                rows.append({
                    str(v): (str(row[v]) if row[v] is not None else None)
                    for v in vars_list
                })
            return rows

        router_kwargs["kg_call"] = make_kg_doc_router_call(_sparql_fn)

    router = build_doc_router(
        registry=registry, bm25=bm25, top_n=args.top_n, **router_kwargs,
    )

    records = _benchmark_to_records(bm_path, registry)
    if args.stratified:
        records = _stratified_sample(records, args.stratified)
        log.info("Stratified sample: %d records", len(records))
    if args.limit:
        records = records[: args.limit]

    summary = _eval(router, records, top_n=args.top_n, output_path=Path(args.out))
    print()
    print(f"OVERALL recall@{args.top_n}: "
          f"{summary['overall']['recall_at_top_n']*100:.1f}%  "
          f"(n={summary['overall']['n']})")
    for qt, s in summary["by_query_type"].items():
        print(f"  {qt:>26s}: {s['recall']*100:6.1f}%  ({s['hit']}/{s['n']})")


if __name__ == "__main__":
    main()
