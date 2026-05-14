"""Re-compute am_faithfulness_score and full strata from saved predictions.

Used by Phase C after the metric semantics changed (1.0 → 0.0 for the
"no grounds available" fallback). Walks every per-question result row in
predictions.jsonl, recomputes the metric, and writes a fresh
metrics.json + metrics.md without re-running the 25-min benchmark.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.eval.metrics import am_faithfulness_score  # noqa: E402
from akn_rlm.eval.runner import _format_markdown  # noqa: E402
from akn_rlm.eval.stratified import stratify  # noqa: E402


def reaggregate(run_dir: Path) -> dict:
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open(encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]

    for r in rows:
        if r.get("predicted_abstain"):
            am = 1.0
        else:
            am = am_faithfulness_score(
                r.get("answer_text", "") or "",
                r.get("predicted_citations") or [],
            )
        r["am_faithfulness_score"] = am

    strata = stratify(rows)

    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError

    (run_dir / "metrics.json").write_text(
        json.dumps(strata, ensure_ascii=False, indent=2, default=_default),
        encoding="utf-8",
    )
    (run_dir / "metrics.md").write_text(_format_markdown(strata), encoding="utf-8")
    return strata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run-id directories under eval_results/")
    ap.add_argument("--root", default="eval_results")
    args = ap.parse_args()

    for run in args.runs:
        d = Path(args.root) / run
        if not d.exists():
            print(f"  ! skip (missing): {d}")
            continue
        s = reaggregate(d)
        ov = s.get("overall", {})
        print(f"=== {run} ===")
        for k in ("citation_f1", "mrr_doc", "mrr_article", "answer_faithfulness",
                  "am_faithfulness_score", "citation_groundedness",
                  "hcr", "abstention_f1", "mean_latency_s"):
            print(f"  {k:30}  {ov.get(k, 0.0):.4f}")
        print()


if __name__ == "__main__":
    main()
