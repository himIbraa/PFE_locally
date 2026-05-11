"""E1-E4 ablation runner — strat5 (40 q) per config, then comparison.

Six runs in order:
    1. F5 baseline (no enhancers)            -> rlm_dispatched_strat5_e0
    2. F5 + E1 (concept->amendment)          -> rlm_dispatched_strat5_e1
    3. F5 + E2 (reverse NLI verifier)        -> rlm_dispatched_strat5_e2
    4. F5 + E3 (Gemma paraphrase retrieval)  -> rlm_dispatched_strat5_e3
    5. F5 + E4 (HyDE)                        -> rlm_dispatched_strat5_e4
    6. F5 + E1+E2+E3+E4 (all)                -> rlm_dispatched_strat5_eall

After each run, scrape metrics.json. At the end, print + write a
side-by-side comparison table + identify the strongest config per
metric.

Each strat5 run is ~3-5 min locally. Full ablation ≈ 20-30 min.

Usage:
    python scripts/run_ablation_e1_e4.py
    python scripts/run_ablation_e1_e4.py --skip e2 --skip e4   # skip configs
    python scripts/run_ablation_e1_e4.py --only e1 eall        # run only these
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval_results"
THESIS_DIR = REPO_ROOT.parent / "thesis_comparisons"

# Each config: (run_id, label, extra_cli_flags)
CONFIGS = [
    ("rlm_dispatched_strat5_e0",   "F5 baseline (no enhancers)",      []),
    ("rlm_dispatched_strat5_e1",   "F5 + E1 (concept_amendment)",     ["--e1"]),
    ("rlm_dispatched_strat5_e2",   "F5 + E2 (NLI reverse)",           ["--e2"]),
    ("rlm_dispatched_strat5_e3",   "F5 + E3 (paraphrase)",            ["--e3"]),
    ("rlm_dispatched_strat5_e4",   "F5 + E4 (HyDE)",                  ["--e4"]),
    ("rlm_dispatched_strat5_eall", "F5 + E1+E2+E3+E4 (all)",          ["--enhancers-all"]),
]

# Locked F5 thesis numbers (Windows full-244; for context column)
F5_FULL244 = {
    "citation_f1":          0.3011,
    "mrr_at_10_article":    0.2686,
    "mrr_at_10_doc":        0.5567,
    "doc_citation_f1":      0.6122,
    "recall_at_10_article": 0.2155,
    "hcr":                  0.0000,
    "abstention_f1":        0.7075,
    "mean_latency_s":       4.5086,
}

KEYS = [
    "citation_f1", "mrr_at_10_article", "mrr_at_10_doc",
    "doc_citation_f1", "recall_at_10_article", "hcr",
    "abstention_f1", "mean_latency_s",
]

DIR_BETTER = {
    "citation_f1":          "high",
    "mrr_at_10_article":    "high",
    "mrr_at_10_doc":        "high",
    "doc_citation_f1":      "high",
    "recall_at_10_article": "high",
    "hcr":                  "low",
    "abstention_f1":        "high",
    "mean_latency_s":       "low",
}


def _python() -> str:
    """Use the same interpreter the user invoked us with."""
    return sys.executable


def _run_dispatcher(run_id: str, extra: list[str]) -> dict | None:
    """Call run_dispatcher.py with --stratified 5 + extra flags. Return metrics."""
    script = REPO_ROOT / "scripts" / "run_dispatcher.py"
    cmd = [_python(), str(script), "--stratified", "5", "--run-id", run_id] + extra
    log_path = EVAL_DIR / f"_ablation_{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[ablation] $ {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "wb") as fout:
        ret = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[ablation] {run_id} done in {dt:.0f}s (exit={ret.returncode}) — log: {log_path}")
    metrics_path = EVAL_DIR / run_id / "metrics.json"
    if not metrics_path.exists():
        print(f"[ablation] WARNING — metrics.json missing at {metrics_path}")
        return None
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f).get("overall", {})


def _better(a: float | None, b: float | None, direction: str) -> bool:
    if a is None or b is None:
        return False
    if direction == "high":
        return a > b
    return a < b


def _format_row(label: str, m: dict | None) -> str:
    cells = []
    for k in KEYS:
        v = m.get(k) if isinstance(m, dict) else None
        cells.append(f"{v:>14.4f}" if isinstance(v, (int, float)) else f"{'(none)':>14s}")
    return f"{label:38s}" + "".join(cells)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", action="append", default=[],
                   help="config keys to skip (e0, e1, e2, e3, e4, eall)")
    p.add_argument("--only", nargs="+", default=None,
                   help="run only these configs (e0 e1 ...)")
    args = p.parse_args()

    skip = {s.lower() for s in args.skip}
    only = {s.lower() for s in args.only} if args.only else None

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict | None] = {}
    for run_id, label, extra in CONFIGS:
        key = run_id.split("_")[-1]   # e0/e1/e2/e3/e4/eall
        if key in skip or (only is not None and key not in only):
            print(f"[ablation] skipping {run_id} ({key})")
            results[run_id] = None
            continue
        results[run_id] = _run_dispatcher(run_id, extra)

    # ------ comparison table ----------------------------------------------
    header = f"{'config':38s}" + "".join(f"{k:>14s}" for k in KEYS)
    lines = ["", "=" * len(header), header, "-" * len(header)]
    lines.append(_format_row("F5 full-244 (locked Windows)", F5_FULL244))
    for run_id, label, _ in CONFIGS:
        lines.append(_format_row(label, results.get(run_id)))
    lines.append("=" * len(header))

    # ------ winners --------------------------------------------------------
    lines.append("")
    lines.append("STRONGEST per metric (excluding F5 full-244 reference):")
    for k in KEYS:
        best_label, best_val = None, None
        for run_id, label, _ in CONFIGS:
            m = results.get(run_id)
            if m is None: continue
            v = m.get(k)
            if v is None: continue
            if best_val is None or _better(v, best_val, DIR_BETTER[k]):
                best_val, best_label = v, label
        if best_label is None:
            lines.append(f"  {k:22s}: (no data)")
        else:
            lines.append(f"  {k:22s}: {best_val:.4f}  <- {best_label}")

    # ------ pareto winner (Cite F1 + MRR art + R@10 art) -------------------
    lines.append("")
    lines.append("PARETO winner (rank-sum of Cite F1, MRR art, R@10 art): lower = better")
    pareto_keys = ["citation_f1", "mrr_at_10_article", "recall_at_10_article"]
    scores: dict[str, float] = {}
    for run_id, label, _ in CONFIGS:
        m = results.get(run_id)
        if m is None: continue
        rank_sum = 0
        for k in pareto_keys:
            v = m.get(k, -1)
            others = sorted(
                [results[rid].get(k, -1) for rid, _, _ in CONFIGS if results.get(rid)],
                reverse=True,
            )
            try:
                rank_sum += others.index(v) + 1
            except ValueError:
                rank_sum += 99
        scores[label] = rank_sum
    for label, s in sorted(scores.items(), key=lambda kv: kv[1]):
        lines.append(f"  rank-sum {s:>3d}  {label}")

    out = "\n".join(lines)
    print(out)

    # ------ persist artifact -----------------------------------------------
    stamp = time.strftime("%Y%m%d_%H%M%S")
    txt_path = THESIS_DIR / f"ablation_e1_e4_{stamp}.txt"
    md_path  = THESIS_DIR / f"ablation_e1_e4_{stamp}.md"
    json_path = THESIS_DIR / f"ablation_e1_e4_{stamp}.json"
    txt_path.write_text(out, encoding="utf-8")
    json_path.write_text(json.dumps({
        "configs": [{"run_id": rid, "label": lbl, "metrics": results.get(rid)}
                    for rid, lbl, _ in CONFIGS],
        "f5_full244_reference": F5_FULL244,
    }, indent=2), encoding="utf-8")

    # markdown table for thesis
    md = ["# E1-E4 ablation — strat5 (40 q)",
          "",
          "Generated " + stamp,
          "",
          "| Config | " + " | ".join(KEYS) + " |",
          "|---|" + "|".join(["---:" for _ in KEYS]) + "|",
          "| F5 full-244 (locked Windows) | " +
              " | ".join(f"{F5_FULL244[k]:.4f}" for k in KEYS) + " |"]
    for run_id, label, _ in CONFIGS:
        m = results.get(run_id)
        if m is None:
            cells = ["—"] * len(KEYS)
        else:
            cells = [f"{m.get(k, 0):.4f}" for k in KEYS]
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n[ablation] artifacts:\n  {txt_path}\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()
