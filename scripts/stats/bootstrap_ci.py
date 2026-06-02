"""Bootstrap confidence intervals for every run × metric × stratum.

Usage:
    python -m scripts.stats.bootstrap_ci           # all 244-record runs
    python -m scripts.stats.bootstrap_ci <run_id>  # single run

Writes results/stats/ci_<run_id>.json for each run.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
EVAL_DIR   = ROOT / "eval_results"
OUT_DIR    = ROOT / "results" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.stats.metrics import per_question_scores, abstention_f1

RNG_SEED    = 42
N_BOOT      = 1000
STRAT_FIELDS = ["query_type", "difficulty", "language"]


def _load(run_id: str) -> list[dict]:
    path = EVAL_DIR / run_id / "predictions.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Bootstrap mean + 95 % percentile CI from a 1-D array of per-question scores."""
    n = len(values)
    means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(N_BOOT)
    ])
    return {
        "mean":  float(values.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n":     n,
    }


def _abst_f1_boot(records: list[dict], rng: np.random.Generator) -> dict:
    """Bootstrap CI for the batch abstention F1 by resampling whole records."""
    n = len(records)
    idx = np.arange(n)
    boot_vals = []
    for _ in range(N_BOOT):
        sample = [records[i] for i in rng.choice(idx, size=n, replace=True)]
        boot_vals.append(abstention_f1(sample))
    arr = np.array(boot_vals)
    return {
        "mean":  float(abstention_f1(records)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n":     n,
    }


def compute_ci(run_id: str) -> dict:
    records  = _load(run_id)
    rng      = np.random.default_rng(RNG_SEED)
    scores   = [per_question_scores(r) for r in records]
    SCALAR_METRICS = ["citation_f1", "hcr", "jir", "mrr", "ndcg"]

    result = {"run_id": run_id, "strata": {}}

    def _process_group(label: str, group_recs: list[dict], group_scores: list[dict]):
        out = {}
        for m in SCALAR_METRICS:
            arr = np.array([s[m] for s in group_scores])
            out[m] = _boot_ci(arr, rng)
        out["abstention_f1"] = _abst_f1_boot(group_recs, rng)
        result["strata"][label] = out

    # overall
    _process_group("overall", records, scores)

    # per stratum field
    for field in STRAT_FIELDS:
        vals = sorted({r.get(field, "unknown") for r in records})
        for v in vals:
            grp_recs   = [r for r in records if r.get(field) == v]
            grp_scores = [per_question_scores(r) for r in grp_recs]
            _process_group(f"{field}={v}", grp_recs, grp_scores)

    return result


def main():
    target_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    if target_ids is None:
        # all runs that have exactly 244 records
        target_ids = []
        for d in sorted(EVAL_DIR.iterdir()):
            fp = d / "predictions.jsonl"
            if not fp.exists():
                continue
            lines = [l for l in fp.read_text().splitlines() if l.strip()]
            if len(lines) == 244:
                target_ids.append(d.name)
        print(f"Found {len(target_ids)} full runs: {target_ids}")

    for run_id in target_ids:
        print(f"  bootstrapping {run_id} …", end=" ", flush=True)
        ci = compute_ci(run_id)
        out_path = OUT_DIR / f"ci_{run_id}.json"
        out_path.write_text(json.dumps(ci, indent=2, ensure_ascii=False))
        f1 = ci["strata"]["overall"]["citation_f1"]["mean"]
        print(f"Cite F1={f1:.4f}  → {out_path.name}")


if __name__ == "__main__":
    main()
