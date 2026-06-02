"""Paired bootstrap p-values + effect sizes for AKN-RLM vs every baseline.

NOTE: The canonical reference runs (rlm_dispatched_full_phase_e_final and
rlm_dispatched_full_e4_trajectory) were not synced to this machine. The best
locally available AKN-RLM run is rlm_dispatched_full_adu_pervasive
(Phase C SOTA, Cite F1 = 0.3010). That run is used as AKN_RUN below.

Outputs
-------
results/stats/paired_pvalues.csv
"""
from __future__ import annotations
import json, csv, math, sys
from pathlib import Path
import numpy as np

ROOT     = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval_results"
OUT_DIR  = ROOT / "results" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from scripts.stats.metrics import per_question_scores, abstention_f1

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AKN_RUN = "rlm_dispatched_full_adu_pervasive"  # best locally available

BASELINE_RUNS = [
    "baseline_bm25_full",
    "baseline_dense_full",
    "baseline_hybrid_full",
    "baseline_hybrid_rerank_full",
    "baseline_kg_full",
    "baseline_kg_hybrid_full",
]

SCALAR_METRICS  = ["citation_f1", "hcr", "jir", "mrr", "ndcg"]
BINARY_METRICS  = {"hcr", "jir"}   # absolute risk diff + risk ratio
BATCH_METRICS   = {"abstention_f1"}
ALL_METRICS     = SCALAR_METRICS + ["abstention_f1"]

RNG_SEED = 42
N_BOOT   = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(run_id: str) -> list[dict]:
    path = EVAL_DIR / run_id / "predictions.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _cohens_d(deltas: np.ndarray) -> float:
    """Cohen's d on paired deltas (mean / std)."""
    s = deltas.std(ddof=1)
    return float(deltas.mean() / s) if s > 0 else 0.0


def _paired_boot_scalar(
    akn_arr: np.ndarray,
    base_arr: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Paired bootstrap for a per-question scalar metric."""
    deltas = akn_arr - base_arr
    n = len(deltas)
    boot_means = np.array([
        rng.choice(deltas, size=n, replace=True).mean()
        for _ in range(N_BOOT)
    ])
    delta_mean = float(deltas.mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    frac_le_0 = float((boot_means <= 0).mean())
    frac_ge_0 = float((boot_means >= 0).mean())
    p_raw = float(2 * min(frac_le_0, frac_ge_0))
    p_raw = max(p_raw, 1.0 / N_BOOT)  # floor at 1/B

    # Effect size
    if deltas.max() == deltas.min():
        effect = {"type": "zero", "value": 0.0}
    elif akn_arr.max() <= 1.0 and set(akn_arr.tolist()) <= {0.0, 1.0}:
        # binary
        akn_mean  = float(akn_arr.mean())
        base_mean = float(base_arr.mean())
        rr = akn_mean / base_mean if base_mean > 0 else float("inf")
        effect = {"type": "ard_rr", "ard": delta_mean, "rr": rr}
    else:
        effect = {"type": "cohens_d", "value": _cohens_d(deltas)}

    return {
        "delta_mean": delta_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_raw": p_raw,
        "effect": effect,
    }


def _paired_boot_abst_f1(
    akn_recs: list[dict],
    base_recs: list[dict],
    rng: np.random.Generator,
) -> dict:
    """Paired bootstrap for batch abstention F1 via record resampling."""
    n = len(akn_recs)
    idx = np.arange(n)
    boot_deltas = []
    for _ in range(N_BOOT):
        ii = rng.choice(idx, size=n, replace=True)
        d = abstention_f1([akn_recs[i] for i in ii]) - abstention_f1([base_recs[i] for i in ii])
        boot_deltas.append(d)
    arr = np.array(boot_deltas)
    delta_mean = float(abstention_f1(akn_recs) - abstention_f1(base_recs))
    ci_lo = float(np.percentile(arr, 2.5))
    ci_hi = float(np.percentile(arr, 97.5))
    frac_le_0 = float((arr <= 0).mean())
    frac_ge_0 = float((arr >= 0).mean())
    p_raw = float(2 * min(frac_le_0, frac_ge_0))
    p_raw = max(p_raw, 1.0 / N_BOOT)
    return {
        "delta_mean": delta_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_raw": p_raw,
        "effect": {"type": "cohens_d", "value": _cohens_d(arr)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng      = np.random.default_rng(RNG_SEED)
    akn_recs = _load(AKN_RUN)
    akn_sc   = [per_question_scores(r) for r in akn_recs]

    rows = []  # will become CSV
    raw_results = {}  # for JSON CI file

    n_comparisons = len(BASELINE_RUNS) * len(ALL_METRICS)
    print(f"AKN run : {AKN_RUN}  (n={len(akn_recs)})")
    print(f"Baselines: {len(BASELINE_RUNS)}  Metrics: {len(ALL_METRICS)}  "
          f"Bonferroni n={n_comparisons}")

    for base_id in BASELINE_RUNS:
        print(f"  pairing vs {base_id} …", flush=True)
        base_recs = _load(base_id)
        # align on question_id order
        akn_map  = {r["question_id"]: (r, s) for r, s in zip(akn_recs, akn_sc)}
        base_map = {r["question_id"]: r for r in base_recs}
        common   = sorted(set(akn_map) & set(base_map))
        if len(common) < len(akn_recs):
            print(f"    WARNING: only {len(common)}/{len(akn_recs)} questions align")
        akn_aligned  = [akn_map[q][0] for q in common]
        akn_sc_align = [akn_map[q][1] for q in common]
        base_aligned = [base_map[q] for q in common]
        base_sc_al   = [per_question_scores(r) for r in base_aligned]

        raw_results[base_id] = {}

        for metric in SCALAR_METRICS:
            akn_arr  = np.array([s[metric] for s in akn_sc_align])
            base_arr = np.array([s[metric] for s in base_sc_al])
            res = _paired_boot_scalar(akn_arr, base_arr, rng)
            raw_results[base_id][metric] = res

            eff = res["effect"]
            if eff["type"] == "cohens_d":
                eff_str = f"d={eff['value']:.3f}"
            elif eff["type"] == "ard_rr":
                eff_str = f"ARD={eff['ard']:.3f};RR={eff['rr']:.3f}"
            else:
                eff_str = "0"

            rows.append({
                "baseline":    base_id,
                "metric":      metric,
                "akn_mean":    round(akn_arr.mean(), 5),
                "base_mean":   round(base_arr.mean(), 5),
                "delta_mean":  round(res["delta_mean"], 5),
                "ci_lo":       round(res["ci_lo"], 5),
                "ci_hi":       round(res["ci_hi"], 5),
                "p_raw":       round(res["p_raw"], 6),
                "p_bonferroni": round(min(1.0, res["p_raw"] * n_comparisons), 6),
                "sig_level":   _sig_stars(min(1.0, res["p_raw"] * n_comparisons)),
                "effect":      eff_str,
            })

        # abstention_f1
        res = _paired_boot_abst_f1(akn_aligned, base_aligned, rng)
        raw_results[base_id]["abstention_f1"] = res
        eff = res["effect"]
        eff_str = f"d={eff['value']:.3f}" if eff["type"] == "cohens_d" else "0"
        rows.append({
            "baseline":    base_id,
            "metric":      "abstention_f1",
            "akn_mean":    round(abstention_f1(akn_aligned), 5),
            "base_mean":   round(abstention_f1(base_aligned), 5),
            "delta_mean":  round(res["delta_mean"], 5),
            "ci_lo":       round(res["ci_lo"], 5),
            "ci_hi":       round(res["ci_hi"], 5),
            "p_raw":       round(res["p_raw"], 6),
            "p_bonferroni": round(min(1.0, res["p_raw"] * n_comparisons), 6),
            "sig_level":   _sig_stars(min(1.0, res["p_raw"] * n_comparisons)),
            "effect":      eff_str,
        })

    # write CSV
    csv_path = OUT_DIR / "paired_pvalues.csv"
    fieldnames = ["baseline","metric","akn_mean","base_mean","delta_mean",
                  "ci_lo","ci_hi","p_raw","p_bonferroni","sig_level","effect"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # also write raw JSON for LaTeX step
    json_path = OUT_DIR / "paired_pvalues_raw.json"
    json_path.write_text(json.dumps(raw_results, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
