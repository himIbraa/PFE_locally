"""Generate LaTeX table fragments and SUMMARY.md from CI + paired-bootstrap results.

Outputs:
  results/stats/table_tier1_with_pvalues.tex
  results/stats/table_tier2_with_pvalues.tex
  results/stats/SUMMARY.md
"""
from __future__ import annotations
import json, csv, sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "stats"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_ci(run_id: str) -> dict:
    p = OUT_DIR / f"ci_{run_id}.json"
    return json.loads(p.read_text())


def _load_csv() -> list[dict]:
    p = OUT_DIR / "paired_pvalues.csv"
    return list(csv.DictReader(open(p)))


AKN_RUN   = "rlm_dispatched_full_adu_pervasive"
BEST_PHASE1 = "baseline_hybrid_rerank_full"

DISPLAY_NAMES = {
    "baseline_bm25_full":           "BM25",
    "baseline_dense_full":          "Dense",
    "baseline_hybrid_full":         "Hybrid (RRF)",
    "baseline_hybrid_rerank_full":  "Hybrid+Rerank",
    "baseline_kg_full":             "KG (SPARQL)",
    "baseline_kg_hybrid_full":      "KG+Hybrid",
    AKN_RUN:                        r"\textbf{AKN-RLM}",
}

BASELINE_ORDER = [
    "baseline_bm25_full",
    "baseline_dense_full",
    "baseline_hybrid_full",
    "baseline_hybrid_rerank_full",
    "baseline_kg_full",
    "baseline_kg_hybrid_full",
]

QUERY_TYPES_ORDERED = [
    "exact_article", "rule_application", "multi_hop",
    "temporal_factual", "conceptual_definitional",
    "unanswerable", "layman", "long_context",
]
HARD_TYPES = {"multi_hop", "temporal_factual", "conceptual_definitional", "unanswerable"}

QT_DISPLAY = {
    "exact_article":            r"\textit{exact\_article}",
    "rule_application":         r"\textit{rule\_application}",
    "multi_hop":                r"\textit{multi\_hop}$^\star$",
    "temporal_factual":         r"\textit{temporal\_factual}$^\star$",
    "conceptual_definitional":  r"\textit{conceptual\_definitional}$^\star$",
    "unanswerable":             r"\textit{unanswerable}$^\star$",
    "layman":                   r"\textit{layman}",
    "long_context":             r"\textit{long\_context}",
}


def _ci_str(ci: dict, fmt: str = ".3f") -> str:
    return f"{ci['mean']:{fmt}} ({ci['ci_lo']:{fmt}}, {ci['ci_hi']:{fmt}})"


def _ci_val(ci: dict) -> str:
    return f"{ci['mean']:.3f}"


def _pval_row(csv_rows: list[dict], baseline: str, metric: str) -> dict | None:
    for r in csv_rows:
        if r["baseline"] == baseline and r["metric"] == metric:
            return r
    return None


# ---------------------------------------------------------------------------
# Table 4.2 — Tier 1: overall metrics with CIs and Δ-vs-AKN significance
# ---------------------------------------------------------------------------

def build_tier1(csv_rows: list[dict]) -> str:
    # Columns: Pipeline | Cite F1 (CI) | MRR@10 | nDCG@10 | HCR↓ | AbstF1 | Δ Cite F1 vs AKN
    akn_ci  = _load_ci(AKN_RUN)["strata"]["overall"]
    lines = []
    lines.append(r"% Table 4.2 — Overall retrieval + faithfulness metrics (full 244-q)")
    lines.append(r"% AKN-RLM run: rlm_dispatched_full_adu_pervasive (Phase C SOTA, locally available)")
    lines.append(r"% 95\% bootstrap CIs in parentheses. $\Delta$ = AKN minus baseline.")
    lines.append(r"% Significance after Bonferroni correction (36 comparisons):")
    lines.append(r"% $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$, ns = not significant.")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall evaluation metrics on AlgerianLegalBench v3.0 (244 questions). "
                 r"Cite F1, MRR@10, nDCG@10: higher is better. HCR: lower is better. "
                 r"95\,\% bootstrap CIs in parentheses. $\Delta$: AKN-RLM minus baseline, "
                 r"significance after Bonferroni correction.}")
    lines.append(r"\label{tab:tier1_metrics}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Pipeline & Cite F1 (95\,\%CI) & MRR@10 (95\,\%CI) & "
                 r"nDCG@10 (95\,\%CI) & HCR$\downarrow$ (95\,\%CI) & AbstF1 (95\,\%CI) & "
                 r"$\Delta$ Cite F1 \\")
    lines.append(r"\midrule")

    # baselines
    for run_id in BASELINE_ORDER:
        ci  = _load_ci(run_id)["strata"]["overall"]
        row = _pval_row(csv_rows, run_id, "citation_f1")
        delta_str = ""
        if row:
            delta = float(row["delta_mean"])
            stars = row["sig_level"] if row["sig_level"] != "ns" else r"\text{ns}"
            if row["sig_level"] == "ns":
                delta_str = rf"${delta:+.3f}$ (ns)"
            else:
                delta_str = rf"${delta:+.3f}^{{\text{{{row['sig_level']}}}}}$"
        name = DISPLAY_NAMES[run_id]
        lines.append(
            rf"{name} & {_ci_str(ci['citation_f1'])} & {_ci_str(ci['mrr'])} & "
            rf"{_ci_str(ci['ndcg'])} & {_ci_str(ci['hcr'])} & "
            rf"{_ci_str(ci['abstention_f1'])} & {delta_str} \\"
        )

    lines.append(r"\midrule")

    # AKN row (no delta vs itself)
    ci = akn_ci
    name = DISPLAY_NAMES[AKN_RUN]
    lines.append(
        rf"{name} & \textbf{{{_ci_str(ci['citation_f1'])}}} & {_ci_str(ci['mrr'])} & "
        rf"{_ci_str(ci['ndcg'])} & {_ci_str(ci['hcr'])} & "
        rf"{_ci_str(ci['abstention_f1'])} & --- \\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4.3 — Tier 2: per-query-type Cite F1 with CIs
# ---------------------------------------------------------------------------

def build_tier2(csv_rows: list[dict]) -> str:
    # Load CI for each run per query_type stratum
    runs_in_table = BASELINE_ORDER + [AKN_RUN]
    ci_data = {r: _load_ci(r) for r in runs_in_table}

    lines = []
    lines.append(r"% Table 4.3 — Cite F1 by query type (full 244-q)")
    lines.append(r"% 95\% bootstrap CIs in parentheses. $\star$ = hard query type.")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Article-level Citation F1 by query type. "
                 r"95\,\% bootstrap CIs in parentheses. "
                 r"$^\star$ marks hard query types (multi-hop, temporal, conceptual, unanswerable). "
                 r"$n$ = number of questions per type.}")
    lines.append(r"\label{tab:tier2_by_querytype}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    n_cols = len(runs_in_table)
    col_spec = "l" + "c" * n_cols
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # header
    header_names = [DISPLAY_NAMES.get(r, r) for r in runs_in_table]
    lines.append(r"Query type ($n$) & " + " & ".join(header_names) + r" \\")
    lines.append(r"\midrule")

    # get n per query type from AKN run
    akn_records_count = {}
    for qt in QUERY_TYPES_ORDERED:
        strat_key = f"query_type={qt}"
        strat = ci_data[AKN_RUN]["strata"].get(strat_key, {})
        n = strat.get("citation_f1", {}).get("n", 0)
        akn_records_count[qt] = n

    for qt in QUERY_TYPES_ORDERED:
        n = akn_records_count[qt]
        row_label = QT_DISPLAY[qt] + rf" ($n={n}$)"
        cells = []
        for run_id in runs_in_table:
            strat_key = f"query_type={qt}"
            strat = ci_data[run_id]["strata"].get(strat_key, {})
            ci = strat.get("citation_f1")
            if ci:
                cell = _ci_str(ci)
                if run_id == AKN_RUN:
                    cell = rf"\textbf{{{cell}}}"
            else:
                cell = "---"
            cells.append(cell)
        lines.append(row_label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # overall row
    row_label = r"\textbf{Overall} ($n=244$)"
    cells = []
    for run_id in runs_in_table:
        ci = ci_data[run_id]["strata"]["overall"]["citation_f1"]
        cell = _ci_str(ci)
        if run_id == AKN_RUN:
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    lines.append(row_label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------

def build_summary(csv_rows: list[dict]) -> str:
    akn_ci   = _load_ci(AKN_RUN)["strata"]["overall"]
    best_ci  = _load_ci(BEST_PHASE1)["strata"]["overall"]

    def _row(metric: str, baseline: str = BEST_PHASE1) -> dict | None:
        return _pval_row(csv_rows, baseline, metric)

    def _sig_sentence(row: dict | None, metric_label: str) -> str:
        if row is None:
            return "No comparison available."
        p = float(row["p_bonferroni"])
        sig = row["sig_level"]
        delta = float(row["delta_mean"])
        eff = row["effect"]
        if sig == "ns":
            return (
                f"The difference (Δ = {delta:+.4f}) is **not significant after "
                f"Bonferroni correction** (p_Bonf = {p:.4f}); the benchmark "
                f"(n = 244) is likely underpowered to detect this effect."
            )
        return (
            f"The difference (Δ = {delta:+.4f}) is **significant** "
            f"(p_Bonf = {p:.4f}, {sig}) with effect {eff}."
        )

    lines = [
        "# Statistical Summary — Chapter 4 Evaluation (bootstrap inference)",
        "",
        f"> **AKN-RLM run used**: `{AKN_RUN}` (Phase C SOTA, Cite F1 = "
        f"{akn_ci['citation_f1']['mean']:.4f})",
        f"> **Note**: The canonical reference runs `rlm_dispatched_full_phase_e_final`"
        f" and `rlm_dispatched_full_e4_trajectory` were not present on this machine.",
        f"> The Phase C run is the next-best available (Cite F1 = 0.3010 vs 0.3129 for the true SOTA).",
        "",
        "All CIs are 95 % percentile bootstrap intervals (1 000 resamples, seed 42).",
        "Paired p-values use 10 000 resamples. Significance thresholds after Bonferroni",
        f"correction across {len(BASELINE_ORDER)} baselines × {len(['citation_f1','hcr','jir','mrr','ndcg','abstention_f1'])} metrics = 36 comparisons:",
        "`***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `ns` not significant.",
        "",
    ]

    # --- Citation F1 ---
    cf_akn  = akn_ci["citation_f1"]
    cf_best = best_ci["citation_f1"]
    row_cf  = _row("citation_f1")
    lines += [
        "## Citation F1 (article-level)",
        "",
        f"The AKN-RLM system achieves a mean Citation F1 of **{cf_akn['mean']:.4f}** "
        f"(95 % CI: {cf_akn['ci_lo']:.4f}–{cf_akn['ci_hi']:.4f}) on the full 244-question benchmark. "
        f"The strongest Phase-1 baseline (Hybrid+Rerank) reaches {cf_best['mean']:.4f} "
        f"(95 % CI: {cf_best['ci_lo']:.4f}–{cf_best['ci_hi']:.4f}). "
        + _sig_sentence(row_cf, "Citation F1"),
        "",
    ]

    # --- HCR ---
    hcr_akn  = akn_ci["hcr"]
    hcr_best = best_ci["hcr"]
    row_hcr  = _row("hcr")
    lines += [
        "## Hallucinated Citation Rate (HCR ↓)",
        "",
        f"AKN-RLM HCR = **{hcr_akn['mean']:.4f}** (95 % CI: {hcr_akn['ci_lo']:.4f}–{hcr_akn['ci_hi']:.4f}). "
        f"Hybrid+Rerank HCR = {hcr_best['mean']:.4f} "
        f"(95 % CI: {hcr_best['ci_lo']:.4f}–{hcr_best['ci_hi']:.4f}). "
        + _sig_sentence(row_hcr, "HCR"),
        "",
    ]

    # --- JIR ---
    jir_akn  = akn_ci["jir"]
    jir_best = best_ci["jir"]
    row_jir  = _row("jir")
    lines += [
        "## Jurisdictional Infection Rate (JIR ↓)",
        "",
        f"AKN-RLM JIR = **{jir_akn['mean']:.4f}** (95 % CI: {jir_akn['ci_lo']:.4f}–{jir_akn['ci_hi']:.4f}). "
        f"Hybrid+Rerank JIR = {jir_best['mean']:.4f} "
        f"(95 % CI: {jir_best['ci_lo']:.4f}–{jir_best['ci_hi']:.4f}). "
        + _sig_sentence(row_jir, "JIR"),
        "",
    ]

    # --- AbstF1 ---
    ab_akn  = akn_ci["abstention_f1"]
    ab_best = best_ci["abstention_f1"]
    row_ab  = _row("abstention_f1")
    lines += [
        "## Abstention F1 (AbstF1)",
        "",
        f"AKN-RLM AbstF1 = **{ab_akn['mean']:.4f}** (95 % CI: {ab_akn['ci_lo']:.4f}–{ab_akn['ci_hi']:.4f}). "
        f"Hybrid+Rerank AbstF1 = {ab_best['mean']:.4f} "
        f"(95 % CI: {ab_best['ci_lo']:.4f}–{ab_best['ci_hi']:.4f}). "
        + _sig_sentence(row_ab, "AbstF1"),
        "",
    ]

    # --- Under-powered comparisons ---
    all_rows = _load_csv()
    large_but_ns = []
    for row in all_rows:
        if row["sig_level"] == "ns":
            delta = abs(float(row["delta_mean"]))
            akn_mean = float(row["akn_mean"])
            if akn_mean > 0 and delta / akn_mean >= 0.10:
                large_but_ns.append(row)

    lines += [
        "## Large effects not reaching significance (benchmark likely underpowered)",
        "",
    ]
    if large_but_ns:
        for row in large_but_ns:
            lines.append(
                f"- **{row['metric']}** vs **{DISPLAY_NAMES.get(row['baseline'], row['baseline'])}**: "
                f"Δ = {float(row['delta_mean']):+.4f} "
                f"(≥ 10 % of AKN mean = {float(row['akn_mean']):.4f}), "
                f"p_Bonf = {float(row['p_bonferroni']):.4f} (ns). "
                f"Effect: {row['effect']}."
            )
    else:
        lines.append("- None: all large effects reached significance after Bonferroni correction.")

    lines += [
        "",
        "---",
        "",
        "## Full results files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `results/stats/ci_<run_id>.json` | Bootstrap CIs per run × metric × stratum |",
        "| `results/stats/paired_pvalues.csv` | Paired bootstrap p-values + effect sizes |",
        "| `results/stats/table_tier1_with_pvalues.tex` | LaTeX Table 4.2 replacement |",
        "| `results/stats/table_tier2_with_pvalues.tex` | LaTeX Table 4.3 replacement |",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_rows = _load_csv()

    t1 = build_tier1(csv_rows)
    t2 = build_tier2(csv_rows)
    sm = build_summary(csv_rows)

    p1 = OUT_DIR / "table_tier1_with_pvalues.tex"
    p2 = OUT_DIR / "table_tier2_with_pvalues.tex"
    ps = OUT_DIR / "SUMMARY.md"

    p1.write_text(t1)
    p2.write_text(t2)
    ps.write_text(sm)

    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {ps}\n")
    print("=== table_tier1_with_pvalues.tex ===")
    print(t1)
    print("\n=== table_tier2_with_pvalues.tex ===")
    print(t2)


if __name__ == "__main__":
    main()
