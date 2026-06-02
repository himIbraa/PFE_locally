# Statistical Summary — Chapter 4 Evaluation (bootstrap inference)

> **AKN-RLM run used**: `rlm_dispatched_full_adu_pervasive` (Phase C SOTA, Cite F1 = 0.3010)
> **Note**: The canonical reference runs `rlm_dispatched_full_phase_e_final` and `rlm_dispatched_full_e4_trajectory` were not present on this machine.
> The Phase C run is the next-best available (Cite F1 = 0.3010 vs 0.3129 for the true SOTA).

All CIs are 95 % percentile bootstrap intervals (1 000 resamples, seed 42).
Paired p-values use 10 000 resamples. Significance thresholds after Bonferroni
correction across 6 baselines × 6 metrics = 36 comparisons:
`***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `ns` not significant.

## Citation F1 (article-level)

The AKN-RLM system achieves a mean Citation F1 of **0.3010** (95 % CI: 0.2515–0.3517) on the full 244-question benchmark. The strongest Phase-1 baseline (Hybrid+Rerank) reaches 0.1052 (95 % CI: 0.0850–0.1256). The difference (Δ = +0.1958) is **significant** (p_Bonf = 0.0036, **) with effect d=0.522.

## Hallucinated Citation Rate (HCR ↓)

AKN-RLM HCR = **0.0000** (95 % CI: 0.0000–0.0000). Hybrid+Rerank HCR = 0.0000 (95 % CI: 0.0000–0.0000). The difference (Δ = +0.0000) is **not significant after Bonferroni correction** (p_Bonf = 1.0000); the benchmark (n = 244) is likely underpowered to detect this effect.

## Jurisdictional Infection Rate (JIR ↓)

AKN-RLM JIR = **0.0164** (95 % CI: 0.0041–0.0328). Hybrid+Rerank JIR = 0.0287 (95 % CI: 0.0082–0.0492). The difference (Δ = -0.0123) is **not significant after Bonferroni correction** (p_Bonf = 1.0000); the benchmark (n = 244) is likely underpowered to detect this effect.

## Abstention F1 (AbstF1)

AKN-RLM AbstF1 = **0.6755** (95 % CI: 0.5775–0.7518). Hybrid+Rerank AbstF1 = 0.0000 (95 % CI: 0.0000–0.0000). The difference (Δ = +0.6755) is **significant** (p_Bonf = 0.0036, **) with effect d=15.261.

## Large effects not reaching significance (benchmark likely underpowered)

- **jir** vs **Dense**: Δ = +0.0082 (≥ 10 % of AKN mean = 0.0164), p_Bonf = 1.0000 (ns). Effect: ARD=0.008;RR=2.000.
- **jir** vs **Hybrid (RRF)**: Δ = +0.0082 (≥ 10 % of AKN mean = 0.0164), p_Bonf = 1.0000 (ns). Effect: ARD=0.008;RR=2.000.
- **jir** vs **Hybrid+Rerank**: Δ = -0.0123 (≥ 10 % of AKN mean = 0.0164), p_Bonf = 1.0000 (ns). Effect: ARD=-0.012;RR=0.571.
- **mrr** vs **Hybrid+Rerank**: Δ = +0.0582 (≥ 10 % of AKN mean = 0.2999), p_Bonf = 0.4824 (ns). Effect: d=0.159.
- **ndcg** vs **Hybrid+Rerank**: Δ = +0.0563 (≥ 10 % of AKN mean = 0.3147), p_Bonf = 0.5184 (ns). Effect: d=0.156.
- **jir** vs **KG (SPARQL)**: Δ = +0.0082 (≥ 10 % of AKN mean = 0.0164), p_Bonf = 1.0000 (ns). Effect: ARD=0.008;RR=2.000.
- **jir** vs **KG+Hybrid**: Δ = +0.0041 (≥ 10 % of AKN mean = 0.0164), p_Bonf = 1.0000 (ns). Effect: ARD=0.004;RR=1.333.

---

## Full results files

| File | Contents |
|------|----------|
| `results/stats/ci_<run_id>.json` | Bootstrap CIs per run × metric × stratum |
| `results/stats/paired_pvalues.csv` | Paired bootstrap p-values + effect sizes |
| `results/stats/table_tier1_with_pvalues.tex` | LaTeX Table 4.2 replacement |
| `results/stats/table_tier2_with_pvalues.tex` | LaTeX Table 4.3 replacement |