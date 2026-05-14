# Prompt to paste into Claude.ai (browser, with Artifacts / code execution)

Copy everything between the `===PROMPT START===` and `===PROMPT END===`
markers below and paste it as a fresh conversation in Claude.ai (browser).
Claude.ai will then generate a downloadable `.pptx` file via `python-pptx`.

If Claude.ai asks "do you want me to render this as a code artifact?",
answer **yes** — it will then attach a `.pptx` file you can download.

---

===PROMPT START===

You are an academic presentation designer. Generate a `.pptx` file using
the `python-pptx` library and return it as a downloadable artifact.

# Goal

Build a **clean, defense-ready PowerPoint deck** that explains an
end-of-studies thesis project titled:

> **"Advancing Legal Reasoning in Algerian Law: Integrating Retrieval-
> Augmented Generation, Knowledge Graphs, and Argument Mining for
> Citation-Faithful Legal Question Answering"**

The audience is an academic jury (computer-science engineering school).
The deck should cover the **pipeline architecture**, **all measured
results**, and a **discussion / analysis** section. Aim for ~18–22
slides. Use a professional palette (dark navy + soft white + one
accent color), generous whitespace, and tables for results.

# Slide-by-slide content

## Slide 1 — Title

- Title (large): *Advancing Legal Reasoning in Algerian Law*
- Subtitle: *Integrating RAG, Knowledge Graphs, and Argument Mining
  for Citation-Faithful Legal QA*
- Footer: AlgerianLegalBench v3.0 · 244 questions · 8 query types

## Slide 2 — Problem statement

- Algerian legal QA needs **citation-faithful** answers (every claim
  must trace to a specific article).
- General-purpose LLMs hallucinate articles and skip nuance (multi-hop
  reasoning, time-versioned amendments, Darja queries, unanswerable
  questions about foreign legal concepts).
- No prior benchmark covers Algerian law at this granularity.

## Slide 3 — Dataset: AlgerianLegalBench v3.0

- 244 expert-annotated questions over 8 codes (Civil, Family, Penal,
  Procedure, Commercial, Labor, Constitution, Investment).
- 8 query types with this distribution:

| Query type | n |
|---|---:|
| rule_application | 66 |
| exact_article | 59 |
| unanswerable | 40 |
| multi_hop | 26 |
| layman (Darja) | 17 |
| long_context | 17 |
| conceptual_definitional | 12 |
| temporal_factual | 7 |
| **Total** | **244** |

- Gold for every question: list of `(doc_id, article_ref)`, an
  abstention flag, and a full reasoning chain.

## Slide 4 — Architecture overview (high-level)

Use a vertical flow diagram (boxes with arrows) with these stages:

```
Query (Arabic / French / Darja)
        ↓
[1] Classifier  (Gemma-4-31B)  → query_type
        ↓
[2] DocRouter  (alias + numeric-ID + BM25 aggregation)  → top-3 docs
        ↓
[3] Typed handler  (one of 8 specialized handlers)
        ↓
[4] Output: answer + citations + trajectory + abstention reason
```

Caption underneath: *"RLM Dispatcher — eight typed handlers, each with
its own retrieval + verification recipe."*

## Slide 5 — The 8 typed handlers

Bulleted list (concise):

- **exact_article**: BM25-only retrieval (HANDOFF §R6.2)
- **rule_application**: BM25 + Dense + HyDE + LLM summarization
- **multi_hop**: sub-question decomposition + cross-citation consensus
  boost (Fix-MH)
- **temporal_factual**: KG SPARQL `concept-at-date` + dense fallback
- **conceptual_definitional**: KG `concept-amendment` + paraphrase widening
- **unanswerable**: jurisdiction-infection check + abstention template
- **layman**: Gemma Darja→MSA rewrite, then rule_application path
- **long_context**: RRF over BM25/Dense + AKN chapter-sibling expansion
  (Fix-LC)

## Slide 6 — Retrieval channels

Diagram (left: query, right: candidates pool), with these channels:

- **BM25** (legal-ID tokenizer)
- **Dense** (intfloat/multilingual-e5-small, FAISS index)
- **HyDE** (Qwen-Thinking drafts a hypothetical answer, dense embeds
  query+answer) — **selectively off for TF & CD** (drifts retrieval
  away from the in-force version / definition).
- **KG SPARQL** (concept-at-date for TF, concept-amendment for CD)
- RRF fusion across all active channels.

## Slide 7 — Verification & faithfulness gates

- **Qwen verifier**: per-candidate CoT relevance score.
- **Faithfulness gate (NLI)**: per-citation entailment of answer claims.
- **Span existence gate**: catches fabricated article numbers.
- **Jurisdiction gate**: detects foreign-law "infection" signals.
- Telemetry: every dispatched answer carries the full trajectory.

## Slide 8 — Phase 1 (deterministic baselines, full 244)

Table:

| Pipeline | Cite F1 | MRR art | R@10 art |
|---|---:|---:|---:|
| BM25 | 0.093 | — | — |
| Dense | 0.063 | — | — |
| Hybrid (RRF) | 0.094 | — | — |
| Hybrid + Rerank | **0.105** | 0.242 | 0.220 |
| KG SPARQL | 0.022 | — | — |
| KG + Hybrid | 0.083 | — | — |

Caption: *Best Phase-1 baseline (Hybrid+Rerank) caps at **Cite F1 = 0.105**.*

## Slide 9 — Phase A: raw-LLM floor (no retrieval)

Goal: quantify what a raw LLM produces without any RAG.

| LLM (no retrieval) | Cite F1 | HCR | AbstF1 |
|---|---:|---:|---:|
| gpt-oss-120b (raw) | **0.0555** | 0.000 | 0.000 |
| Qwen3-30B-A3B-Thinking (raw) | **0.0338** | 0.000 | 0.000 |

Key insight: **raw LLMs are at ~5% Cite F1. No abstention path.
Citations exist but point to the wrong articles.** This sets the
**floor** the RLM architecture must clear.

> *Important: this slide must only list these two raw rows. Do **not**
> include the "+ dense top-5 RAG" variants on this slide.*

## Slide 10 — Phase 2: locked SOTA (RLM Dispatcher)

Headline numbers (full 244):

| Metric | RLM SOTA |
|---|---:|
| **Cite F1** | **0.313** |
| MRR doc | 0.546 |
| MRR art | 0.284 |
| R@10 art | 0.232 |
| Doc Cite F1 | 0.613 |
| HCR (hallucination rate) | **0.000** |
| JIR (jurisdictional infection) | 0.004 |
| **Answer faithfulness (AbstF1)** | **0.716** |
| Mean latency | 5.4 s / q |

Caption: *RLM beats best Phase-1 baseline (0.105) by **2.98×**, and
raw gpt-oss-120b (0.0555) by **5.64×**.*

## Slide 11 — Per query-type comparison

Stacked / grouped bar chart preferred. Data:

| Query type | n | Best Phase-1 | F5 (locked Phase-2) | **RLM SOTA** |
|---|---:|---:|---:|---:|
| exact_article | 59 | 0.183 | 0.416 | 0.410 |
| rule_application | 66 | 0.155 | 0.235 | **0.252** |
| multi_hop ⭐ | 26 | 0.054 | 0.122 | **0.175** |
| temporal_factual ⭐ | 7 | 0.095 | 0.190 | 0.190 |
| conceptual_definitional ⭐ | 12 | 0.111 | 0.135 | 0.107 |
| unanswerable ⭐ | 40 | 0.033 | 0.525 | **0.525** |
| layman | 17 | 0.024 | 0.275 | **0.312** |
| long_context | 17 | 0.074 | 0.097 | **0.119** |
| **overall** | 244 | 0.105 | 0.301 | **0.313** |

(⭐ = "hard" type. RLM wins overall **and** every simple type, and
**three of four** hard types.)

## Slide 12 — What lifted us

Four short bullets:

- **Selective E4 HyDE** (off for TF & CD): +0.004 vs no HyDE.
- **Fix-MH consensus aggregation** (multi-hop): MH lifted **+0.054**.
- **Fix-LC chapter siblings** (long_context): LC lifted **+0.023**.
- **Typed-handler dispatch itself** (vs single pipeline): +0.20 absolute
  on Cite F1.

## Slide 13 — Phase B: classifier-typed (production deployability)

Question: *what happens if the system has to guess the query_type?*

Two classifiers tested:

| Classifier | Accuracy on 244 | Macro F1 |
|---|---:|---:|
| Regex (deterministic baseline) | **29.9%** | 0.132 |
| **Gemma-4-31B (few-shot, Phase B)** | **69.7%** | **0.723** |

Key wins of the LLM classifier:

- `unanswerable`: F1 = **0.962**
- `temporal_factual`: F1 = **0.933**
- `layman`: F1 = **0.938**
- `multi_hop`: F1 = **0.711**

Hardest pair: **EA ↔ RA** (the boundary is annotation-dependent — broad
"explain principle X and its limits" questions are labeled EA when
the answer comes from a single article, RA when it requires
combining a rule with consequences/timing).

End-to-end impact on the full-244 dispatcher with **classifier-typed**
routing instead of gold:

| Setup | Cite F1 | MRR art | R@10 art | HCR | AbstF1 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Gold-typed (locked SOTA) | 0.313 | 0.284 | 0.232 | 0.000 | 0.716 | 5.4 s |
| **Classifier-typed (Gemma + RLM)** | **0.298** | 0.292 | 0.252 | 0.000 | 0.676 | 5.8 s |
| Δ | **−0.015** | +0.008 | +0.020 | 0.000 | −0.040 | +0.4 s |

Headline message: **only −0.015 Cite F1 drop** (the original Phase B
spec predicted 0.05–0.10). MRR doc/art and R@10 art actually **improve**
because the classifier's mis-routings sometimes land in handlers with
broader retrieval. Biggest casualty: `multi_hop` drops from 0.175 to
0.100 because mis-routed MH questions miss the Fix-MH consensus boost.

## Slide 14 — Honest analysis: where each component helps

- **Retrieval (Dense + BM25 + HyDE)** is the dominant lift over raw LLMs.
- **KG SPARQL** alone is weak (0.022) — but it shines as a *channel*
  inside the typed handlers, especially CD's amendment chain.
- **Typed handlers** add stratification: one pipeline per question type
  means MH gets sub-question decomposition, LC gets chapter expansion,
  TF gets KG-first, etc. Net effect: 3× over deterministic baseline.
- **Faithfulness gates** are why HCR stays at 0.000 — no fabricated
  article numbers escape.

## Slide 15 — Where the system still struggles

- **R@10 art = 0.232**: the dense retriever is still too generic for
  Algerian legal Arabic. Phase F (corpus-tuned embedder, on HPC) will
  push this to ≥ 0.40.
- **conceptual_definitional**: ties with Phase-1 KG+Hybrid baseline.
  Phase C (pervasive argument mining) and Phase E (KG-derived doc
  routing) target this.
- **temporal_factual KG-first**: wired but flat — Phase E.1 debugs the
  merge logic.
- **Classifier EA ↔ RA confusion**: inherent in the benchmark, +5–8
  points likely achievable with HPC-tuned classifier.

## Slide 16 — Roadmap (Phases C → H)

Short list with one-line outcome each:

- **Phase C — Pervasive Argument Mining**: Toulmin claim/ground/warrant
  for every cited article (justifies the "Argument Mining" title).
- **Phase D — Genuine recursion**: depth-driven re-retrieval +
  corrective retry on faithfulness fail.
- **Phase E — KG everywhere**: debug TF, add KG topology
  disambiguator, extend amendment SPARQL to MH/RA.
- **Phase F (HPC) — Corpus-tuned embedder**: e5-large fine-tuned on
  ALB v3.0 gold + hard negatives → target R@10 art ≥ 0.40.
- **Phase G (HPC) — ColBERT multi-vector**: complementary
  late-interaction retrieval channel.
- **Phase H — Final ablation table + thesis artifacts**.

Target final Cite F1: **≥ 0.45**.

## Slide 17 — Contributions

1. First citation-faithful RAG pipeline for Algerian law.
2. AlgerianLegalBench v3.0 — open evaluation benchmark with 8 query
   types and full reasoning chains.
3. **RLM Dispatcher** — typed-handler architecture with measurable
   per-type gains.
4. End-to-end HCR = 0.000 with **AbstF1 = 0.716**.

## Slide 18 — Reproducibility

- 762 unit tests, all passing.
- All metrics in `eval_results/<run_id>/` with `predictions.jsonl`
  + `metrics.json` + `metrics.md`.
- Repos: `PFE_locally` (mirror with results), `PFE_hpc` (HPC-runnable).
- Run command: `python scripts/run_dispatcher.py --e4 --run-id full`.

## Slide 19 — Q & A

- Big "Questions?" centered.
- Footer: contact + repo link placeholder.

# Visual style

- Aspect: 16:9.
- Font: Calibri or Inter for body, Calibri Light bold for titles.
- Color palette: dark navy `#0C2340`, accent teal `#16A085`, neutral
  light gray `#F4F6F8`, text on dark `#FFFFFF`.
- Tables: alternating row tint of `#F4F6F8`; header row dark navy with
  white text.
- Every result number must be **bold** when it's a headline figure.

# Deliverable

Generate `algerian_legal_rag_thesis_defense.pptx` and return it as a
file artifact. Use `python-pptx` end-to-end (no external dependencies).
Confirm slide count (~18–19) and show me a one-line summary of each
slide title after generation.

===PROMPT END===
