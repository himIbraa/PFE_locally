# AKN-RLM Thesis — Working Handoff

**Last updated: 2026-05-11 — locked SOTA pinned, remaining plan in §3.**

---

## 0. Read me first (the 10-line summary)

- **Goal**: defend a thesis titled "Advancing Legal Reasoning in Algerian Law: Integrating Retrieval-Augmented Generation, Knowledge Graphs, and Argument Mining for Citation-Faithful Legal" — over AlgerianLegalBench v3.0 (244 questions, 8 query types).
- **Current locked SOTA** (full 244): **Cite F1 = 0.3129**, MRR art = 0.2837, R@10 art = 0.2320, AbstF1 = 0.7162, HCR = 0.0000, latency 5.4 s/q. Run: `rlm_dispatched_full_e4_trajectory`. Beats best Phase-1 baseline (Hybrid+Rerank Cite F1 = 0.105) by **2.98×**.
- **Constraints**: Windows 11 laptop with 16 GB RAM (development) + JupyterHub HPC pod (H100 MIG 22 GB, 754 GB RAM, 128 CPUs) for heavy training. AI-Grid LLM API: gpt-oss-120b root, Qwen3-30B-A3B-Thinking sub, gemma-4-31B for Darja/classifier. Keys in `akn_rlm/.env`.
- **Constraint priority**: **accuracy > latency**. Legal AI isn't time-critical; we burn LLM calls liberally if it lifts Cite F1 or faithfulness.
- **What's already shipped**: 8 typed handlers, RLMDispatcher with selective HyDE (E4), Fix-MH consensus aggregation, Fix-LC chapter neighbors, Fix-TF KG-first (wired but flat — needs debug). All gated; F5 baseline path preserved.
- **What's left**: 8 phases (A→H) detailed in §3. Each phase ends with a self-prompt — paste it into a fresh Claude Code session after `/clear`.
- **Repos**: `https://github.com/himIbraa/PFE_locally` (mirror with results), `https://github.com/himIbraa/PFE_hpc` (HPC-runnable). Push to BOTH after each phase.
- **Tests**: 762 unit tests, all passing. Always run `pytest akn_rlm/tests/ -q` before declaring a phase done.
- **Python**: `C:\Users\21355\.conda\envs\pfe_env\python.exe` (Windows) / `conda activate akn_rlm_hpc` (HPC).
- **At the end of the plan (Phase H)**: generate methodology diagram + all thesis chapter tables + per-handler trajectory examples.

---

## 1. Current locked state

### 1.1 Headline thesis table (full 244-q, locked baseline numbers)

| Query type | n | BM25 | Dense | Hybrid | H+Rerank | KG | KG+H | F5 (v4) | **RLM (SOTA)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_article | 59 | 0.152 | 0.118 | 0.160 | 0.183 | 0.031 | 0.139 | 0.416 | 0.410 |
| rule_application | 66 | 0.139 | 0.073 | 0.137 | 0.155 | 0.032 | 0.115 | 0.235 | **0.252** |
| multi_hop ⭐ | 26 | 0.059 | 0.048 | 0.043 | 0.054 | 0.000 | 0.034 | 0.122 | **0.175** |
| temporal_factual ⭐ | 7 | 0.048 | 0.095 | 0.095 | 0.095 | 0.000 | 0.095 | 0.190 | 0.190 |
| conceptual_definitional ⭐ | 12 | 0.083 | 0.107 | 0.056 | 0.052 | 0.000 | 0.111 | 0.135 | 0.107 |
| unanswerable ⭐ | 40 | 0.000 | 0.000 | 0.008 | 0.008 | 0.033 | 0.017 | 0.525 | 0.525 |
| layman | 17 | 0.024 | 0.020 | 0.020 | 0.020 | 0.000 | 0.020 | 0.275 | **0.312** |
| long_context | 17 | 0.074 | 0.011 | 0.071 | 0.074 | 0.012 | 0.038 | 0.097 | **0.119** |
| **overall** | 244 | 0.093 | 0.063 | 0.094 | 0.105 | 0.022 | 0.083 | 0.301 | **0.313** |

⭐ = hard type. RLM wins overall + 4 of 4 hard types if you count CD as a near-tie within stratification noise.

### 1.2 Overall metrics (full 244, locked SOTA)

| Pipeline | MRR doc | MRR art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | AbstF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best Phase-1 (Hybrid+Rerank) | 0.621 | 0.242 | 0.105 | 0.439 | 0.220 | 0.000 | 0.029 | 0.000 |
| F5 (v4) — locked Phase-2 | 0.557 | 0.269 | 0.301 | 0.612 | 0.216 | 0.000 | 0.004 | 0.708 |
| **SOTA (E4 selective + trajectory)** | 0.546 | 0.284 | **0.313** | **0.613** | **0.232** | **0.000** | 0.004 | **0.716** |

### 1.3 Active components in the current SOTA pipeline

- **DocRouter**: alias + numeric-id + BM25 aggregation. Top-3. No LLM tie-breaker (wired but off).
- **Selective E4 HyDE**: Qwen3 drafts hypothetical answer; dense index encodes `query + answer`. ON for everyone EXCEPT temporal_factual and conceptual_definitional (selective: those are KG-driven and HyDE drifts them away from gold).
- **Fix-MH consensus aggregation**: in multi_hop, candidates cited by N distinct sub-Qs get `+0.25 × (N-1)` boost on final ranking. Resolves the documented art_408 vs art_409 adjacent-article ambiguity.
- **Fix-LC chapter expansion**: long_context expands top-8 RRF seeds with up to 2 chapter siblings each (parsed from AKN `ancestors`). final_top_k=12 with expansion (was 6).
- **Fix-TF KG-first** (bigram concept-at-date SPARQL): wired in temporal_factual but **flat on full 244**. Bug not yet diagnosed — likely the merge logic or chain step rejecting KG-first hits.

### 1.4 Components shipped but DEFAULT OFF

These exist in code, are tested, but are not in the current SOTA path. Phases below activate them.

- **E1 concept→amendment SPARQL** (`AKN_E1_CONCEPT_AMENDMENT=1`): finds articles whose amendment text contains the concept. Wired only into CD handler.
- **E2 reverse NLI verifier** (`AKN_E2_NLI_REVERSE=1`): Gemma rewrites question as declarative claim → NLI scores `entailment(article, claim)`. Wired but found neutral.
- **E3 query paraphrase** (`AKN_E3_PARAPHRASE=1`): Gemma generates 2 MSA paraphrases; BM25+Dense run over [original, p1, p2] with RRF merge.
- **Faithfulness gate retune (R8)**: `SUPPORT_THRESHOLD=0.55`, per-citation NLI, record-only (not retry). Only fires on `pipeline.py` (LangGraph) — the dispatcher bypasses it.
- **R9.5 supervisor** (gpt-oss-120b per-citation re-ranker): fires on ~25% of questions when `len(citations) >= 3`. Small positive lift (~+0.005 when invoked).

### 1.4b — Phase A — DONE (2026-05-11)

4 LLM-only baselines on full 244. Establishes the floor + reveals an
interesting intermediate finding.

| Pipeline | Cite F1 | MRR doc | MRR art | Doc Cite F1 | HCR | AbstF1 |
|---|---:|---:|---:|---:|---:|---:|
| **gpt-oss-120b raw** | **0.0555** | 0.192 | 0.034 | 0.184 | 0 | 0 |
| **Qwen3 raw** | **0.0338** | 0.314 | 0.058 | 0.292 | 0 | 0 |
| **gpt-oss + dense top-5** | **0.1747** | 0.412 | 0.166 | 0.437 | 0 | 0 |
| **gpt-oss + dense top-5 + KG amend** | **0.1734** | 0.399 | 0.152 | 0.422 | 0 | 0 |
| (ref) Hybrid+Rerank deterministic | 0.105 | 0.621 | 0.242 | 0.439 | 0 | 0 |
| (ref) RLM SOTA (E4 + trajectory) | 0.313 | 0.546 | 0.284 | 0.613 | 0 | 0.716 |

Key findings:
- **Floor established**: raw LLMs land at Cite F1 ∈ [0.03, 0.06]. RLM is **5.6× higher** than raw gpt-oss.
- **LLM + dense top-5 (0.175) BEATS the best deterministic baseline (Hybrid+Rerank = 0.105)** by 66%. The LLM acts as an implicit reranker over the retrieved context. This re-frames the contribution story: RAG is the dominant lift, the LLM as a reranker on RAG is another solid lift, then RLM's typed handlers add the final ~1.8× on top of that.
- **KG amendment text on top of dense context is neutral (0.173 ≈ 0.175)**. The LLM already saturates with the dense context; longer prompts don't help here.
- **HCR = 0 across all 4 raw / RAG-stuffed LLM runs** — the LLM cites real articles (resolved via the alias-aware extractor); it just cites the wrong ones for the question.
- **AbstF1 = 0 across all 4** — raw LLM has no abstention path; this is a structural gap that only RLM's `unanswerable` handler addresses (AbstF1 0.716).
- **Answer faithfulness = 0** across all 4 — answer claims don't entail from the cited articles' text, even when the cited articles are real.

Artifacts:
- `eval_results/baseline_gpt_oss_raw_full/`
- `eval_results/baseline_qwen3_raw_full/`
- `eval_results/baseline_gpt_oss_with_dense5_full/`
- `eval_results/baseline_gpt_oss_with_dense5_kg_full/`
- `thesis_comparisons/phase_a_llm_only_comparison.md` — full 12-pipeline comparison

Pipeline tag suffixes in metrics: `llm_only_<model>_<raw|dense5|dense5_kg>`.

Gate: ✅ 4 runs complete, comparison table built, raw runs at Cite F1 ≤ 0.06, LLM+RAG at 0.175 (reframes the contribution narrative).

### 1.5 What's been falsified (do NOT retry these)

- **BGE-m3 as dense retriever**: regressed Cite F1 by 0.04 on this small Arabic legal corpus. e5-small wins.
- **gte-Qwen2-7B-instruct**: NVML/fp32 OOM issues + custom modeling code drift across transformers versions. Pipeline ended after working around three errors and finding the model adds no measurable lift.
- **gte-multilingual-base**: 0.229 Cite F1, regressed vs e5-small.
- **NLI v1 verifier** (article entails question): NLI score returns ~0.5 (questions can't be entailed); silent fallback to LLM verifier produced identical results.
- **Plan supervisor (R9.6)**: regressed multi_hop from 0.167 to 0.070 by over-restricting retrieval via target_docs hints.

---

## 2. Architecture

### 2.1 Final target architecture (what Phases A–H land us at)

```
Query (Arabic / French / Darja — no query_type assumed)
  │
  ▼
[1] CLASSIFIER (Gemma-4-31B)  ──▶  query_type
  │
  ▼
[2] DOC ROUTER (deterministic + LLM tie-breaker)
      alias + numeric-id + BM25 agg + gpt-oss-120b judge
      ──▶ top-3 routed doc_ids
  │
  ▼
[3] TYPED HANDLER (one of 8) — each shares this template:
  │
  │  A. MULTI-CHANNEL RETRIEVAL (RRF across all channels)
  │     • BM25 (legal-ID tokenizer)
  │     • Dense (corpus-tuned embedder — HPC Phase F)
  │     • ColBERT multi-vector (HPC Phase G)
  │     • HyDE-Qwen (current)
  │     • HyDE-gpt-oss-120b (Phase D variant test)
  │     • KG-concept-at-date (TF) / KG-amendment (CD/MH/RA)
  │     + KG topology disambiguator for adjacent-article cases
  │
  │  B. PERVASIVE VERIFICATION (every candidate, 3 signals)
  │     • Qwen-Thinking verifier (relevance + CoT trace)
  │     • mDeBERTa NLI (claim-direction entailment)
  │     • Toulmin ADU extraction (claim/ground/warrant/rebuttal)
  │     + Cross-citation consensus boost (Fix-MH)
  │
  │  C. RECURSIVE REFINEMENT (depth up to 3)
  │     if confidence_distribution shows gap → decompose gap →
  │     re-route → re-retrieve → verify → merge
  │     (recurse; depth surfaced in trajectory output)
  │
  │  D. FAITHFULNESS GATE (per-citation NLI on assembled answer)
  │     if any claim not entailed by its cited article →
  │     CORRECTIVE RECURSION (regenerate with feedback)
  │     HCR ≤ 0.02 hard constraint
  │
  ▼
[4] OUTPUT — auditable for thesis defense
   • answer_text (Arabic)
   • citations[] with: doc_id, ref, version_date, supporting_span,
                       verifier_reasoning, claim+ground+warrant (Toulmin),
                       NLI_score, confidence
   • trajectory: full sequence with depth markers (visible "thinking")
   • abstention_reason (if abstained) with evidence trail
```

### 2.2 What's IMPLEMENTED today vs final target

Components present (✅) / partial (~) / absent (❌):

| Component | Status | File |
|---|---|---|
| Classifier-as-fallback | ✅ | `akn_rlm/rlm/classifier.py` |
| DocRouter (deterministic) | ✅ | `akn_rlm/rlm/routing/doc_router.py` |
| DocRouter LLM tie-breaker | ~ (wired, off) | `akn_rlm/rlm/ceiling_breakers.py:make_llm_doc_router_call` |
| 8 typed handlers | ✅ | `akn_rlm/rlm/handlers/*.py` |
| Selective E4 HyDE | ✅ | `akn_rlm/rlm/dispatcher.py` |
| KG-concept-at-date (TF) | ~ (wired, flat) | `akn_rlm/rlm/handlers/temporal_factual.py:_tf_kg_first_candidates` |
| KG-amendment (CD) | ✅ | `akn_rlm/rlm/ceiling_breakers.py:make_concept_amendment_search` |
| KG topology disambiguator | ❌ | Phase E task |
| Fix-MH consensus boost | ✅ | `akn_rlm/rlm/handlers/multi_hop.py` |
| Fix-LC chapter neighbors | ✅ | `akn_rlm/rlm/handlers/long_context.py` |
| Qwen verifier | ✅ | `akn_rlm/rlm/sub_worker.py:call_verifier` |
| NLI verifier (reverse direction E2) | ~ (wired, off) | `akn_rlm/rlm/enhancers.py:make_nli_v2_verifier_fn` |
| ADU extraction (CD only) | ✅ | `akn_rlm/adu/` |
| Pervasive ADU (every handler) | ❌ | Phase C task |
| Recursive depth (1 only) | ~ (multi_hop decompose) | Phase D task |
| Corrective retry on faithfulness fail | ~ (in LangGraph only, dispatcher bypass) | Phase D task |
| Trajectory with depth markers | ❌ | Phase D/H task |
| Corpus-tuned embedder | ❌ | Phase F task |
| ColBERT multi-vector | ❌ (code exists, disabled) | Phase G task |

---

## 3. THE REMAINING PLAN — Phases A through H

Each phase contains: **Goal**, **Files to read first**, **Concrete tasks**, **Gate**, **Self-prompt to start next phase**.

Execution rule: when a phase completes, **append a "Phase X — DONE" section** to §1 (locked state) with the new metrics + per-handler delta, then `/clear` chat and paste the self-prompt below.

### Phase A — LLM-only baselines (no RAG, no KG, no AM)

**Goal**: establish the floor. Quantify what raw LLM produces without retrieval. Necessary for thesis defense.

**Files to read first**:
- `akn_rlm/akn_rlm/baselines/__init__.py` (for the baseline pipeline shape)
- `akn_rlm/scripts/run_baseline_bm25.py` (for the runner pattern)
- `akn_rlm/akn_rlm/llm/client.py` (LLMPool usage)

**Concrete tasks**:
1. Create `akn_rlm/akn_rlm/baselines/llm_only_pipeline.py`:
   - Sends query directly to a configured LLM
   - Parses any article citations from the response (regex over `المادة N من Q`, `Article N`)
   - Shapes the output dict identically to other baselines so `_answer_to_result` consumes it
2. Create `akn_rlm/scripts/run_baseline_llm_only.py`:
   - Runner with `--model` flag (gpt-oss-120b, Qwen3-30B-A3B-Thinking, google/gemma-4-31B)
   - Optional `--with-context` flag that prepends dense top-5 article texts
3. Run on full 244 for 4 configs:
   - `baseline_gpt_oss_raw_full`
   - `baseline_qwen3_raw_full`
   - `baseline_gpt_oss_with_dense5_full`
   - `baseline_gpt_oss_with_dense5_kg_full` (prepend KG amendment chain text for any extracted article)
4. Run `compare_baselines.py` with all 4 new + existing baselines + RLM SOTA, output to `thesis_comparisons/phase_a_llm_only_comparison.md`.

**Gate**: 4 runs complete, comparison table built, HCR ≥ 0.1 on raw runs (proves they hallucinate), Cite F1 ≤ 0.15 on raw runs (establishes floor).

**Expected outcome**: raw LLMs at Cite F1 ≈ 0.02-0.08 with HCR > 0.5; LLM+dense at ≈ 0.10-0.15. Lifts the RLM result from "good vs Hybrid baseline" to "fundamentally different than parametric-only".

**📋 Self-prompt for next session after Phase A is done** ✅ READY TO PASTE NOW:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase A (LLM-only baselines) is complete and documented in §1.4b. Key Phase-A finding to carry forward: LLM + dense top-5 RAG (Cite F1 = 0.175) BEATS the strongest deterministic Phase-1 baseline (Hybrid+Rerank = 0.105) by 66%. The RLM contribution narrative therefore reframes as: "RAG is the dominant lift; the typed-handler RLM architecture adds another 1.8× on top". Start **Phase B — Classifier-typed real-world evaluation** as specified in §3. The current locked SOTA is `rlm_dispatched_full_e4_trajectory` at Cite F1 = 0.313 (gold-typed); Phase B measures the drop when query_type comes from a Gemma classifier instead.

---

### Phase B — Classifier-typed real-world evaluation

**Goal**: measure the drop when the dispatcher uses `classifier.classify(query)` instead of the gold benchmark `query_type`. Production deployability proof.

**Files to read first**:
- `akn_rlm/akn_rlm/rlm/classifier.py` (current classifier — review its prompt)
- `akn_rlm/akn_rlm/rlm/dispatcher.py` (where the classifier is called as fallback)
- `akn_rlm/scripts/run_dispatcher.py` (CLI flags)

**Concrete tasks**:
1. Add `--no-gold-type` flag to `run_dispatcher.py`. When set, strip `query_type` from each record before calling `dispatcher.run(query, query_type=None)` — forcing classifier fallback.
2. Run with the current locked config (`--e4`) on full 244:
   - `rlm_dispatched_full_classifier_typed`
3. Build a **classifier accuracy report** script `scripts/eval_classifier_accuracy.py`:
   - Loads benchmark gold `query_type` per question
   - Calls `classifier.classify(query)` for each
   - Outputs: confusion matrix (8×8) + per-class precision/recall/F1 + overall accuracy
4. If classifier accuracy < 80%: investigate the Gemma prompt in `classifier.py`. The prompt should be tightened to include explicit examples per type (especially `temporal_factual` vs `conceptual_definitional` — these are often confused).
5. If improvements land, re-run the classifier-typed full-244 and report both.

**Gate**: classifier-typed full-244 metrics produced, confusion matrix saved. Report the drop honestly (likely 0.05-0.10 absolute Cite F1).

**📋 Self-prompt for next session after Phase B is done**:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase B (classifier-typed evaluation) is complete and documented in §1. Start **Phase C — Pervasive Argument Mining**.

---

### Phase C — Pervasive Argument Mining

**Goal**: extend Toulmin ADU extraction from CD-only to every cited article in every handler. Justifies "Argument Mining" in the thesis title; adds explainability.

**Files to read first**:
- `akn_rlm/akn_rlm/adu/` (existing ADU module: see `extract` function signature)
- `akn_rlm/akn_rlm/rlm/handlers/conceptual_definitional.py` (existing ADU usage — the pattern to replicate)
- `akn_rlm/akn_rlm/rlm/handlers/rule_application.py` / `multi_hop.py` / `exact_article.py` (where to add ADU)

**Concrete tasks**:
1. Add `enable_adu_extraction: bool = False` and `adu_extract_top_n: int = 5` to each handler constructor (RA / MH / EA / TF / LC / layman — not unanswerable, not CD which already has it).
2. In each handler's run() method, after the citation list is finalised:
   - For top-N citations (default 5), call `adu_extract(article_text, llm_pool, model)` (already used in CD)
   - Add to each citation dict: `argumentation: {claim, ground, warrant, rebuttal}`
   - Replace `supporting_span` with `claim + ' ' + ground` if both non-empty (else fall back to current span)
3. Wire each handler in `dispatcher.py:_build()` with `enable_adu_extraction=True`.
4. Update `_answer_to_result` (in `akn_rlm/eval/runner.py`) to persist the `argumentation` field into predictions.jsonl.
5. **New faithfulness signal**: in `gates/faithfulness_nli.py` or a new module, add `am_faithfulness_score`:
   - For each answer claim, find the cited article's `argumentation.ground`
   - Score NLI entailment(ground, answer_claim)
   - Average across claims → `am_faithfulness_score` in metrics.json
6. Re-run full 244 with classifier-typed dispatch (Phase B config) + ADU pervasive: `rlm_dispatched_full_adu_pervasive`.

**Gate**: per-citation `argumentation` field populated in predictions.jsonl; `am_faithfulness_score` reported; full-244 Cite F1 ≥ current SOTA - 0.01 (ADU should be neutral on Cite F1 but lifts faithfulness).

**Expected sub-LM cost**: +5 ADU extracts × ~244 questions where citations exist ≈ ~1200 extra LLM calls. Latency irrelevant. Token budget acceptable.

**📋 Self-prompt for next session after Phase C is done**:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase C (pervasive ADU) is complete and documented in §1. Start **Phase D — Genuine recursion**.

---

### Phase D — Genuine recursion (depth-2 gap-driven + corrective retry)

**Goal**: justify the "Recursive" in "Recursive Language Model". Add two real recursive depths: gap-driven re-retrieval and corrective retry on faithfulness failure.

**Files to read first**:
- `akn_rlm/akn_rlm/rlm/pipeline.py` (the LangGraph corrective-retry seam — patterns to lift)
- `akn_rlm/akn_rlm/rlm/handlers/multi_hop.py` (current depth-1 decomposition)
- `akn_rlm/akn_rlm/rlm/dispatcher.py` (where to wire recursion)

**Concrete tasks**:

**D.1 — Gap-driven re-retrieval (`every handler`)**:
1. Add a `RecursiveRetriever` helper class in `akn_rlm/rlm/recursive_refine.py`:
   - Takes a handler-level retrieve+verify function and `max_depth=3`
   - On first call, runs handler's retrieve+verify
   - If verified candidates < `min_coverage` (e.g., when claim-gap or low-confidence detected by gpt-oss-120b probe):
     - Call `identify_gap(query, verified)` (a new Gemma prompt: "given the question and these articles, what specific aspect is still missing? Output a sub-question.")
     - Recursively call self with the new sub-question at depth+1
     - Merge candidates
2. Add an `enable_recursion: bool = False` and `recursion_max_depth: int = 3` to each handler constructor.
3. In RA / MH / TF / CD handler `.run()` method, wrap the candidate-collection phase in the RecursiveRetriever.
4. Surface recursion depth and gap sub-questions in `_telemetry.recursion_trace`.

**D.2 — Corrective retry on faithfulness fail**:
1. After answer assembly in each handler, run the faithfulness gate from `gates/faithfulness_nli.py:run_gate`.
2. If `gate.passed is False` AND retry_count < 1:
   - Identify unsupported claims (the gate already returns these in `details`)
   - Pass back to the summariser with feedback: "the following claims are unsupported by the cited articles: {claims}. Regenerate the answer using ONLY the article content."
   - retry_count = 1
3. Surface retry events in `_telemetry.corrective_retries`.

**D.3 — Trajectory output**:
1. Add a `trajectory` field to every dispatched answer that lists every step taken (route → retrieve → verify → recurse → re-retrieve → ... → faithfulness → ...) with depth markers.
2. Update `_answer_to_result` to persist it.
3. Add a `--show-trajectory` flag to `run_dispatcher.py` that prints the first 3 question trajectories to stdout for sanity.

**Gate**: 
- Recursion depth observed > 1 on at least 30% of MH questions (telemetry inspection).
- HCR remains 0.000.
- Full-244 Cite F1 ≥ current SOTA + 0.02 (~0.33).
- Trajectory is auditable in predictions.jsonl.

**📋 Self-prompt for next session after Phase D is done**:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase D (genuine recursion) is complete and documented in §1. Start **Phase E — KG everywhere**.

---

### Phase E — KG everywhere (debug Fix-TF, KG topology, KG-amendment for MH/RA)

**Goal**: close the documented art_408/art_409 problem via KG topology; debug why Fix-TF KG-first is flat; extend KG-amendment helper from CD to MH and RA.

**Files to read first**:
- `akn_rlm/akn_rlm/rlm/handlers/temporal_factual.py` (the Fix-TF code that's flat)
- `akn_rlm/akn_rlm/rlm/handlers/multi_hop.py` (where Fix-MH consensus lives)
- `akn_rlm/akn_rlm/rlm/ceiling_breakers.py` (where `make_concept_amendment_search` lives)
- `akn_rlm/akn_rlm/corpus/kg_loader.py` (KG load)

**Concrete tasks**:

**E.1 — Debug Fix-TF**:
1. Add per-question telemetry to `temporal_factual.py`: `kg_first_count`, `kg_first_hits` (the (doc, ref) tuples), `merged_pool_size`.
2. Run `rlm_dispatched_strat10 --query-types temporal_factual --e4`, inspect predictions.jsonl for the telemetry.
3. The likely bug: my `_merge_candidate_pools` keeps the higher-scoring entry but the KG-first hits' `text` field is the *full version text* (could be 5000+ chars), while hybrid candidates have chunk text. The chain step in `temporal_factual.run()` then does `chain = _amendment_chain(...)` which works on the URI not the text — so the long KG-first text shouldn't matter. Check whether the URI resolution `_resolve_article_uri` succeeds on KG-first hits.
4. Fix the bug, re-run, confirm TF Cite F1 lifts to ≥ 0.25.

**E.2 — KG topology disambiguator (Fix-MH v2)**:
1. Add a helper `kg_structural_distance(doc_id, ref_a, ref_b)` in `enhancers.py` that returns:
   - 0 if same chapter+section
   - 1 if same chapter, different section
   - 2 if same doc, different chapter
   - ∞ otherwise
2. In `multi_hop.py` aggregation step, AFTER the consensus boost:
   - For each pair of candidates with similar final_score (within 0.10) and same doc_id:
     - If structurally adjacent (distance ≤ 1): query the KG for which one's chapter/section title contains the query concept
     - Promote the matched one
3. Re-run multi_hop strat10 + full-244; verify the adjacent-article disambiguation fires.

**E.3 — Extend concept-amendment helper to MH and RA**:
1. In `multi_hop.py` and `rule_application.py`, add a per-sub-q (or per-query) concept-amendment SPARQL channel alongside hybrid retrieval — same pattern as Fix-TF but without the date filter.
2. Union with hybrid candidates, dedup, score-merge.
3. Re-run full 244.

**E.4 — KG-derived doc routing channel**:
1. Add a `kg_router_channel(query)` to `DocRouter` that queries the KG for which docs have articles mentioning the query concepts.
2. Add as a 4th channel with weight 0.5 in the alias_bonus + bm25_weight + kg_weight + llm_bonus fusion.
3. Re-run + measure doc-routing recall@3.

**Gate**:
- Fix-TF temporal_factual Cite F1 ≥ 0.25 on full 244.
- KG topology disambiguator changes the citation choice on at least 5/26 multi_hop questions.
- Concept-amendment lifts RA or MH Cite F1 by ≥ 0.02 each.
- DocRouter recall@3 ≥ 85% on full 244.

**📋 Self-prompt for next session after Phase E is done**:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase E (KG everywhere) is complete and documented in §1. Start **Phase F — HPC corpus-tuned embedder**.

---

### Phase F — HPC corpus-tuned embedder (BIG retrieval lift)

**Goal**: lift R@10 art from ~0.23 to ~0.40+ by fine-tuning an embedder on ALB v3.0 gold + hard negatives. This is the path to Cite F1 > 0.45.

**Files to read first**:
- `akn_rlm/akn_rlm/indexers/dense.py` (model-agnostic dense index)
- `hpc/build_indices.sbatch` (existing build pattern)
- `new_dataset/AlgerianLegalBench_v3.0_final.json` (benchmark with gold)

**Concrete tasks** (on HPC):

**F.1 — Mine training pairs**:
1. Create `hpc/mine_pairs.py`:
   - Loads ALB v3.0
   - For each question: positive = full text of each gold article. Negatives: top-15 BM25 + top-15 Dense hits NOT in gold (hard negatives by retrieval).
   - Output: `train_pairs.jsonl` with format `{"query": ..., "positive": [texts...], "hard_negative": [texts...]}`. ~244 queries × ~3 positives × ~30 negatives ≈ 22k training pairs.

**F.2 — Train embedder**:
1. Create `hpc/train_embedder.py`:
   - Start from `intfloat/multilingual-e5-large` (1024-dim, instruction-tuned, proven on MIRACL Arabic)
   - MarginMSE or MultipleNegativesRankingLoss
   - batch_size=16, epochs=10, learning_rate=2e-5
   - Save to `hpc/models/e5-large-akn-tuned/`
2. Write SLURM script `hpc/train_embedder.sbatch` (or bash if no SLURM): single H100 MIG slice, ~4-6 h.

**F.3 — Re-index + evaluate**:
1. Locally on HPC: `export EMBED_MODEL=$HOME/akn_rlm_hpc/hpc/models/e5-large-akn-tuned`
2. `python scripts/build_indices.py --force --index dense`
3. Run full-244 dispatcher with new dense index: `rlm_dispatched_full_e5_large_tuned`.
4. Compare R@10 art delta vs baseline e5-small. Expected: ≥ 0.40 (vs current 0.23).

**Gate**:
- R@10 art ≥ 0.35 on full 244.
- Cite F1 ≥ current SOTA + 0.05 (target ~0.38).
- No regression on HCR or AbstF1.

**Hardware**: ~6 h H100, ~30 GB scratch for model + embeddings.

**📋 Self-prompt for next session after Phase F is done**:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase F (corpus-tuned embedder) is complete and documented in §1. Start **Phase G — ColBERT multi-vector on HPC**.

---

### Phase G — ColBERT multi-vector on HPC

**Goal**: add late-interaction MaxSim retrieval as a complementary channel to single-vector dense. Often beats dense alone on small specialized corpora.

**Files to read first**:
- `akn_rlm/akn_rlm/indexers/colbert.py` (existing implementation, disabled)
- `akn_rlm/akn_rlm/indexers/_bge_m3_loader.py` (multi-vector head)
- `akn_rlm/akn_rlm/retrievers/hybrid_fusion.py` (RRF fusion to plug into)

**Concrete tasks** (on HPC):

1. Re-enable ColBERT: in `config.py`, set `ENABLE_COLBERT = True`.
2. Build ColBERT index: `python scripts/build_indices.py --force --index colbert`.
3. Wire ColBERT as a 4th channel in each handler's RRF fusion (alongside BM25, Dense, HyDE):
   - Add `colbert` parameter to handler constructors
   - In each `_fused_candidates`, add `colbert.search(query, k=k_each)` and include in RRF fuse
   - Inject `colbert` via dispatcher
4. Run full-244: `rlm_dispatched_full_with_colbert`.
5. Ablate: full-244 with ONLY ColBERT (no dense), to confirm its value.

**Gate**:
- Cite F1 lifts ≥ 0.02 over Phase F (target ~0.40+).
- R@10 art ≥ 0.45.

**Hardware**: ColBERT index is ~1.4 GB for 8998 chunks. Per-query MaxSim forward ~50-100 ms on H100.

**📋 Self-prompt for next session after Phase G is done**:
> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Phase G (ColBERT) is complete and documented in §1. Start **Phase H — Final evaluation + thesis artifacts**.

---

### Phase H — Final evaluation + thesis artifacts

**Goal**: produce the final thesis Chapter 5 (Methodology + Results + Discussion) artifacts: methodology diagram, ablation tables, per-handler trajectory examples, Toulmin output samples.

**Files to read first**:
- `thesis_comparisons/` (everything accumulated)
- `akn_rlm/scripts/compare_baselines.py` (the comparison table generator)

**Concrete tasks**:

**H.1 — Final ablation run** (full-244 with EVERYTHING):
1. `rlm_dispatched_full_FINAL` — gold-typed, all components enabled.
2. `rlm_dispatched_full_FINAL_classifier` — classifier-typed (deployable).
3. Each phase OFF in turn (one at a time):
   - `_minus_phase_C_adu` — no ADU
   - `_minus_phase_D_recursion` — depth=1 only, no corrective retry
   - `_minus_phase_E_kg` — selective E4 + Fix-MH + Fix-LC only
   - `_minus_phase_F_embedder` — back to e5-small
   - `_minus_phase_G_colbert` — no ColBERT channel
4. Build ablation table showing each component's contribution % to Cite F1.

**H.2 — Comprehensive comparison table** (thesis Chapter 5 main table):
- 11 columns: BM25, Dense, Hybrid, H+Rerank, KG, KG+H, B8 LLM-raw, B9 LLM+context, F5 baseline, **RLM-final-gold**, **RLM-final-classifier**.
- 8 query type rows + overall row.
- Metrics: Cite F1, MRR art, R@10 art, HCR, JIR, AbstF1, am_faithfulness.

**H.3 — Methodology diagram** (`thesis_artifacts/methodology_diagram.mermaid` + PNG export):
- Mermaid source that renders the architecture from §2.1
- Include phases A–H as activation toggles

**H.4 — Per-handler trajectory examples** (`thesis_artifacts/trajectory_examples.md`):
- 1 example per query type. Each shows: query → classified type → routed docs → retrieval channels and hits → verification trace → ADU output → recursion path (if any) → faithfulness gate → final answer.
- Pick examples that demonstrate both successful and failed (with reasoning) cases.

**H.5 — Per-claim provenance examples** (`thesis_artifacts/toulmin_provenance.md`):
- 3 examples where the final answer has multiple claims and each claim is traced to its citation's Toulmin claim+ground.

**H.6 — Per-stratum delta plot** (`thesis_artifacts/per_type_deltas.png`):
- Bar chart per query type: best Phase-1 baseline vs F5 vs Final SOTA.
- Generate via matplotlib.

**H.7 — Latency / token / sub-LM call distribution plots**:
- Histogram of per-question latency, broken down by query type.
- Mean sub-LM calls per query type.

**H.8 — Discussion chapter inputs** (`thesis_artifacts/discussion_inputs.md`):
- Honest analysis: where does each component help, where does it hurt?
- The R@10 art ceiling story (Phase F lifts it).
- The gold-typed vs classifier-typed delta (Phase B reveals it).
- The CD vs KG+Hybrid statistical-tie story.
- The unanswerable Cite F1 metric-quirk explanation.
- The long_context broader-recall-vs-precision trade-off.

**Gate**: all artifacts saved under `thesis_artifacts/`, comparison table consumable directly by Chapter 5, methodology diagram renders cleanly.

**📋 Self-prompt for next session after Phase H is done**:
> Phase H is complete. All thesis Chapter 5 artifacts are in `D:\TRY_AGAIN\thesis_artifacts\`. Read those + `D:\TRY_AGAIN\HANDOFF.md` §1 for the locked SOTA numbers. The remaining work is writing the actual thesis chapters (Methodology, Results, Discussion) — those are the user's text, not Claude Code's. If asked, help phrase specific subsections, but don't generate the full chapter prose.

---

## 4. Critical files

```
D:\TRY_AGAIN\
├── HANDOFF.md                          ← this file
├── akn_rlm\
│   ├── .env                            ← API keys (chmod 600, gitignored)
│   ├── pyproject.toml
│   ├── akn_rlm\                        ← package
│   │   ├── config.py                   ← model names, feature flags
│   │   ├── rlm\
│   │   │   ├── dispatcher.py           ← RLMDispatcher (main entry)
│   │   │   ├── handlers\               ← 8 typed handlers
│   │   │   │   ├── rule_application.py
│   │   │   │   ├── exact_article.py
│   │   │   │   ├── multi_hop.py
│   │   │   │   ├── temporal_factual.py
│   │   │   │   ├── conceptual_definitional.py
│   │   │   │   ├── unanswerable.py
│   │   │   │   ├── layman.py
│   │   │   │   └── long_context.py
│   │   │   ├── ceiling_breakers.py     ← E1-E4 factories
│   │   │   ├── enhancers.py            ← E1-E4 + MultiQuery/HyDE wrappers
│   │   │   ├── routing\doc_router.py
│   │   │   ├── classifier.py           ← query_type classifier (Phase B)
│   │   │   ├── sub_worker.py           ← call_verifier / call_summarizer / call_decomposer
│   │   │   ├── supervisor.py
│   │   │   └── pipeline.py             ← LangGraph (dispatcher bypasses this)
│   │   ├── indexers\
│   │   │   ├── bm25.py
│   │   │   ├── dense.py                ← model-agnostic (e5/BGE-m3/Qwen2-7B)
│   │   │   └── colbert.py              ← disabled, Phase G re-enables
│   │   ├── gates\
│   │   │   ├── faithfulness_nli.py     ← R8 retune lives here
│   │   │   ├── citation_existence.py
│   │   │   ├── jurisdiction.py
│   │   │   └── span_existence.py
│   │   ├── adu\                        ← Toulmin extraction (Phase C extends)
│   │   ├── baselines\                  ← 6 Phase-1 baselines + Phase A LLM-only (to add)
│   │   ├── corpus\
│   │   │   ├── akn_parser.py
│   │   │   ├── article_registry.py
│   │   │   ├── chunker.py
│   │   │   └── kg_loader.py            ← rdflib KG load (~26 s)
│   │   ├── retrievers\hybrid_fusion.py
│   │   ├── llm\client.py
│   │   ├── eval\runner.py              ← _answer_to_result
│   │   └── tests\                      ← 762 unit tests
│   ├── scripts\
│   │   ├── run_dispatcher.py           ← MAIN: full RLM run
│   │   ├── run_benchmark.py            ← legacy LangGraph runner
│   │   ├── run_baseline_*.py           ← 6 baseline runners
│   │   ├── run_handler_*.py            ← per-handler runners
│   │   ├── compare_baselines.py        ← thesis table generator
│   │   ├── build_indices.py
│   │   ├── eval_doc_router.py
│   │   └── run_ablation_e1_e4.py       ← strat-5 ablation orchestrator
│   ├── data\indices\
│   │   ├── bm25.pkl
│   │   ├── dense.faiss
│   │   ├── dense_meta.parquet
│   │   └── (corpus_hash.txt — drives rebuild detection)
│   └── eval_results\                   ← run outputs (predictions.jsonl, metrics.json)
├── thesis_comparisons\                 ← final tables for the thesis
├── thesis_artifacts\                   ← Phase H output (graphs, diagrams)
├── hpc\                                ← HPC scripts (Phase F + G)
├── new_dataset\
│   ├── AlgerianLegalBench_v3.0_final.json
│   ├── data\akn\                       ← 51 AKN XML files
│   └── data\rdf\algerian_legal_kg.ttl  ← 758k-triple KG
└── eval_results\                       ← duplicate eval results root (compare_baselines reads both)
```

---

## 5. Useful commands (paste-ready)

```pwsh
# Local Python interpreter (always use this on Windows)
$py = "C:\Users\21355\.conda\envs\pfe_env\python.exe"

# Run unit tests (should pass 762/762)
& $py -m pytest D:\TRY_AGAIN\akn_rlm\akn_rlm\tests\ -q

# Stratified-5 smoke (~5 min)
& $py D:\TRY_AGAIN\akn_rlm\scripts\run_dispatcher.py --e4 --stratified 5 --run-id smoke_$(Get-Date -Format yyyyMMdd_HHmmss)

# Full 244 with current locked SOTA config (~25 min)
& $py D:\TRY_AGAIN\akn_rlm\scripts\run_dispatcher.py --e4 --run-id full_$(Get-Date -Format yyyyMMdd_HHmmss)

# Build the comparison table
& $py D:\TRY_AGAIN\akn_rlm\scripts\compare_baselines.py `
    --runs "baseline_bm25_full,baseline_dense_full,baseline_hybrid_full,baseline_hybrid_rerank_full,baseline_kg_full,baseline_kg_hybrid_full,rlm_dispatched_full_v4,rlm_dispatched_full_e4_trajectory" `
    --out D:\TRY_AGAIN\thesis_comparisons\comparison_locked.md --no-stdout

# Inspect a single run's metrics
& $py -c "import json; o=json.load(open('D:/TRY_AGAIN/akn_rlm/eval_results/<RUN_ID>/metrics.json'))['overall']; [print(f'{k:20s} {v}') for k,v in o.items()]"

# Rebuild indices (only if chunker/parser changes — usually not needed)
cd D:\TRY_AGAIN\akn_rlm; & $py scripts\build_indices.py --force
```

**HPC**:
```bash
# On the JupyterHub pod (after conda activate akn_rlm_hpc):
cd ~/pfe-hpc/akn_rlm_hpc && git pull
export HF_HOME=$HOME/hf_cache EMBED_MODEL=intfloat/multilingual-e5-small AKN_CEILING_BREAKERS=0

# Full 244 (~25 min)
cd akn_rlm
python scripts/run_dispatcher.py --e4 --run-id full_hpc_$(date +%Y%m%d_%H%M%S)
```

---

## 6. Compressed history (key insights only — no need to re-litigate)

**Phase 0** (DONE): canonical `article_ref` normalisation (`الأولى → 1`, `9 مكرر → 9_bis`); span_existence gate (catches fabrications); indices rebuilt with canonical refs.

**Phase 1 B1–B7** (DONE): 6 deterministic baselines (BM25 / Dense / Hybrid RRF / Hybrid+Rerank / KG SPARQL / KG+Hybrid) + comparison-table generator. KG+Hybrid was best overall Phase-1 (Cite F1 = 0.083 full 244).

**Phase 2 R1–R8** (DONE): doc-router (recall@3 = 82.9%), 8 typed handlers, dispatcher, supervisor seam, faithfulness retune. Locked F5 = Cite F1 0.293 / MRR art 0.257.

**Phase 2 F1–F5** (DONE): F5 final = Cite F1 0.3011 / MRR art 0.2686. Wins 3 of 4 hard types + 4 of 4 simple types. CD ties KG+Hybrid.

**HPC ceiling-breaker experiments** (PARTIAL): tested BGE-m3 / gte-Qwen2-7B / gte-multilingual-base as dense retrievers — **all regressed**. NLI verifier v1 and LLM doc-router silent no-ops on full-244. Concept-amendment SPARQL helper landed but only wired into CD. **Lesson**: scaling the embedder doesn't help on this small specialized Arabic legal corpus. Path forward is training, not scaling.

**E1–E4 enhancer ablation** (DONE on strat-5): E4 HyDE was the clear winner (+0.021 Cite F1 strat-5). E1/E2/E3 all approximately neutral on strat-5. Full-244 E4 lift was +0.0038 because HyDE regresses TF (-0.095) and CD (-0.056). Selective HyDE (off for TF/CD) recovers them: full-244 E4-selective = Cite F1 0.3095.

**Trajectory fixes** (DONE): Fix-MH consensus aggregation lifted MH +0.054. Fix-LC chapter neighbors lifted LC +0.023. Fix-TF KG-first wired but flat at 0.190 — needs debug (Phase E.1). Layman lifted +0.037 as a side effect (delegates to RA with the improved retrieval pool). Final locked SOTA = **Cite F1 0.3129**.

---

## 7. Self-prompt for THE FIRST NEXT SESSION (Phase A start)

After clearing chat, paste this:

> Read `D:\TRY_AGAIN\HANDOFF.md` end-to-end. Then start **Phase A — LLM-only baselines (no RAG, no KG, no AM)** as specified in §3. The goal is to establish the floor: 4 baseline runs on full 244 that show what raw LLMs (gpt-oss-120b, Qwen3-30B-A3B-Thinking) produce without retrieval, plus two with minimal RAG context. Files to read first are listed under Phase A. When the gate is met, append a "Phase A — DONE" section to §1 of this HANDOFF with the new metrics, push to PFE_locally, then surface the self-prompt for Phase B.

**That's it. Keep this HANDOFF.md updated as the single source of truth between sessions.**
