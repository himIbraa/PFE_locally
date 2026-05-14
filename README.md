# AKN-RLM — Recursive Language Model for Algerian Legal QA

End-to-end legal QA pipeline over **AlgerianLegalBench v3.0** (244 questions,
8 query types). Integrates Retrieval-Augmented Generation, a 758 k-triple
Knowledge Graph, and Toulmin Argument Mining behind a single typed-handler
dispatcher.

**Locked SOTA (deployable, classifier-typed dispatch)**:
**Cite F1 = 0.3045**, HCR = 0.000, am_faithfulness_score = 0.471, mean
latency 9.7 s/q. Run: `rlm_dispatched_full_phase_e_final` — see
[`HANDOFF.md`](HANDOFF.md) §1.4f Task #6.

Beats the strongest deterministic Phase-1 baseline (Hybrid + Rerank,
Cite F1 = 0.105) by **2.9×**.

The deployable pipeline at a glance:

```
query (AR / FR / Darja)
  ↓  Gemma-4-31B classifier  →  query_type
  ↓  doc-router (alias + numeric + BM25)  →  top-3 doc_ids
  ↓  typed handler  (1 of 8)
        ├── RRF(BM25, Dense, HyDE-Qwen)         retrieval
        ├── sub-LM verifier  (Qwen3)            relevance
        ├── KG amendment chain (TF / CD)        version anchoring
        ├── pervasive Toulmin ADU               claim / ground / warrant
        ├── gap-driven recursion (depth 1-3)    "Recursive" mechanism
        └── corrective retry on faithfulness    HCR ≤ 0.02 contract
  ↓  AnswerResponse{ answer_text, citations[], references[], trajectory[] }
```

---

## Quickstart

```pwsh
# Windows
$py = "C:\Users\21355\.conda\envs\pfe_env\python.exe"

# One-shot query
& $py D:\TRY_AGAIN\akn_rlm\scripts\answer_query.py `
    --query "ما هي شروط الزواج في قانون الأسرة الجزائري؟"

# Interactive REPL
& $py D:\TRY_AGAIN\akn_rlm\scripts\answer_query.py --repl

# JSON output (pipe into a UI or curl)
& $py D:\TRY_AGAIN\akn_rlm\scripts\answer_query.py `
    --query "..." --json
```

Prerequisites:

- Python env: `pfe_env` (Windows) or `akn_rlm_hpc` (HPC).
- Indices built once: `python scripts/build_indices.py` (creates
  `data/indices/bm25.pkl` + `data/indices/dense.faiss`).
- LLM credentials in `akn_rlm/.env` (gitignored — copy
  `.env.example` and fill in).

---

## `answer_query` — the UI endpoint

`akn_rlm.api.answer.answer_query(query: str) -> AnswerResponse` is the
**single entry point** the UI calls. It hides the lazy dispatcher
construction, pipeline execution, and post-processing behind one
function.

```python
from akn_rlm.api import answer_query

response = answer_query("ما هي شروط الزواج في قانون الأسرة الجزائري؟")

print(response.answer_text)
for c in response.citations:
    print(f"[{c.doc_title} art.{c.article_ref}] {c.supporting_span}")
for step in response.trajectory:
    print(f"  → {step.summary}")
```

### What it does, step by step

1. **Lazy initialisation (first call only)**: parses the AKN corpus,
   loads the BM25 + Dense indices, connects to the LLM pool, builds
   the doc-router, and wires the Phase E dispatcher with the
   highest-performing config (selective HyDE, pervasive ADU,
   recursion + corrective retry, `recursion_coverage_min=4` for
   MH/RA). Cost: ~5-10 s. Subsequent calls reuse the singleton.

2. **Classify + route**: a Gemma-4-31B prompt predicts the query
   type (`exact_article` / `rule_application` / `multi_hop` /
   `temporal_factual` / `conceptual_definitional` / `unanswerable`
   / `layman` / `long_context`). The doc-router predicts up to 3
   relevant `doc_id`s.

3. **Dispatch to the typed handler**. Each of the 8 handlers shares
   the same template (retrieval → verification → recursion →
   summarisation) but specialises one stage (e.g. TF anchors on a
   KG amendment chain at a specific date; CD does Toulmin ADU
   pre-citation-construction; MH decomposes into sub-questions).

4. **Gap-driven recursion (Phase D)** wraps the handler's
   retrieve+verify loop. After depth 1, a gpt-oss-120b "gap probe"
   decides whether the accumulator covers the question; if not, it
   emits a sub-question and the handler additively re-retrieves at
   depth 2 (and optionally 3). Depth markers are recorded in the
   trajectory output.

5. **Pervasive Toulmin ADU (Phase C)** extracts
   `claim / ground / warrant / rebuttal / backing` on every cited
   article (top-5) before answer assembly. The citation's
   `supporting_span` becomes `claim + ground` so the synthesised
   answer reads from a tighter premise.

6. **Synthesise → faithfulness gate → corrective retry**. The
   summariser produces an Arabic answer over the surviving
   citations; the per-citation NLI gate scores claim/ground
   entailment; on gate failure, the summariser re-runs ONCE with
   "use only cited articles" feedback.

7. **Wrap into `AnswerResponse`** for the UI.

### Response shape

```python
@dataclass
class AnswerResponse:
    query: str                          # the original query
    query_type_predicted: str           # classifier output
    handler_used: str                   # dispatched handler key
    answer_text: str                    # synthesised Arabic answer
    citations: list[Citation]           # see below
    references: list[str]               # markdown-ready "[N] doc, art X"
    trajectory: list[TrajectoryStep]    # explainability spine
    abstained: bool                     # True for the unanswerable path
    abstention_reason: str | None
    latency_s: float                    # wall-clock for this call
    sub_call_count: int                 # sub-LM calls billed
    recursion_depth_max: int            # 1 / 2 / 3
    corrective_retry_fired: bool

@dataclass
class Citation:
    doc_id: str                         # canonical "84-11_1984-06-09"
    article_ref: str                    # canonical "9_bis"
    doc_title: str                      # "قانون الأسرة، المعدل والمتمم"
    supporting_span: str                # claim + ground
    text: str                           # full article text
    confidence: float                   # verifier confidence
    version_date: str | None            # "2005-02-27" for TF/MH
    kg_source: str | None               # "kg" / "fallback"
    argumentation: dict | None          # Toulmin block
    verifier_relevant: bool | None

@dataclass
class TrajectoryStep:
    step: str                           # raw step name
    depth: int                          # 0 / 1 / 2 / 3
    summary: str                        # Arabic human-readable sentence
    detail: dict                        # raw step payload
```

The `to_dict()` method on each dataclass returns a JSON-serialisable
view, so wiring to FastAPI / Flask is one line.

### Example output (multi-hop)

```
QUERY:    كيف تتفاعل أحكام البيع في القانون المدني مع أحكام قانون الأسرة
          في تنظيم تصرفات القاصر؟
TYPE:     multi_hop   HANDLER: multi_hop
LATENCY:  62.77s   SUB-LM CALLS: 19   RECURSION DEPTH: 1   CORRECTIVE RETRY: False

=== EXPLAINABILITY (pipeline trajectory) ===
   1. [route/d0]            توجيه الوثائق: 84-11_1984-06-09, 75-58_1975-09-26, 08-09_2008-02-25
   2. [decompose_sweep/d1]  تجزئة السؤال والاسترجاع لكل جزء: 3 أسئلة فرعية، 3 مرشحاً تم التحقق منه
   3. [recursion/d1]        الاسترجاع التكراري (عمق 1): الاسترجاع الأولي، +3 استشهاد — "..."
   4. [recursion/d2]        الاسترجاع التكراري (عمق 2): حوض رقيق لكن لا توجد فجوة قابلة للاسترجاع، +0
   5. [summarize/d0]        تركيب الإجابة
   6. [faithfulness_gate/d0] بوابة المطابقة الدلالية: نجح من البداية

=== ANSWER ===
يخضع تصرفات القاصر في القانون المدني لرقابة الولي … (full Arabic answer)

=== CITATIONS ===
  [1] قانون الأسرة، المعدل والمتمم, art. 88   conf=1.000
        Toulmin.claim:  يتعين على الولي أن يتصرف في أموال القاصر تصرف الرجل الحريص …
        Toulmin.ground: تصرف الولي في أموال القاصر، وتحديداً في حالات: بيع العقار، …
  [2] قانون الإجراءات المدنية والإدارية, art. 478   conf=0.950
  [3] قانون الإجراءات المدنية والإدارية, art. 469   conf=0.900

=== REFERENCES ===
  [1] قانون الأسرة، المعدل والمتمم، المادة 88
  [2] قانون الإجراءات المدنية والإدارية، المادة 478
  [3] قانون الإجراءات المدنية والإدارية، المادة 469
```

### Wiring to a UI (FastAPI sketch)

```python
from fastapi import FastAPI
from akn_rlm.api import answer_query

app = FastAPI()

@app.post("/answer")
def answer(req: dict):
    response = answer_query(req["query"])
    return response.to_dict()
```

The first POST pays the lazy-init cost; every subsequent POST hits the
warm dispatcher. The KG is loaded on the first TF/CD/MH dispatch
(not at startup) — keep that in mind if you want to pre-warm.

### What's intentionally NOT in the endpoint

The Phase E KG-CONTAINS channels (E.2 `--e5`, E.3 `--e6`, E.4 `--e7`)
ship in the codebase behind feature flags but the endpoint **does not**
enable them. Empirical measurement showed all three add no Cite F1
lift and E.4 alone explodes router latency from 17.5 ms to 7708 ms per
query. They remain available via the `run_dispatcher.py` CLI for the
thesis ablation table. See [`HANDOFF.md`](HANDOFF.md) §1.4f Tasks
#3/#4/#5 and the `project_e2_mechanism_no_fire`,
`project_e3_unigram_flood`, `project_e4_kg_router_no_lift` memory
files for the falsified-config record.

### Tests

```pwsh
& $py -m pytest D:\TRY_AGAIN\akn_rlm\akn_rlm\tests\ -q
# 893 passed
```

22 of those cover the API endpoint (`test_api_answer.py`): contract,
response shape, Toulmin / abstention paths, trajectory summarisation,
reference formatting.

---

## Repository layout

```
D:\TRY_AGAIN\
├── README.md                          ← this file
├── HANDOFF.md                         ← development history (Phases A-H)
├── akn_rlm\
│   ├── .env                           ← LLM API keys (gitignored)
│   ├── akn_rlm\                       ← Python package
│   │   ├── api\                       ← UI endpoint (answer_query)
│   │   │   └── answer.py
│   │   ├── rlm\
│   │   │   ├── dispatcher.py          ← typed-handler router
│   │   │   ├── handlers\              ← 8 typed handlers
│   │   │   ├── recursive_refine.py    ← Phase D gap-probe recursion
│   │   │   ├── corrective_retry.py    ← faithfulness retry
│   │   │   ├── adu_helpers.py         ← Phase C pervasive Toulmin ADU
│   │   │   ├── enhancers.py           ← E1-E7 enhancers + feature flags
│   │   │   └── routing\doc_router.py
│   │   ├── indexers\bm25.py / dense.py
│   │   ├── gates\faithfulness_nli.py
│   │   ├── adu\                       ← Toulmin extraction
│   │   ├── corpus\kg_loader.py
│   │   └── tests\                     ← 893 unit tests
│   ├── scripts\
│   │   ├── answer_query.py            ← UI demo CLI
│   │   ├── run_dispatcher.py          ← full benchmark runner
│   │   ├── eval_doc_router.py
│   │   └── build_indices.py
│   ├── data\indices\
│   └── eval_results\
├── new_dataset\                       ← AlgerianLegalBench v3.0 + KG
└── thesis_artifacts\
```

---

## Development history

See [`HANDOFF.md`](HANDOFF.md) for the full Phase A → H plan, locked
metrics tables, falsification log, and per-phase self-prompts. Key
phases:

- **Phase A** — LLM-only baselines (floor: raw Cite F1 ≈ 0.03-0.06).
- **Phase B** — classifier-typed dispatch (Gemma-4-31B; drop −0.015
  vs gold-typed).
- **Phase C** — pervasive Toulmin ADU (+0.013 Cite F1, +0.149
  am_faithfulness).
- **Phase D** — gap-driven recursion + corrective retry (mechanism
  shipped, telemetry rich; HCR=0.000 contract preserved).
- **Phase E** — KG everywhere: 4 KG-CONTAINS channels shipped as
  mechanism-only contributions; covmin=4 override for MH/RA gives
  +0.030 MH Cite F1. The architectural lesson — SPARQL CONTAINS is
  too coarse on Algerian Arabic legal text — motivates Phase F.
- **Phase F** — HPC corpus-tuned embedder (next; expected R@10 art
  0.23 → 0.40+).

---

## Citing this work

Master's thesis: *"Advancing Legal Reasoning in Algerian Law:
Integrating Retrieval-Augmented Generation, Knowledge Graphs, and
Argument Mining for Citation-Faithful Legal QA"* — ENSIA, 2026.
