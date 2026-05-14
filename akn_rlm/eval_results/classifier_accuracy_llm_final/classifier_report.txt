# Classifier accuracy report

- Classifier: **LLM (google/gemma-4-31B)**
- Questions evaluated: **244**
- Overall accuracy: **0.6967** (170/244)
- Macro F1: **0.7232**
- Weighted F1: **0.6943**

## Per-class metrics

| Query type | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| rule_application | 66 | 0.568 | 0.758 | 0.649 |
| exact_article | 59 | 0.650 | 0.441 | 0.525 |
| multi_hop | 26 | 0.842 | 0.615 | 0.711 |
| unanswerable | 40 | 0.974 | 0.950 | 0.962 |
| layman | 17 | 1.000 | 0.882 | 0.938 |
| long_context | 17 | 0.591 | 0.765 | 0.667 |
| conceptual_definitional | 12 | 0.385 | 0.417 | 0.400 |
| temporal_factual | 7 | 0.875 | 1.000 | 0.933 |

## Confusion matrix (rows = gold, cols = predicted)

| gold \ pred | RA | EA | MH | UA | Lay | LC | CD | TF | total |
|---|---|---|---|---|---|---|---|---|---|
| **RA** | 50 | 8 | 2 | 0 | 0 | 3 | 3 | 0 | 66 |
| **EA** | 22 | 26 | 1 | 1 | 0 | 4 | 5 | 0 | 59 |
| **MH** | 8 | 0 | 16 | 0 | 0 | 1 | 0 | 1 | 26 |
| **UA** | 2 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | 40 |
| **Lay** | 0 | 2 | 0 | 0 | 15 | 0 | 0 | 0 | 17 |
| **LC** | 4 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 17 |
| **CD** | 2 | 4 | 0 | 0 | 0 | 1 | 5 | 0 | 12 |
| **TF** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| **total** | 88 | 40 | 19 | 39 | 15 | 22 | 13 | 8 | 244 |

Short codes: RA=rule_application, EA=exact_article, MH=multi_hop, UA=unanswerable, Lay=layman, LC=long_context, CD=conceptual_definitional, TF=temporal_factual.
