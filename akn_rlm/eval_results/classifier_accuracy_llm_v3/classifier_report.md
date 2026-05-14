# Classifier accuracy report

- Classifier: **LLM (google/gemma-4-31B)**
- Questions evaluated: **244**
- Overall accuracy: **0.6270** (153/244)
- Macro F1: **0.6899**
- Weighted F1: **0.6276**

## Per-class metrics

| Query type | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| rule_application | 66 | 0.660 | 0.470 | 0.549 |
| exact_article | 59 | 0.652 | 0.254 | 0.366 |
| multi_hop | 26 | 0.524 | 0.846 | 0.647 |
| unanswerable | 40 | 1.000 | 0.950 | 0.974 |
| layman | 17 | 1.000 | 0.882 | 0.938 |
| long_context | 17 | 0.722 | 0.765 | 0.743 |
| conceptual_definitional | 12 | 0.226 | 1.000 | 0.369 |
| temporal_factual | 7 | 0.875 | 1.000 | 0.933 |

## Confusion matrix (rows = gold, cols = predicted)

| gold \ pred | RA | EA | MH | UA | Lay | LC | CD | TF | total |
|---|---|---|---|---|---|---|---|---|---|
| **RA** | 31 | 4 | 11 | 0 | 0 | 4 | 16 | 0 | 66 |
| **EA** | 13 | 15 | 5 | 0 | 0 | 1 | 25 | 0 | 59 |
| **MH** | 3 | 0 | 22 | 0 | 0 | 0 | 0 | 1 | 26 |
| **UA** | 0 | 2 | 0 | 38 | 0 | 0 | 0 | 0 | 40 |
| **Lay** | 0 | 2 | 0 | 0 | 15 | 0 | 0 | 0 | 17 |
| **LC** | 0 | 0 | 4 | 0 | 0 | 13 | 0 | 0 | 17 |
| **CD** | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 12 |
| **TF** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| **total** | 47 | 23 | 42 | 38 | 15 | 18 | 53 | 8 | 244 |

Short codes: RA=rule_application, EA=exact_article, MH=multi_hop, UA=unanswerable, Lay=layman, LC=long_context, CD=conceptual_definitional, TF=temporal_factual.
