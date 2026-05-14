# Classifier accuracy report

- Classifier: **LLM (google/gemma-4-31B)**
- Questions evaluated: **10**
- Overall accuracy: **0.7000** (7/10)
- Macro F1: **0.1779**
- Weighted F1: **0.7962**

## Per-class metrics

| Query type | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| rule_application | 7 | 1.000 | 0.857 | 0.923 |
| exact_article | 3 | 1.000 | 0.333 | 0.500 |
| multi_hop | 0 | 0.000 | 0.000 | 0.000 |
| unanswerable | 0 | 0.000 | 0.000 | 0.000 |
| layman | 0 | 0.000 | 0.000 | 0.000 |
| long_context | 0 | 0.000 | 0.000 | 0.000 |
| conceptual_definitional | 0 | 0.000 | 0.000 | 0.000 |
| temporal_factual | 0 | 0.000 | 0.000 | 0.000 |

## Confusion matrix (rows = gold, cols = predicted)

| gold \ pred | RA | EA | MH | UA | Lay | LC | CD | TF | total |
|---|---|---|---|---|---|---|---|---|---|
| **RA** | 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 7 |
| **EA** | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 3 |
| **MH** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **UA** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Lay** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **LC** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **CD** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TF** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | 6 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 10 |

Short codes: RA=rule_application, EA=exact_article, MH=multi_hop, UA=unanswerable, Lay=layman, LC=long_context, CD=conceptual_definitional, TF=temporal_factual.
