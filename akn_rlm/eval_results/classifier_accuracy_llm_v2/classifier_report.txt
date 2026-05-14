# Classifier accuracy report

- Classifier: **LLM (google/gemma-4-31B)**
- Questions evaluated: **244**
- Overall accuracy: **0.6639** (162/244)
- Macro F1: **0.7186**
- Weighted F1: **0.6729**

## Per-class metrics

| Query type | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| rule_application | 66 | 0.771 | 0.561 | 0.649 |
| exact_article | 59 | 0.606 | 0.339 | 0.435 |
| multi_hop | 26 | 0.524 | 0.846 | 0.647 |
| unanswerable | 40 | 0.974 | 0.925 | 0.949 |
| layman | 17 | 1.000 | 0.882 | 0.938 |
| long_context | 17 | 0.867 | 0.765 | 0.812 |
| conceptual_definitional | 12 | 0.244 | 0.917 | 0.386 |
| temporal_factual | 7 | 0.875 | 1.000 | 0.933 |

## Confusion matrix (rows = gold, cols = predicted)

| gold \ pred | RA | EA | MH | UA | Lay | LC | CD | TF | total |
|---|---|---|---|---|---|---|---|---|---|
| **RA** | 37 | 8 | 11 | 0 | 0 | 2 | 8 | 0 | 66 |
| **EA** | 9 | 20 | 5 | 0 | 0 | 0 | 25 | 0 | 59 |
| **MH** | 2 | 1 | 22 | 0 | 0 | 0 | 0 | 1 | 26 |
| **UA** | 0 | 2 | 0 | 37 | 0 | 0 | 1 | 0 | 40 |
| **Lay** | 0 | 2 | 0 | 0 | 15 | 0 | 0 | 0 | 17 |
| **LC** | 0 | 0 | 4 | 0 | 0 | 13 | 0 | 0 | 17 |
| **CD** | 0 | 0 | 0 | 1 | 0 | 0 | 11 | 0 | 12 |
| **TF** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| **total** | 48 | 33 | 42 | 38 | 15 | 15 | 45 | 8 | 244 |

Short codes: RA=rule_application, EA=exact_article, MH=multi_hop, UA=unanswerable, Lay=layman, LC=long_context, CD=conceptual_definitional, TF=temporal_factual.
