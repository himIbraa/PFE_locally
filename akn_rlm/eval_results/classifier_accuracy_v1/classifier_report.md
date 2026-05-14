# Classifier accuracy report

- Questions evaluated: **244**
- Overall accuracy: **0.2992** (73/244)
- Macro F1: **0.1323**
- Weighted F1: **0.1856**

## Per-class metrics

| Query type | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| rule_application | 66 | 0.284 | 0.955 | 0.437 |
| exact_article | 59 | 0.500 | 0.068 | 0.119 |
| multi_hop | 26 | 0.750 | 0.115 | 0.200 |
| unanswerable | 40 | 0.000 | 0.000 | 0.000 |
| layman | 17 | 0.000 | 0.000 | 0.000 |
| long_context | 17 | 1.000 | 0.059 | 0.111 |
| conceptual_definitional | 12 | 0.222 | 0.167 | 0.190 |
| temporal_factual | 7 | 0.000 | 0.000 | 0.000 |

## Confusion matrix (rows = gold, cols = predicted)

| gold \ pred | RA | EA | MH | UA | Lay | LC | CD | TF | total |
|---|---|---|---|---|---|---|---|---|---|
| **RA** | 63 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 66 |
| **EA** | 50 | 4 | 1 | 0 | 0 | 0 | 4 | 0 | 59 |
| **MH** | 21 | 2 | 3 | 0 | 0 | 0 | 0 | 0 | 26 |
| **UA** | 38 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 40 |
| **Lay** | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 |
| **LC** | 16 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 17 |
| **CD** | 10 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 12 |
| **TF** | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| **total** | 222 | 8 | 4 | 0 | 0 | 1 | 9 | 0 | 244 |

Short codes: RA=rule_application, EA=exact_article, MH=multi_hop, UA=unanswerable, Lay=layman, LC=long_context, CD=conceptual_definitional, TF=temporal_factual.
