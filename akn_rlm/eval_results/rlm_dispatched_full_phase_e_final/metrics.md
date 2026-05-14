# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8238 |
| abstention_f1 | 0.7034 |
| abstention_precision | 0.6538 |
| abstention_recall | 0.7612 |
| am_faithfulness_score | 0.4709 |
| answer_faithfulness | 0.3197 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3045 |
| citation_groundedness | 0.5527 |
| citation_precision | 0.1890 |
| citation_recall | 0.2596 |
| corrective_retry_rate | 0.1926 |
| doc_citation_f1 | 0.6130 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2120 |
| map_doc | 0.5441 |
| mean_latency_s | 9.7421 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.3101 |
| mrr_doc | 0.5792 |
| ndcg_article | 0.3252 |
| ndcg_doc | 0.5906 |
| precision_article | 0.0475 |
| precision_doc | 0.0643 |
| reasoning_chain_score | 0.0081 |
| recall_article | 0.2583 |
| recall_doc | 0.5867 |
| rouge_l | 0.0991 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1279 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5833 | 0.1903 | 0.1667 | 0.3333 | 0.4000 |
| exact_article | 0.6864 | 0.5028 | 0.4020 | 0.5000 | 0.4444 |
| layman | 0.5294 | 0.2059 | 0.2549 | 1.0000 | 0.4444 |
| long_context | 0.7549 | 0.3049 | 0.1290 | 0.0000 | 0.0000 |
| multi_hop | 0.6538 | 0.1859 | 0.1328 | 0.3333 | 0.2857 |
| rule_application | 0.7348 | 0.4273 | 0.2615 | 0.4286 | 0.3333 |
| temporal_factual | 0.7143 | 0.2857 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0375 | 0.0000 | 0.5000 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6339 | 0.4137 | 0.3212 | 0.5000 | 0.4286 |
| hard | 0.4450 | 0.1720 | 0.2583 | 0.9048 | 0.8941 |
| medium | 0.7059 | 0.4092 | 0.3494 | 0.5385 | 0.4375 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6006 | 0.3261 | 0.3159 | 0.7500 | 0.7273 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
