# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8033 |
| abstention_f1 | 0.6757 |
| abstention_precision | 0.6173 |
| abstention_recall | 0.7463 |
| am_faithfulness_score | 0.4619 |
| answer_faithfulness | 0.3320 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2972 |
| citation_groundedness | 0.5322 |
| citation_precision | 0.1966 |
| citation_recall | 0.2209 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6150 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.1878 |
| map_doc | 0.5143 |
| mean_latency_s | 5.8405 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2670 |
| mrr_doc | 0.5485 |
| ndcg_article | 0.2784 |
| ndcg_doc | 0.5611 |
| precision_article | 0.0369 |
| precision_doc | 0.0623 |
| reasoning_chain_score | 0.0054 |
| recall_article | 0.2160 |
| recall_doc | 0.5587 |
| rouge_l | 0.0937 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1191 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.1208 | 0.1071 | 0.0000 | 0.0000 |
| exact_article | 0.6186 | 0.4548 | 0.4277 | 0.4167 | 0.3571 |
| layman | 0.5294 | 0.1765 | 0.2059 | 0.5000 | 0.2500 |
| long_context | 0.8333 | 0.1490 | 0.0925 | 0.0000 | 0.0000 |
| multi_hop | 0.6859 | 0.1474 | 0.0958 | 0.3333 | 0.2857 |
| rule_application | 0.6869 | 0.3939 | 0.2439 | 0.4286 | 0.2727 |
| temporal_factual | 0.7143 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5804 | 0.3720 | 0.3405 | 0.3333 | 0.2857 |
| hard | 0.4547 | 0.1346 | 0.2581 | 0.9524 | 0.9091 |
| medium | 0.6412 | 0.3582 | 0.3160 | 0.4615 | 0.3750 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5682 | 0.2808 | 0.3082 | 0.7344 | 0.6963 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
