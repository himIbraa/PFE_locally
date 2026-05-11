# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8238 |
| abstention_f1 | 0.7075 |
| abstention_precision | 0.6500 |
| abstention_recall | 0.7761 |
| answer_faithfulness | 0.3279 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3095 |
| citation_groundedness | 0.5480 |
| citation_precision | 0.2048 |
| citation_recall | 0.2293 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6096 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.1999 |
| map_doc | 0.5137 |
| mean_latency_s | 5.1916 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2856 |
| mrr_doc | 0.5533 |
| ndcg_article | 0.2960 |
| ndcg_doc | 0.5640 |
| precision_article | 0.0381 |
| precision_doc | 0.0619 |
| reasoning_chain_score | 0.0057 |
| recall_article | 0.2293 |
| recall_doc | 0.5526 |
| rouge_l | 0.0939 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1209 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5417 | 0.1625 | 0.1349 | 0.0000 | 0.0000 |
| exact_article | 0.6017 | 0.4548 | 0.4282 | 0.5000 | 0.4138 |
| layman | 0.4706 | 0.1961 | 0.2471 | 0.5000 | 0.2500 |
| long_context | 0.8235 | 0.2745 | 0.0970 | 0.0000 | 0.0000 |
| multi_hop | 0.8077 | 0.2555 | 0.1636 | 0.6667 | 0.6667 |
| rule_application | 0.6742 | 0.3750 | 0.2453 | 0.4286 | 0.2857 |
| temporal_factual | 0.7857 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5536 | 0.3780 | 0.3512 | 0.4167 | 0.3448 |
| hard | 0.4806 | 0.1705 | 0.2677 | 0.9762 | 0.9425 |
| medium | 0.6412 | 0.3641 | 0.3327 | 0.4615 | 0.3871 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5733 | 0.3003 | 0.3212 | 0.7656 | 0.7259 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.5000 |
