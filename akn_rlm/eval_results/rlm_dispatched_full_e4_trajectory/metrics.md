# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8279 |
| abstention_f1 | 0.7162 |
| abstention_precision | 0.6543 |
| abstention_recall | 0.7910 |
| answer_faithfulness | 0.3320 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3129 |
| citation_groundedness | 0.5295 |
| citation_precision | 0.2042 |
| citation_recall | 0.2349 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6126 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.1975 |
| map_doc | 0.5079 |
| mean_latency_s | 5.4257 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2837 |
| mrr_doc | 0.5458 |
| ndcg_article | 0.3000 |
| ndcg_doc | 0.5584 |
| precision_article | 0.0418 |
| precision_doc | 0.0623 |
| reasoning_chain_score | 0.0056 |
| recall_article | 0.2320 |
| recall_doc | 0.5553 |
| rouge_l | 0.0958 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1202 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.1278 | 0.1071 | 0.0000 | 0.0000 |
| exact_article | 0.5763 | 0.4520 | 0.4102 | 0.5000 | 0.3871 |
| layman | 0.5882 | 0.2059 | 0.3118 | 1.0000 | 0.5714 |
| long_context | 0.7843 | 0.3026 | 0.1194 | 0.0000 | 0.0000 |
| multi_hop | 0.7821 | 0.2083 | 0.1753 | 0.6667 | 0.6667 |
| rule_application | 0.6742 | 0.3856 | 0.2523 | 0.4286 | 0.2857 |
| temporal_factual | 0.7143 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5804 | 0.3869 | 0.3458 | 0.5000 | 0.4286 |
| hard | 0.4482 | 0.1681 | 0.2779 | 0.9762 | 0.9318 |
| medium | 0.6412 | 0.3557 | 0.3338 | 0.4615 | 0.3750 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5654 | 0.2983 | 0.3248 | 0.7812 | 0.7407 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
