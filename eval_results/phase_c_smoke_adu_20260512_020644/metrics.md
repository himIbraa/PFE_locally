# Evaluation Metrics

**Total questions:** 16

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.9375 |
| abstention_f1 | 0.8000 |
| abstention_precision | 1.0000 |
| abstention_recall | 0.6667 |
| am_faithfulness_score | 0.5625 |
| answer_faithfulness | 0.1250 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2554 |
| citation_groundedness | 0.6406 |
| citation_precision | 0.2469 |
| citation_recall | 0.3396 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5729 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2525 |
| map_doc | 0.5833 |
| mean_latency_s | 11.4328 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.3375 |
| mrr_doc | 0.6458 |
| ndcg_article | 0.3623 |
| ndcg_doc | 0.6726 |
| precision_article | 0.0625 |
| precision_doc | 0.0750 |
| reasoning_chain_score | 0.0151 |
| recall_article | 0.3396 |
| recall_doc | 0.6875 |
| rouge_l | 0.1345 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1678 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.7500 | 0.1000 | 0.1667 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.5333 | 0.0000 | 0.0000 |
| layman | 1.0000 | 1.0000 | 0.8333 | 0.0000 | 0.0000 |
| long_context | 0.6667 | 0.1000 | 0.1765 | 0.0000 | 0.0000 |
| multi_hop | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.7500 | 0.5000 | 0.3333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.6889 | 0.0000 | 0.0000 |
| hard | 0.6042 | 0.1500 | 0.1275 | 1.0000 | 1.0000 |
| medium | 0.5000 | 0.2400 | 0.2000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6458 | 0.3375 | 0.2554 | 0.6667 | 0.8000 |
