# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7787 |
| abstention_f1 | 0.6494 |
| abstention_precision | 0.5747 |
| abstention_recall | 0.7463 |
| am_faithfulness_score | 0.5122 |
| answer_faithfulness | 0.3566 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2719 |
| citation_groundedness | 0.5421 |
| citation_precision | 0.1713 |
| citation_recall | 0.2227 |
| corrective_retry_rate | 0.1803 |
| doc_citation_f1 | 0.5676 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.1835 |
| map_doc | 0.5106 |
| mean_latency_s | 19.2499 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2639 |
| mrr_doc | 0.5410 |
| ndcg_article | 0.2774 |
| ndcg_doc | 0.5496 |
| precision_article | 0.0406 |
| precision_doc | 0.0578 |
| reasoning_chain_score | 0.0064 |
| recall_article | 0.2217 |
| recall_doc | 0.5423 |
| rouge_l | 0.0904 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1151 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5833 | 0.1667 | 0.0972 | 0.3333 | 0.4000 |
| exact_article | 0.6356 | 0.4548 | 0.3646 | 0.4167 | 0.3571 |
| layman | 0.5294 | 0.1765 | 0.2157 | 0.5000 | 0.2857 |
| long_context | 0.8529 | 0.2245 | 0.1095 | 0.0000 | 0.0000 |
| multi_hop | 0.5192 | 0.0962 | 0.0851 | 0.6667 | 0.2857 |
| rule_application | 0.6591 | 0.3596 | 0.2111 | 0.4286 | 0.2857 |
| temporal_factual | 0.7143 | 0.3571 | 0.2857 | 0.0000 | 0.0000 |
| unanswerable | 0.0500 | 0.0000 | 0.5000 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5982 | 0.3899 | 0.2865 | 0.3333 | 0.2963 |
| hard | 0.4369 | 0.1471 | 0.2523 | 0.9286 | 0.8298 |
| medium | 0.6294 | 0.3224 | 0.2861 | 0.5385 | 0.4242 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5647 | 0.2775 | 0.2817 | 0.7344 | 0.6667 |
| fr | 0.0833 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
