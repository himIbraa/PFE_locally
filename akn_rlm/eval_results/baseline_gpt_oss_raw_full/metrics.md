# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7254 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0555 |
| citation_groundedness | 0.0000 |
| citation_precision | 0.0249 |
| citation_recall | 0.0314 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.1839 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0205 |
| map_article | 0.0277 |
| map_doc | 0.1732 |
| mean_latency_s | 214.7308 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.0338 |
| mrr_doc | 0.1916 |
| ndcg_article | 0.0359 |
| ndcg_doc | 0.1954 |
| precision_article | 0.0045 |
| precision_doc | 0.0221 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.0314 |
| recall_doc | 0.1878 |
| rouge_l | 0.0623 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0969 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.2373 | 0.1102 | 0.1455 | 0.0000 | 0.0000 |
| layman | 0.2059 | 0.0000 | 0.0588 | 0.0000 | 0.0000 |
| long_context | 0.0441 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi_hop | 0.3269 | 0.0385 | 0.0495 | 0.0000 | 0.0000 |
| rule_application | 0.1818 | 0.0114 | 0.0103 | 0.0000 | 0.0000 |
| temporal_factual | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unanswerable | 0.1500 | 0.0000 | 0.0500 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.2054 | 0.0804 | 0.0997 | 0.0000 | 0.0000 |
| hard | 0.2112 | 0.0121 | 0.0337 | 0.0000 | 0.0000 |
| medium | 0.1588 | 0.0294 | 0.0529 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.1843 | 0.0356 | 0.0584 | 0.0000 | 0.0000 |
| fr | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
