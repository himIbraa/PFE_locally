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
| citation_f1 | 0.1734 |
| citation_groundedness | 0.0000 |
| citation_precision | 0.0989 |
| citation_recall | 0.1318 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4221 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1169 |
| map_doc | 0.3703 |
| mean_latency_s | 8.1396 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1521 |
| mrr_doc | 0.3993 |
| ndcg_article | 0.1568 |
| ndcg_doc | 0.4114 |
| precision_article | 0.0176 |
| precision_doc | 0.0459 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1318 |
| recall_doc | 0.4092 |
| rouge_l | 0.0575 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0854 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6250 | 0.2917 | 0.2917 | 0.0000 | 0.0000 |
| exact_article | 0.4729 | 0.3339 | 0.2662 | 0.0000 | 0.0000 |
| layman | 0.3529 | 0.0588 | 0.0392 | 0.0000 | 0.0000 |
| long_context | 0.2647 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi_hop | 0.4808 | 0.0128 | 0.0096 | 0.0000 | 0.0000 |
| rule_application | 0.4273 | 0.1705 | 0.1064 | 0.0000 | 0.0000 |
| temporal_factual | 0.4048 | 0.1905 | 0.1667 | 0.0000 | 0.0000 |
| unanswerable | 0.2000 | 0.0000 | 0.3500 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.4179 | 0.2804 | 0.2060 | 0.0000 | 0.0000 |
| hard | 0.3528 | 0.0647 | 0.1486 | 0.0000 | 0.0000 |
| medium | 0.4435 | 0.1735 | 0.1820 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.4027 | 0.1600 | 0.1824 | 0.0000 | 0.0000 |
| fr | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
