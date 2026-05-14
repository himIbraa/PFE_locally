# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8500 |
| abstention_f1 | 0.6667 |
| abstention_precision | 0.7500 |
| abstention_recall | 0.6000 |
| answer_faithfulness | 0.2000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2785 |
| citation_groundedness | 0.6529 |
| citation_precision | 0.2175 |
| citation_recall | 0.2267 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6292 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1900 |
| map_doc | 0.5750 |
| mean_latency_s | 7.3899 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.3050 |
| mrr_doc | 0.6000 |
| ndcg_article | 0.3162 |
| ndcg_doc | 0.6131 |
| precision_article | 0.0450 |
| precision_doc | 0.0675 |
| reasoning_chain_score | 0.0132 |
| recall_article | 0.2267 |
| recall_doc | 0.6250 |
| rouge_l | 0.1015 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1395 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6000 | 0.0400 | 0.0667 | 0.0000 | 0.0000 |
| exact_article | 0.8000 | 0.6000 | 0.4133 | 0.2500 | 0.4000 |
| layman | 0.6000 | 0.6000 | 0.4000 | 0.0000 | 0.0000 |
| long_context | 0.9000 | 0.4000 | 0.1016 | 0.0000 | 0.0000 |
| multi_hop | 0.7000 | 0.3000 | 0.1800 | 0.0000 | 0.0000 |
| rule_application | 0.6000 | 0.4000 | 0.3333 | 0.0000 | 0.0000 |
| temporal_factual | 0.6000 | 0.1000 | 0.1333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7000 | 0.6000 | 0.4400 | 0.2500 | 0.2857 |
| hard | 0.5500 | 0.2000 | 0.2537 | 1.0000 | 1.0000 |
| medium | 0.6000 | 0.2200 | 0.1667 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6000 | 0.3050 | 0.2785 | 0.6000 | 0.6667 |
