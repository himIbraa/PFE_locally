# Evaluation Metrics

**Total questions:** 8

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 1.0000 |
| abstention_f1 | 1.0000 |
| abstention_precision | 1.0000 |
| abstention_recall | 1.0000 |
| answer_faithfulness | 0.1250 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2625 |
| citation_groundedness | 0.6406 |
| citation_precision | 0.2333 |
| citation_recall | 0.3958 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6250 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2958 |
| map_doc | 0.6667 |
| mean_latency_s | 12.0615 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.4000 |
| mrr_doc | 0.7292 |
| ndcg_article | 0.4234 |
| ndcg_doc | 0.7664 |
| precision_article | 0.0625 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0209 |
| recall_article | 0.3958 |
| recall_doc | 0.8125 |
| rouge_l | 0.1749 |
| sacrebleu | 0.0000 |
| token_f1 | 0.2301 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 1.0000 | 0.2000 | 0.3333 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.2667 | 0.0000 | 0.0000 |
| layman | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| long_context | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi_hop | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 1.0000 | 1.0000 | 0.5000 | 0.0000 | 0.0000 |
| temporal_factual | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.6333 | 0.0000 | 0.0000 |
| hard | 0.4583 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| medium | 1.0000 | 0.6000 | 0.4167 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.7292 | 0.4000 | 0.2625 | 1.0000 | 1.0000 |
