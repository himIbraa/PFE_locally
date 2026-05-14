# Evaluation Metrics

**Total questions:** 16

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.9375 |
| abstention_f1 | 0.8000 |
| abstention_precision | 1.0000 |
| abstention_recall | 0.6667 |
| am_faithfulness_score | 0.4232 |
| answer_faithfulness | 0.1250 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2908 |
| citation_groundedness | 0.7604 |
| citation_precision | 0.2990 |
| citation_recall | 0.3500 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6333 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2720 |
| map_doc | 0.7188 |
| mean_latency_s | 11.3669 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.4740 |
| mrr_doc | 0.7500 |
| ndcg_article | 0.4811 |
| ndcg_doc | 0.7664 |
| precision_article | 0.0688 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0156 |
| recall_article | 0.3500 |
| recall_doc | 0.7812 |
| rouge_l | 0.1505 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1798 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.7500 | 0.1250 | 0.1667 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.5333 | 0.0000 | 0.0000 |
| layman | 1.0000 | 1.0000 | 0.7500 | 0.0000 | 0.0000 |
| long_context | 1.0000 | 0.5000 | 0.1765 | 0.0000 | 0.0000 |
| multi_hop | 1.0000 | 0.1667 | 0.1667 | 0.0000 | 0.0000 |
| rule_application | 0.5000 | 0.5000 | 0.2000 | 0.0000 | 0.0000 |
| temporal_factual | 0.7500 | 0.5000 | 0.3333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.6889 | 0.0000 | 0.0000 |
| hard | 0.6875 | 0.2917 | 0.1691 | 1.0000 | 1.0000 |
| medium | 0.7000 | 0.4500 | 0.2467 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.7500 | 0.4740 | 0.2908 | 0.6667 | 0.8000 |
