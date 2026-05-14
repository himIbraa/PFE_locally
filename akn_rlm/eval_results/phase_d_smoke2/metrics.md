# Evaluation Metrics

**Total questions:** 16

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8750 |
| abstention_f1 | 0.5000 |
| abstention_precision | 1.0000 |
| abstention_recall | 0.3333 |
| am_faithfulness_score | 0.2858 |
| answer_faithfulness | 0.0625 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2765 |
| citation_groundedness | 0.7625 |
| citation_precision | 0.2698 |
| citation_recall | 0.3500 |
| corrective_retry_rate | 0.2500 |
| doc_citation_f1 | 0.6562 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.3097 |
| map_doc | 0.6771 |
| mean_latency_s | 19.6226 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.4688 |
| mrr_doc | 0.7396 |
| ndcg_article | 0.4572 |
| ndcg_doc | 0.7745 |
| precision_article | 0.0750 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0134 |
| recall_article | 0.3500 |
| recall_doc | 0.8125 |
| rouge_l | 0.1916 |
| sacrebleu | 0.0000 |
| token_f1 | 0.2322 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.4667 | 0.0000 | 0.0000 |
| layman | 1.0000 | 1.0000 | 0.7857 | 0.0000 | 0.0000 |
| long_context | 0.6667 | 0.5000 | 0.1765 | 0.0000 | 0.0000 |
| multi_hop | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.7500 | 0.7500 | 0.4500 | 0.0000 | 0.0000 |
| temporal_factual | 0.7500 | 0.5000 | 0.3333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6667 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.6444 | 0.0000 | 0.0000 |
| hard | 0.6042 | 0.2500 | 0.1275 | 0.5000 | 0.6667 |
| medium | 0.8000 | 0.5000 | 0.2943 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.7396 | 0.4688 | 0.2765 | 0.3333 | 0.5000 |
