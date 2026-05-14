# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8750 |
| abstention_f1 | 0.7059 |
| abstention_precision | 0.8571 |
| abstention_recall | 0.6000 |
| answer_faithfulness | 0.1750 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2954 |
| citation_groundedness | 0.6230 |
| citation_precision | 0.2294 |
| citation_recall | 0.2525 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6700 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2016 |
| map_doc | 0.6375 |
| mean_latency_s | 5.3571 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.3307 |
| mrr_doc | 0.6667 |
| ndcg_article | 0.3469 |
| ndcg_doc | 0.6795 |
| precision_article | 0.0525 |
| precision_doc | 0.0800 |
| reasoning_chain_score | 0.0132 |
| recall_article | 0.2525 |
| recall_doc | 0.7000 |
| rouge_l | 0.1052 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1436 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4667 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |
| exact_article | 0.8000 | 0.6000 | 0.4133 | 0.2500 | 0.4000 |
| layman | 0.8000 | 0.4667 | 0.4600 | 0.0000 | 0.0000 |
| long_context | 0.8667 | 0.3000 | 0.1299 | 0.0000 | 0.0000 |
| multi_hop | 1.0000 | 0.4286 | 0.2200 | 0.0000 | 0.0000 |
| rule_application | 0.8000 | 0.6000 | 0.3400 | 0.0000 | 0.0000 |
| temporal_factual | 0.6000 | 0.2000 | 0.1333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.8000 | 0.5333 | 0.4067 | 0.2500 | 0.3333 |
| hard | 0.6167 | 0.2321 | 0.2708 | 1.0000 | 1.0000 |
| medium | 0.6333 | 0.3250 | 0.2333 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6667 | 0.3307 | 0.2954 | 0.6000 | 0.7059 |
