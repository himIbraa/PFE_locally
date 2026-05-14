# Evaluation Metrics

**Total questions:** 27

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.9630 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0370 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0985 |
| citation_groundedness | 0.5977 |
| citation_precision | 0.0844 |
| citation_recall | 0.1370 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5667 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.0999 |
| map_doc | 0.6852 |
| mean_latency_s | 5.4483 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2337 |
| mrr_doc | 0.7593 |
| ndcg_article | 0.2418 |
| ndcg_doc | 0.7835 |
| precision_article | 0.0370 |
| precision_doc | 0.0963 |
| reasoning_chain_score | 0.0186 |
| recall_article | 0.1222 |
| recall_doc | 0.7778 |
| rouge_l | 0.0955 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1374 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| long_context | 0.8500 | 0.3167 | 0.0893 | 0.0000 | 0.0000 |
| multi_hop | 0.8500 | 0.2143 | 0.1100 | 0.0000 | 0.0000 |
| temporal_factual | 0.5000 | 0.1429 | 0.0952 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| hard | 0.7692 | 0.2427 | 0.1023 | 0.0000 | 0.0000 |
| medium | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.7593 | 0.2337 | 0.0985 | 0.0000 | 0.0000 |
