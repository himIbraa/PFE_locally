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
| citation_f1 | 0.2871 |
| citation_groundedness | 0.6317 |
| citation_precision | 0.2112 |
| citation_recall | 0.2567 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6250 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2046 |
| map_doc | 0.5646 |
| mean_latency_s | 8.2175 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.3187 |
| mrr_doc | 0.6083 |
| ndcg_article | 0.3311 |
| ndcg_doc | 0.6321 |
| precision_article | 0.0525 |
| precision_doc | 0.0725 |
| reasoning_chain_score | 0.0132 |
| recall_article | 0.2567 |
| recall_doc | 0.6500 |
| rouge_l | 0.1112 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1507 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4667 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |
| exact_article | 0.8000 | 0.6000 | 0.4133 | 0.2500 | 0.4000 |
| layman | 0.8000 | 0.5000 | 0.3733 | 0.0000 | 0.0000 |
| long_context | 0.7000 | 0.4000 | 0.1299 | 0.0000 | 0.0000 |
| multi_hop | 0.8000 | 0.3000 | 0.1800 | 0.0000 | 0.0000 |
| rule_application | 0.8000 | 0.6000 | 0.4000 | 0.0000 | 0.0000 |
| temporal_factual | 0.5000 | 0.1000 | 0.1333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.8000 | 0.5500 | 0.3633 | 0.2500 | 0.3333 |
| hard | 0.5000 | 0.2000 | 0.2608 | 1.0000 | 1.0000 |
| medium | 0.6333 | 0.3250 | 0.2633 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6083 | 0.3187 | 0.2871 | 0.6000 | 0.7059 |
