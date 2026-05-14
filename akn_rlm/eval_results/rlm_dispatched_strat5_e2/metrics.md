# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8250 |
| abstention_f1 | 0.6316 |
| abstention_precision | 0.6667 |
| abstention_recall | 0.6000 |
| answer_faithfulness | 0.2250 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2745 |
| citation_groundedness | 0.6571 |
| citation_precision | 0.2092 |
| citation_recall | 0.2275 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5992 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1808 |
| map_doc | 0.5625 |
| mean_latency_s | 2.7789 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2612 |
| mrr_doc | 0.6000 |
| ndcg_article | 0.2755 |
| ndcg_doc | 0.6065 |
| precision_article | 0.0450 |
| precision_doc | 0.0650 |
| reasoning_chain_score | 0.0122 |
| recall_article | 0.2275 |
| recall_doc | 0.5875 |
| rouge_l | 0.0956 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1296 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.0400 | 0.0667 | 0.0000 | 0.0000 |
| exact_article | 0.8000 | 0.6000 | 0.3733 | 0.2500 | 0.4000 |
| layman | 0.6000 | 0.4000 | 0.3600 | 0.0000 | 0.0000 |
| long_context | 1.0000 | 0.2500 | 0.1364 | 0.0000 | 0.0000 |
| multi_hop | 0.8000 | 0.1000 | 0.1000 | 0.0000 | 0.0000 |
| rule_application | 0.4000 | 0.4000 | 0.2933 | 0.0000 | 0.0000 |
| temporal_factual | 0.7000 | 0.3000 | 0.2667 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7000 | 0.5000 | 0.3533 | 0.2500 | 0.2857 |
| hard | 0.6250 | 0.1625 | 0.2758 | 1.0000 | 0.9091 |
| medium | 0.4500 | 0.2200 | 0.1933 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6000 | 0.2612 | 0.2745 | 0.6000 | 0.6316 |
