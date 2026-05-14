# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7992 |
| abstention_f1 | 0.6755 |
| abstention_precision | 0.6071 |
| abstention_recall | 0.7612 |
| am_faithfulness_score | 0.4896 |
| answer_faithfulness | 0.3443 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3010 |
| citation_groundedness | 0.5120 |
| citation_precision | 0.1965 |
| citation_recall | 0.2549 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5959 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0164 |
| map_article | 0.2124 |
| map_doc | 0.5236 |
| mean_latency_s | 6.9037 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2999 |
| mrr_doc | 0.5567 |
| ndcg_article | 0.3147 |
| ndcg_doc | 0.5696 |
| precision_article | 0.0471 |
| precision_doc | 0.0631 |
| reasoning_chain_score | 0.0080 |
| recall_article | 0.2549 |
| recall_doc | 0.5710 |
| rouge_l | 0.1010 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1271 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5833 | 0.1625 | 0.1250 | 0.3333 | 0.4000 |
| exact_article | 0.6525 | 0.4972 | 0.3998 | 0.5000 | 0.4286 |
| layman | 0.5294 | 0.1765 | 0.2235 | 0.5000 | 0.2500 |
| long_context | 0.7451 | 0.2500 | 0.1260 | 0.0000 | 0.0000 |
| multi_hop | 0.6282 | 0.1538 | 0.1206 | 0.6667 | 0.4444 |
| rule_application | 0.6944 | 0.4340 | 0.2659 | 0.4286 | 0.2727 |
| temporal_factual | 0.7143 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0375 | 0.0125 | 0.5100 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5982 | 0.3988 | 0.3022 | 0.4167 | 0.3571 |
| hard | 0.4450 | 0.1699 | 0.2690 | 0.9286 | 0.8864 |
| medium | 0.6647 | 0.3923 | 0.3390 | 0.5385 | 0.4000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5812 | 0.3154 | 0.3123 | 0.7500 | 0.7007 |
| fr | 0.0833 | 0.0000 | 0.0833 | 1.0000 | 0.4286 |
