# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8156 |
| abstention_f1 | 0.6853 |
| abstention_precision | 0.6447 |
| abstention_recall | 0.7313 |
| am_faithfulness_score | 0.4980 |
| answer_faithfulness | 0.3115 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3017 |
| citation_groundedness | 0.5470 |
| citation_precision | 0.1886 |
| citation_recall | 0.2619 |
| corrective_retry_rate | 0.2049 |
| doc_citation_f1 | 0.6098 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.2103 |
| map_doc | 0.5495 |
| mean_latency_s | 10.0071 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2942 |
| mrr_doc | 0.5861 |
| ndcg_article | 0.3095 |
| ndcg_doc | 0.5974 |
| precision_article | 0.0463 |
| precision_doc | 0.0656 |
| reasoning_chain_score | 0.0082 |
| recall_article | 0.2619 |
| recall_doc | 0.5943 |
| rouge_l | 0.1049 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1318 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5694 | 0.1625 | 0.1250 | 0.3333 | 0.4000 |
| exact_article | 0.6864 | 0.4921 | 0.4143 | 0.4167 | 0.4000 |
| layman | 0.5294 | 0.1618 | 0.2471 | 0.5000 | 0.2857 |
| long_context | 0.8137 | 0.2485 | 0.1107 | 0.0000 | 0.0000 |
| multi_hop | 0.7051 | 0.1058 | 0.1025 | 0.3333 | 0.2857 |
| rule_application | 0.7273 | 0.4255 | 0.2487 | 0.4286 | 0.2857 |
| temporal_factual | 0.7143 | 0.3571 | 0.2857 | 0.0000 | 0.0000 |
| unanswerable | 0.0375 | 0.0125 | 0.5125 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6339 | 0.4033 | 0.3153 | 0.3333 | 0.3200 |
| hard | 0.4725 | 0.1600 | 0.2559 | 0.9048 | 0.8837 |
| medium | 0.6922 | 0.3851 | 0.3481 | 0.5385 | 0.4375 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6121 | 0.3094 | 0.3130 | 0.7188 | 0.7132 |
| fr | 0.0833 | 0.0000 | 0.0833 | 1.0000 | 0.4286 |
