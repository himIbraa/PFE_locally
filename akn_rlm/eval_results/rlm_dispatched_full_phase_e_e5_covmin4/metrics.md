# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8115 |
| abstention_f1 | 0.6806 |
| abstention_precision | 0.6364 |
| abstention_recall | 0.7313 |
| am_faithfulness_score | 0.4888 |
| answer_faithfulness | 0.3156 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2928 |
| citation_groundedness | 0.5430 |
| citation_precision | 0.1781 |
| citation_recall | 0.2567 |
| corrective_retry_rate | 0.1803 |
| doc_citation_f1 | 0.6146 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.2054 |
| map_doc | 0.5457 |
| mean_latency_s | 13.3615 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2928 |
| mrr_doc | 0.5867 |
| ndcg_article | 0.3072 |
| ndcg_doc | 0.5980 |
| precision_article | 0.0455 |
| precision_doc | 0.0652 |
| reasoning_chain_score | 0.0084 |
| recall_article | 0.2546 |
| recall_doc | 0.5895 |
| rouge_l | 0.0965 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1236 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5833 | 0.1944 | 0.1667 | 0.3333 | 0.4000 |
| exact_article | 0.6780 | 0.4766 | 0.3989 | 0.4167 | 0.3704 |
| layman | 0.4706 | 0.1471 | 0.1882 | 0.5000 | 0.2500 |
| long_context | 0.7549 | 0.2304 | 0.0878 | 0.0000 | 0.0000 |
| multi_hop | 0.7436 | 0.1859 | 0.1231 | 0.3333 | 0.2857 |
| rule_application | 0.7500 | 0.4129 | 0.2477 | 0.4286 | 0.3158 |
| temporal_factual | 0.7143 | 0.2857 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0375 | 0.0125 | 0.5083 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6071 | 0.3771 | 0.3012 | 0.3333 | 0.2857 |
| hard | 0.4725 | 0.1610 | 0.2542 | 0.9048 | 0.8837 |
| medium | 0.7118 | 0.3971 | 0.3341 | 0.5385 | 0.4667 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6085 | 0.3080 | 0.3036 | 0.7188 | 0.7023 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
