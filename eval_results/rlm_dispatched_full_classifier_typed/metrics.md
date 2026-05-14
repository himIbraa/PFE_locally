# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8033 |
| abstention_f1 | 0.6757 |
| abstention_precision | 0.6173 |
| abstention_recall | 0.7463 |
| answer_faithfulness | 0.3320 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2980 |
| citation_groundedness | 0.5286 |
| citation_precision | 0.1844 |
| citation_recall | 0.2525 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5889 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2039 |
| map_doc | 0.5301 |
| mean_latency_s | 5.7784 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2918 |
| mrr_doc | 0.5710 |
| ndcg_article | 0.3062 |
| ndcg_doc | 0.5811 |
| precision_article | 0.0443 |
| precision_doc | 0.0627 |
| reasoning_chain_score | 0.0085 |
| recall_article | 0.2519 |
| recall_doc | 0.5683 |
| rouge_l | 0.0948 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1198 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.1625 | 0.1250 | 0.3333 | 0.3333 |
| exact_article | 0.6441 | 0.5028 | 0.4240 | 0.4167 | 0.3704 |
| layman | 0.5588 | 0.1961 | 0.2863 | 1.0000 | 0.5714 |
| long_context | 0.8137 | 0.2340 | 0.1151 | 0.0000 | 0.0000 |
| multi_hop | 0.6923 | 0.1667 | 0.0999 | 0.3333 | 0.2222 |
| rule_application | 0.7197 | 0.3932 | 0.2265 | 0.4286 | 0.3000 |
| temporal_factual | 0.7143 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0375 | 0.0125 | 0.5125 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6161 | 0.4107 | 0.3366 | 0.4167 | 0.3846 |
| hard | 0.4790 | 0.1559 | 0.2541 | 0.9048 | 0.8636 |
| medium | 0.6529 | 0.3782 | 0.3258 | 0.5385 | 0.4118 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5963 | 0.3069 | 0.3091 | 0.7344 | 0.7015 |
| fr | 0.0833 | 0.0000 | 0.0833 | 1.0000 | 0.4286 |
