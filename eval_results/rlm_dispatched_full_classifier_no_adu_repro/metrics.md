# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8033 |
| abstention_f1 | 0.6757 |
| abstention_precision | 0.6173 |
| abstention_recall | 0.7463 |
| am_faithfulness_score | 0.3410 |
| answer_faithfulness | 0.3320 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2879 |
| citation_groundedness | 0.5256 |
| citation_precision | 0.1788 |
| citation_recall | 0.2446 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5877 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.2003 |
| map_doc | 0.5229 |
| mean_latency_s | 9.5399 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2882 |
| mrr_doc | 0.5663 |
| ndcg_article | 0.3016 |
| ndcg_doc | 0.5789 |
| precision_article | 0.0434 |
| precision_doc | 0.0631 |
| reasoning_chain_score | 0.0081 |
| recall_article | 0.2425 |
| recall_doc | 0.5683 |
| rouge_l | 0.0971 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1249 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.1736 | 0.1389 | 0.3333 | 0.3333 |
| exact_article | 0.6610 | 0.4944 | 0.3897 | 0.5000 | 0.4444 |
| layman | 0.4706 | 0.1373 | 0.2431 | 1.0000 | 0.4444 |
| long_context | 0.7549 | 0.2075 | 0.1086 | 0.0000 | 0.0000 |
| multi_hop | 0.7628 | 0.1859 | 0.0909 | 0.0000 | 0.0000 |
| rule_application | 0.6894 | 0.3920 | 0.2335 | 0.4286 | 0.2727 |
| temporal_factual | 0.7143 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0500 | 0.0250 | 0.5125 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5893 | 0.4018 | 0.3082 | 0.5000 | 0.4138 |
| hard | 0.4773 | 0.1637 | 0.2465 | 0.8810 | 0.8810 |
| medium | 0.6588 | 0.3642 | 0.3247 | 0.5385 | 0.4000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5912 | 0.3031 | 0.2985 | 0.7344 | 0.7015 |
| fr | 0.0833 | 0.0000 | 0.0833 | 1.0000 | 0.4286 |
