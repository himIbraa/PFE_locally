# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7910 |
| abstention_f1 | 0.6667 |
| abstention_precision | 0.5930 |
| abstention_recall | 0.7612 |
| am_faithfulness_score | 0.4945 |
| answer_faithfulness | 0.3525 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2797 |
| citation_groundedness | 0.5176 |
| citation_precision | 0.1637 |
| citation_recall | 0.2339 |
| corrective_retry_rate | 0.2336 |
| doc_citation_f1 | 0.5745 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1833 |
| map_doc | 0.4980 |
| mean_latency_s | 118.2070 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2577 |
| mrr_doc | 0.5464 |
| ndcg_article | 0.2749 |
| ndcg_doc | 0.5599 |
| precision_article | 0.0414 |
| precision_doc | 0.0607 |
| reasoning_chain_score | 0.0084 |
| recall_article | 0.2302 |
| recall_doc | 0.5499 |
| rouge_l | 0.0917 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1180 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5417 | 0.1625 | 0.1250 | 0.3333 | 0.3333 |
| exact_article | 0.6102 | 0.4299 | 0.3676 | 0.5000 | 0.4138 |
| layman | 0.4118 | 0.1471 | 0.2431 | 1.0000 | 0.3636 |
| long_context | 0.7353 | 0.1657 | 0.1103 | 0.0000 | 0.0000 |
| multi_hop | 0.7821 | 0.1474 | 0.1116 | 0.3333 | 0.3333 |
| rule_application | 0.6742 | 0.3697 | 0.2245 | 0.4286 | 0.2609 |
| temporal_factual | 0.7143 | 0.2857 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0375 | 0.0000 | 0.5000 | 0.9500 | 0.9744 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5268 | 0.3423 | 0.2870 | 0.5000 | 0.3750 |
| hard | 0.4790 | 0.1325 | 0.2448 | 0.9048 | 0.8941 |
| medium | 0.6412 | 0.3535 | 0.3172 | 0.5385 | 0.3889 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5661 | 0.2710 | 0.2899 | 0.7500 | 0.6857 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
