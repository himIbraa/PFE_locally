# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8361 |
| abstention_f1 | 0.7260 |
| abstention_precision | 0.6709 |
| abstention_recall | 0.7910 |
| answer_faithfulness | 0.3238 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3049 |
| citation_groundedness | 0.5374 |
| citation_precision | 0.1991 |
| citation_recall | 0.2122 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6096 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1924 |
| map_doc | 0.5171 |
| mean_latency_s | 5.3262 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2788 |
| mrr_doc | 0.5553 |
| ndcg_article | 0.2885 |
| ndcg_doc | 0.5633 |
| precision_article | 0.0361 |
| precision_doc | 0.0598 |
| reasoning_chain_score | 0.0055 |
| recall_article | 0.2122 |
| recall_doc | 0.5478 |
| rouge_l | 0.0962 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1232 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5694 | 0.1833 | 0.0794 | 0.0000 | 0.0000 |
| exact_article | 0.6102 | 0.4520 | 0.4277 | 0.4167 | 0.3571 |
| layman | 0.5882 | 0.1471 | 0.2824 | 1.0000 | 0.5714 |
| long_context | 0.8333 | 0.2255 | 0.0949 | 0.0000 | 0.0000 |
| multi_hop | 0.6346 | 0.1282 | 0.1216 | 1.0000 | 0.6667 |
| rule_application | 0.7045 | 0.4318 | 0.2572 | 0.4286 | 0.3158 |
| temporal_factual | 0.7857 | 0.1429 | 0.0952 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5982 | 0.3780 | 0.3554 | 0.4167 | 0.3704 |
| hard | 0.4288 | 0.1424 | 0.2629 | 1.0000 | 0.9333 |
| medium | 0.6804 | 0.3788 | 0.3227 | 0.4615 | 0.4138 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5711 | 0.2932 | 0.3164 | 0.7812 | 0.7463 |
| fr | 0.2500 | 0.0000 | 0.0833 | 1.0000 | 0.5000 |
