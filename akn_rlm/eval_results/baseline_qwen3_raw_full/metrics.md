# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7254 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0338 |
| citation_groundedness | 0.0000 |
| citation_precision | 0.0260 |
| citation_recall | 0.0686 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.2917 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.0435 |
| map_doc | 0.2824 |
| mean_latency_s | 4.3235 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.0583 |
| mrr_doc | 0.3135 |
| ndcg_article | 0.0670 |
| ndcg_doc | 0.3141 |
| precision_article | 0.0139 |
| precision_doc | 0.0320 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.0686 |
| recall_doc | 0.2835 |
| rouge_l | 0.0020 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0045 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.0833 | 0.0185 | 0.0000 | 0.0000 |
| exact_article | 0.2712 | 0.0811 | 0.0634 | 0.0000 | 0.0000 |
| layman | 0.1176 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.4118 | 0.0588 | 0.0362 | 0.0000 | 0.0000 |
| multi_hop | 0.4808 | 0.0609 | 0.0202 | 0.0000 | 0.0000 |
| rule_application | 0.3485 | 0.0716 | 0.0408 | 0.0000 | 0.0000 |
| temporal_factual | 0.4286 | 0.1429 | 0.0317 | 0.0000 | 0.0000 |
| unanswerable | 0.1750 | 0.0031 | 0.0056 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.2321 | 0.0795 | 0.0570 | 0.0000 | 0.0000 |
| hard | 0.3350 | 0.0360 | 0.0154 | 0.0000 | 0.0000 |
| medium | 0.3412 | 0.0713 | 0.0407 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.3125 | 0.0613 | 0.0355 | 0.0000 | 0.0000 |
| fr | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
