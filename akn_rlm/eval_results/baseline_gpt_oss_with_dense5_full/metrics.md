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
| citation_f1 | 0.1747 |
| citation_groundedness | 0.0000 |
| citation_precision | 0.0997 |
| citation_recall | 0.1256 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4370 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0123 |
| map_article | 0.1157 |
| map_doc | 0.3847 |
| mean_latency_s | 2.6549 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1661 |
| mrr_doc | 0.4116 |
| ndcg_article | 0.1704 |
| ndcg_doc | 0.4219 |
| precision_article | 0.0189 |
| precision_doc | 0.0471 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1256 |
| recall_doc | 0.4208 |
| rouge_l | 0.0541 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0831 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5417 | 0.2500 | 0.2500 | 0.0000 | 0.0000 |
| exact_article | 0.5068 | 0.3339 | 0.2597 | 0.0000 | 0.0000 |
| layman | 0.3529 | 0.0588 | 0.0392 | 0.0000 | 0.0000 |
| long_context | 0.2647 | 0.0588 | 0.0107 | 0.0000 | 0.0000 |
| multi_hop | 0.5641 | 0.1410 | 0.0447 | 0.0000 | 0.0000 |
| rule_application | 0.4551 | 0.1641 | 0.0944 | 0.0000 | 0.0000 |
| temporal_factual | 0.2619 | 0.1905 | 0.1524 | 0.0000 | 0.0000 |
| unanswerable | 0.1750 | 0.0000 | 0.3750 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.4357 | 0.2982 | 0.2141 | 0.0000 | 0.0000 |
| hard | 0.3430 | 0.0971 | 0.1613 | 0.0000 | 0.0000 |
| medium | 0.4788 | 0.1627 | 0.1651 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.4157 | 0.1747 | 0.1838 | 0.0000 | 0.0000 |
| fr | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
