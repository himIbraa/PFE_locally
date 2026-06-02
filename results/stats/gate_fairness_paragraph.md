## Gate Fairness — §4.2 insert (after headline table)

AKN-RLM's Hallucinated Citation Rate of 0.0000 could in principle be an
artefact of the citation-existence gate
(`gates/citation_existence.py`), which filters any citation whose `doc_id` or
`article_ref` cannot be resolved in the runtime ArticleRegistry. To rule this
out, we ran a dedicated ablation (`locked_no_citation_gate`, n=244)
identical to the canonical Phase C configuration except that
`AKN_NO_CITATION_GATE=1` bypasses both `citation_existence` and `span_existence`
in `root_controller`. Table~\ref{tab:gate_fairness} presents the resulting
$2\times 2$ comparison.

The empirical result is unambiguous: with the gate disabled, AKN-RLM's HCR
remains 0.0000 and its Cite~F1 is 0.2972 (vs.\ 0.3010
with the gate; the difference is within the bootstrap CI). The gate is
structurally redundant in the dispatcher path because every citation candidate
originates from BM25/dense retrieval over the closed Algerian Legal Corpus —
the verifier sub-LM only ever selects from articles it has been shown, so
non-corpus identifiers cannot enter the citation set in the first place. The
same retrieval-over-closed-corpus property holds for the best Phase-1 baseline
(Hybrid+Rerank), which records HCR~$=$~0.0000 with or without the
gate (0.0000).

The residual performance gap — AKN-RLM Cite~F1 $=$ 0.3010 vs.\
Hybrid+Rerank Cite~F1 $=$ 0.1052 — is therefore entirely attributable
to the typed-handler RLM architecture rather than to differential filtering at
the gate. The gate is best understood as a defence-in-depth safeguard for
future open-corpus deployments (where retrieval is unconstrained), not as a
post-hoc bias in the closed-corpus evaluation reported here.
