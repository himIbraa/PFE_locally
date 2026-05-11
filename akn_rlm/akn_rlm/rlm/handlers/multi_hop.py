"""Multi-hop query handler — first typed Phase-2 handler.

Pipeline (each step is a small slot the LLM fills, never freeform code):

  1. Doc-route the query via :class:`DocRouter` (alias + numeric-id +
     BM25 channels) to get 1-3 likely ``doc_id`` predictions. Empty
     prediction → fall back to corpus-wide retrieval.

  2. LLM-decompose the question into atomic sub-questions via
     :func:`call_decomposer`. On parse failure or empty result the
     handler falls back to a single sub-question = the original query.
     Sub-questions tagged ``type="foreign_law"`` by the decomposer are
     skipped (an `unanswerable` handler will own those in R5).

  3. For each surviving sub-question:
     a. RRF-fuse(BM25, Dense) — the same fusion used by B3/B4. Hits
        whose ``doc_id`` is not in the routed set are filtered out
        post-hoc (cheap; ``k_each`` is set deep enough that the routed
        docs still surface enough articles).
     b. Take top-``top_k_per_subq`` candidates from the fused list.
     c. Verify the top-``verify_top_n`` via the sub-LM verifier
        (:func:`call_verifier`). Survivors are those with
        ``relevant=True`` AND ``confidence >= verify_threshold``.

  4. Aggregate every verified citation across sub-questions, dedup by
     ``(doc_id, canonical article_ref)``, sort by best verifier
     confidence, and take the top-``final_top_k``.

  5. Synthesise an answer over the surviving citations via
     :func:`call_summarizer`. If the summariser returns ``null`` we fall
     back to the deterministic Arabic template used by the baselines so
     the answer is never empty as long as we have valid citations.

  6. Build the answer dict in the same shape as the baseline pipelines
     so :func:`akn_rlm.eval.runner._answer_to_result` can consume it
     unchanged.

The handler is intentionally self-contained (no LangGraph, no
``RootController``). It is callable as a baseline-shaped pipeline so
the existing scripts can run it through the same evaluation harness.

Sub-LM call budget per query: 1 decomposer + ``max_sub_qs *
verify_top_n`` verifier calls + 1 summariser = ≤ 1 + 3*3 + 1 = 11
calls. That leaves headroom inside the project's
``max_sub_calls=12`` budget (config.py) but is well above the
multi_hop=8 budget called out in HANDOFF §3 — call sites that need
the tighter envelope can lower ``max_sub_qs`` and/or
``verify_top_n``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse
from akn_rlm.rlm.routing import DocRouter, build_doc_router
from akn_rlm.rlm.sub_worker import call_decomposer, call_summarizer, call_verifier
from akn_rlm.rlm.supervisor import (
    DEFAULT_PLAN_MIN_CONTENT_TOKENS,
    PlanSupervisorFn,
    SupervisorFn,
    should_supervise,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# R9.3: budget expansion. Multi_hop questions span 2-4 articles often
# scattered across docs; widening the search lifts both Cite F1 and
# MRR art. Total worst-case sub-LM call count under the new budget:
#   1 decomposer + max_sub_qs * verify_top_n + 1 summariser
#   = 1 + 5*4 + 1 = 22 calls
# which is the rationale for the per-handler ``DEFAULT_MAX_SUB_CALLS``
# override of 25 below (the global project budget stays at 12).
DEFAULT_TOP_K_PER_SUBQ: int = 8
DEFAULT_VERIFY_TOP_N: int = 4
# F5: reverted F4's 10→5 back to R9.3's 10. F4 shows the tighter
# top_k didn't lift MH (0.121→0.118) — wider candidate pool is
# better for MH because the gold articles are scattered across
# docs. Precision lift instead comes from R9.5 supervisor re-rank.
DEFAULT_FINAL_TOP_K: int = 10
DEFAULT_K_EACH: int = 30
DEFAULT_MAX_SUB_QS: int = 5
DEFAULT_VERIFY_THRESHOLD: float = 0.5
DEFAULT_ROUTE_TOP_N: int = 3
#: Per-handler max sub-LM call envelope. The project default is 12
#: (``akn_rlm.rlm.recursion_budget``) but multi_hop legitimately needs
#: more headroom under the R9.3 retune. Stored on the handler and
#: surfaced in ``_telemetry["max_sub_calls"]`` so eval scripts can
#: detect when a query approached the cap.
DEFAULT_MAX_SUB_CALLS: int = 25
#: Fix-MH consensus boost: for each *additional* sub-Q that cites the same
#: (doc_id, article_ref), add this to the final ranking score. The verifier
#: confidence on its own is roughly uniform across adjacent legal articles
#: inside a code, so the consensus across sub-questions is a stronger
#: discriminator. Calibrated to flip the ranking when two candidates' raw
#: confidences differ by < 0.25.
DEFAULT_CONSENSUS_BOOST: float = 0.25
SUPPORT_SPAN_LEN: int = 280

# Telemetry tag — stays stable so the comparison script can pick it out.
TELEMETRY_BASELINE: str = "rlm_multi_hop"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

DecomposerFn = Callable[[Any, str, str], dict]
VerifierFn = Callable[[Any, str, dict, str], dict]
SummarizerFn = Callable[[Any, str, list[dict], str], dict]


class MultiHopHandler:
    """Typed multi-hop handler: route -> decompose -> per-sub-q hybrid + verify -> synth."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        top_k_per_subq: int = DEFAULT_TOP_K_PER_SUBQ,
        verify_top_n: int = DEFAULT_VERIFY_TOP_N,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        k_each: int = DEFAULT_K_EACH,
        max_sub_qs: int = DEFAULT_MAX_SUB_QS,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
        route_top_n: int = DEFAULT_ROUTE_TOP_N,
        max_sub_calls: int = DEFAULT_MAX_SUB_CALLS,
        consensus_boost: float = DEFAULT_CONSENSUS_BOOST,
        decomposer_fn: Optional[DecomposerFn] = None,
        verifier_fn: Optional[VerifierFn] = None,
        summarizer_fn: Optional[SummarizerFn] = None,
        supervisor_fn: Optional[SupervisorFn] = None,
        plan_supervisor_fn: Optional[PlanSupervisorFn] = None,
        plan_min_content_tokens: int = DEFAULT_PLAN_MIN_CONTENT_TOKENS,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._sub_model = sub_model
        self._top_k_per_subq = top_k_per_subq
        self._verify_top_n = verify_top_n
        self._final_top_k = final_top_k
        self._k_each = k_each
        self._max_sub_qs = max_sub_qs
        self._verify_threshold = verify_threshold
        self._route_top_n = route_top_n
        self._max_sub_calls = int(max_sub_calls)
        self._consensus_boost = float(consensus_boost)
        # Injectable LLM-call wrappers so unit tests don't hit the real LLM.
        self._decomposer_fn = decomposer_fn or call_decomposer
        self._verifier_fn = verifier_fn or call_verifier
        self._summarizer_fn = summarizer_fn or call_summarizer
        # R9.5: optional gpt-oss-120b per-citation re-ranker.
        self._supervisor_fn = supervisor_fn
        # R9.6: optional gpt-oss-120b sub-question planner. When set
        # AND the query has >= ``plan_min_content_tokens`` content
        # tokens, it replaces the Qwen-based decomposer with a strong-
        # model plan that also predicts ``target_docs`` per sub-question.
        self._plan_supervisor_fn = plan_supervisor_fn
        self._plan_min_content_tokens = int(plan_min_content_tokens)

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain("empty_query", routed=[], sub_qs=[], sub_calls=0)

        # 1. Doc-route
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)
        log.debug("multi_hop route: %s", routed_ids)

        # 2. Decompose. R9.6: try the gpt-oss-120b plan supervisor
        # FIRST when the query is long enough; on parse failure or
        # short query, fall back to the Qwen-based decomposer.
        sub_calls = 0
        plan_supervisor_used = False
        sub_qs: list[dict[str, Any]] = []

        if (self._plan_supervisor_fn is not None
                and self._count_content_tokens(query) >= self._plan_min_content_tokens):
            try:
                plan = self._plan_supervisor_fn(
                    self._llm_pool, query, routed_doc_ids=routed_ids,
                )
                sub_calls += 1
            except Exception as exc:
                log.warning(
                    "multi_hop plan supervisor failed (%s) — falling back to decomposer",
                    exc,
                )
                plan = {}
            if isinstance(plan, dict) and plan.get("sub_questions"):
                # The plan supervisor already filters target_docs against
                # routed_ids and caps the list, so trust its shape.
                sub_qs = []
                for sq in plan["sub_questions"][: self._max_sub_qs]:
                    text = (sq.get("text") or "").strip()
                    if not text:
                        continue
                    sub_qs.append({
                        "id":          sq.get("id") or f"sq{len(sub_qs) + 1}",
                        "text":        text,
                        "type":        sq.get("type") or "rule_application",
                        "target_docs": list(sq.get("target_docs") or []),
                    })
                if sub_qs:
                    plan_supervisor_used = True

        if not sub_qs:
            try:
                decomp = self._decomposer_fn(
                    self._llm_pool, query, self._sub_model,
                )
                sub_calls += 1
            except Exception as exc:
                log.warning(
                    "multi_hop decomposer failed: %s — single sub-q fallback",
                    exc,
                )
                decomp = {}

            sub_qs = self._extract_sub_questions(decomp, query)
            if not sub_qs:
                # All sub-questions tagged foreign_law — abstain.
                return self._abstain(
                    "decomposer_only_foreign_law",
                    routed=routed_ids,
                    sub_qs=[],
                    sub_calls=sub_calls,
                )

        # 3. Per-sub-q retrieval + verification
        # accumulator keyed on (doc_id, canonical_ref) → best citation dict
        accumulated: dict[tuple[str, str], dict[str, Any]] = {}
        # Fix-MH: count how many DISTINCT sub-Qs cite each (doc, ref). True
        # multi-hop gold sits on the intersection of sub-questions, so the
        # consensus count is a stronger signal than verifier confidence
        # alone (the latter is roughly uniform across adjacent legal
        # articles inside the same code).
        consensus_sq: dict[tuple[str, str], set[str]] = {}
        any_retrieval = False
        sub_q_traces: list[dict[str, Any]] = []

        for sq in sub_qs:
            sq_text = sq.get("text", "").strip()
            if not sq_text:
                continue

            # R9.6: per-sub-q ``target_docs`` from the plan supervisor
            # narrow the retrieval pool further than the doc-router
            # alone. Empty/missing target_docs falls back to the
            # routed_ids set.
            sq_targets = sq.get("target_docs") or []
            sq_routed = list(sq_targets) if sq_targets else routed_ids
            fused = self._fused_candidates(sq_text, sq_routed)
            sq_trace: dict[str, Any] = {
                "id": sq.get("id", ""),
                "text": sq_text,
                "candidates": len(fused),
                "verified": 0,
                "target_docs": list(sq_targets),
            }

            if not fused:
                sub_q_traces.append(sq_trace)
                continue
            any_retrieval = True

            # Verify the most promising candidates first.
            for cand in fused[: self._verify_top_n]:
                cand_article = self._candidate_to_article(cand)
                try:
                    verdict = self._verifier_fn(
                        self._llm_pool, sq_text, cand_article, self._sub_model
                    )
                    sub_calls += 1
                except Exception as exc:
                    log.warning("multi_hop verifier failed (%s) — skipping candidate", exc)
                    continue

                if not verdict.get("relevant"):
                    continue
                conf = float(verdict.get("confidence", 0.0) or 0.0)
                if conf < self._verify_threshold:
                    continue

                key = (cand_article["doc_id"], cand_article["article_ref"])
                supporting_quote = verdict.get("supporting_span") or ""
                citation = self._build_citation(
                    cand_article, supporting_quote=supporting_quote, confidence=conf
                )
                # Keep the highest-confidence verdict if the same article comes
                # back from multiple sub-questions.
                prior = accumulated.get(key)
                if prior is None or conf > float(prior.get("confidence", 0.0)):
                    accumulated[key] = citation
                # Fix-MH: record which DISTINCT sub-Q surfaced this article.
                # Same sub-Q hitting twice doesn't count as consensus.
                consensus_sq.setdefault(key, set()).add(sq.get("id", sq_text[:40]))
                sq_trace["verified"] += 1

            sub_q_traces.append(sq_trace)

        if not any_retrieval:
            return self._abstain(
                "no_hits",
                routed=routed_ids,
                sub_qs=sub_q_traces,
                sub_calls=sub_calls,
            )
        if not accumulated:
            return self._abstain(
                "no_verified_articles",
                routed=routed_ids,
                sub_qs=sub_q_traces,
                sub_calls=sub_calls,
            )

        # 4. Aggregate + truncate to final_top_k
        # Fix-MH: re-rank by (confidence + consensus_boost). consensus_boost
        # = DEFAULT_CONSENSUS_BOOST * (n_sub_qs_citing - 1) so:
        #   - articles cited by 1 sub-Q  → no boost (legacy behaviour)
        #   - articles cited by 2 sub-Qs → +DEFAULT_CONSENSUS_BOOST
        #   - articles cited by 3 sub-Qs → +2*DEFAULT_CONSENSUS_BOOST
        # The article on the multi-hop chain is the intersection point; an
        # adjacent-but-wrong article (HANDOFF §R2: civ 409 vs gold 408)
        # rarely gets cited by more than one sub-Q.
        # We also stamp the consensus into the citation telemetry so we can
        # audit which articles won by consensus vs raw confidence.
        ranked_with_score: list[tuple[float, dict[str, Any]]] = []
        for key, cit in accumulated.items():
            base_conf = float(cit.get("confidence", 0.0))
            consensus_n = len(consensus_sq.get(key, ())) or 1
            boost = float(self._consensus_boost) * max(0, consensus_n - 1)
            final_score = base_conf + boost
            cit["consensus_sub_qs"] = consensus_n
            cit["consensus_boost"] = boost
            ranked_with_score.append((final_score, cit))
        ranked_with_score.sort(key=lambda kv: kv[0], reverse=True)
        ranked = [cit for _, cit in ranked_with_score]
        final_citations = ranked[: self._final_top_k]

        # 4b. R9.5 supervisor (smart-trigger). Re-rank with the strong
        # model when the top verifier confidence is in the uncertainty
        # band — exactly the regime multi_hop spends most of its budget
        # in (cross-doc adjacencies that the small Qwen verifier can't
        # discriminate).
        supervisor_used = False
        if self._supervisor_fn is not None and should_supervise(final_citations):
            try:
                supervised = self._supervisor_fn(
                    self._llm_pool, query, final_citations,
                )
                sub_calls += 1
                if supervised:
                    final_citations = supervised
                supervisor_used = True
            except Exception as exc:
                log.warning(
                    "multi_hop supervisor failed (%s) — keeping pre-supervisor citations",
                    exc,
                )

        if not final_citations:
            return self._abstain(
                "supervisor_dropped_all",
                routed=routed_ids,
                sub_qs=sub_q_traces,
                sub_calls=sub_calls,
            )

        # 5. Synthesise
        answer_text = self._template_answer(final_citations)
        try:
            synth = self._summarizer_fn(
                self._llm_pool, query, final_citations, self._sub_model
            )
            sub_calls += 1
            summary = synth.get("summary")
            if isinstance(summary, str) and summary.strip():
                answer_text = summary.strip()
        except Exception as exc:
            log.warning("multi_hop summariser failed (%s) — using template answer", exc)

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   [t["text"] for t in sub_q_traces if t.get("text")],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 1,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed_ids,
                "sub_questions":   sub_q_traces,
                "sub_call_count":  sub_calls,
                "max_sub_calls":   self._max_sub_calls,
                "supervisor_used": supervisor_used,
                "plan_supervisor_used": plan_supervisor_used,
            },
        }

    # ------------------------------------------------------------------
    # Decomposition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_content_tokens(query: str) -> int:
        if not query:
            return 0
        return sum(1 for tok in query.split() if len(tok) >= 2)

    def _extract_sub_questions(self, decomp: dict, query: str) -> list[dict[str, Any]]:
        """Pull sub-questions out of decomposer output, drop foreign_law, cap count.

        Falls back to a single sub-question (the original query) when the
        decomposer output is missing or empty.
        """
        raw = []
        if isinstance(decomp, dict):
            raw = decomp.get("sub_questions") or []
        kept: list[dict[str, Any]] = []
        for sq in raw:
            if not isinstance(sq, dict):
                continue
            if (sq.get("type") or "").lower() == "foreign_law":
                continue
            text = (sq.get("text") or "").strip()
            if not text:
                continue
            kept.append({
                "id":   sq.get("id") or f"sq{len(kept) + 1}",
                "text": text,
                "type": sq.get("type") or "rule_application",
            })
            if len(kept) >= self._max_sub_qs:
                break

        if not kept and not raw:
            # Decomposer returned nothing usable AT ALL — fall back to the
            # original query as a single sub-question. We deliberately do NOT
            # do this when raw is non-empty (e.g. the only sub-questions were
            # foreign_law) — that path should trigger the abstention.
            kept.append({"id": "sq0", "text": query, "type": "rule_application"})

        return kept

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _fused_candidates(
        self, sub_question: str, routed_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Fuse BM25 + Dense for a sub-question and post-filter by routed docs."""
        try:
            bm25_hits: list[BM25Hit] = self._bm25.search(sub_question, k=self._k_each)
        except Exception as exc:
            log.warning("multi_hop BM25 search failed: %s", exc)
            bm25_hits = []
        try:
            dense_hits: list[DenseHit] = self._dense.search(sub_question, k=self._k_each)
        except Exception as exc:
            log.warning("multi_hop dense search failed: %s", exc)
            dense_hits = []

        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        dense_dicts = self._hits_to_dicts(dense_hits, retriever="dense")
        if not bm25_dicts and not dense_dicts:
            return []

        fused = rrf_fuse([bm25_dicts, dense_dicts])
        if routed_ids:
            allowed = set(routed_ids)
            filtered = [h for h in fused if h.get("doc_id") in allowed]
            # If filtering wipes out every candidate, fall back to the
            # unrestricted fused list — better to retrieve from the whole
            # corpus than to return nothing.
            if filtered:
                fused = filtered

        return fused[: self._top_k_per_subq]

    @staticmethod
    def _hits_to_dicts(
        hits: list[BM25Hit] | list[DenseHit],
        *,
        retriever: str,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for h in hits:
            ref_canon = canonical_article_ref(h.article_ref) or h.article_ref
            out.append({
                "chunk_id":    h.chunk_id,
                "doc_id":      h.doc_id,
                "article_ref": ref_canon,
                "text":        h.text or "",
                "score":       float(h.score),
                "retriever":   retriever,
            })
        return out

    def _candidate_to_article(self, cand: dict[str, Any]) -> dict[str, Any]:
        """Shape a fused candidate into the dict expected by call_verifier."""
        doc_id = cand.get("doc_id", "")
        ref_canon = canonical_article_ref(cand.get("article_ref", "")) or cand.get(
            "article_ref", ""
        )
        return {
            "doc_id":      doc_id,
            "article_ref": ref_canon,
            "text":        cand.get("text", "") or "",
            "doc_title":   self._doc_title(doc_id),
            "score":       float(cand.get("score", 0.0)),
        }

    # ------------------------------------------------------------------
    # Citation / answer assembly
    # ------------------------------------------------------------------

    def _build_citation(
        self,
        article: dict[str, Any],
        *,
        supporting_quote: str,
        confidence: float,
    ) -> dict[str, Any]:
        text = article.get("text", "") or ""
        # Prefer the verifier's exact supporting quote when it really is a
        # substring of the article text — the span-existence gate later in
        # the pipeline will demand that. Fall back to text[:280] otherwise.
        if supporting_quote and supporting_quote in text:
            span = supporting_quote[:SUPPORT_SPAN_LEN]
        else:
            span = text[:SUPPORT_SPAN_LEN]
        return {
            "doc_id":            article["doc_id"],
            "article_ref":       article["article_ref"],
            "doc_title":         article.get("doc_title", "") or article["doc_id"],
            "supporting_span":   span,
            "text":              text,
            "confidence":        float(confidence),
            "verifier_relevant": True,
        }

    def _doc_title(self, doc_id: str) -> str:
        try:
            entry = self._registry.get_doc(doc_id)
        except Exception:
            entry = None
        return getattr(entry, "doc_title", "") or doc_id

    @staticmethod
    def _template_answer(citations: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in citations:
            doc_title = c.get("doc_title") or c.get("doc_id", "")
            ref = c.get("article_ref", "")
            text = c.get("supporting_span") or c.get("text", "")
            parts.append(f"وفقًا لـ {doc_title}، المادة {ref}: {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def _abstain(
        self,
        reason: str,
        *,
        routed: list[str],
        sub_qs: list,
        sub_calls: int,
        plan_supervisor_used: bool = False,
        supervisor_used: bool = False,
    ) -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":    0,
                "gate_results":   {},
                "baseline":       TELEMETRY_BASELINE,
                "routed_doc_ids": routed,
                "sub_questions":  sub_qs,
                "sub_call_count": sub_calls,
                "max_sub_calls":  self._max_sub_calls,
                "plan_supervisor_used": plan_supervisor_used,
                "supervisor_used": supervisor_used,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_multi_hop_handler(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> MultiHopHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return MultiHopHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
