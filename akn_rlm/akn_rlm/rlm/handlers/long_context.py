"""Long-context query handler — Phase 2 / R6.4.

Pipeline (per HANDOFF §3):

  1. Doc-route the query via :class:`DocRouter` (alias + numeric-id +
     BM25 channels) to get 1-3 likely ``doc_id`` predictions.

  2. **Broad** RRF-fuse(BM25, Dense) at ``k_each=20`` (the
     "broad hybrid k=20" called out in HANDOFF §3 — long_context
     queries demand multiple articles per query, so we deliberately
     widen the retrieval pool relative to rule_application's k=30
     per-side / final top-K=8). Restrict to the routed docs (with
     full-pool fallback when filtering wipes everything).

  3. Take top-``final_top_k`` candidates from the fused list. Default
     10 because long_context gold has 4-6 articles per question; we
     want headroom above the gold-set size so the summariser can pick.

  4. **Real summariser sub-LM call** — this is the discriminator
     HANDOFF §3 calls out: "current pipeline doesn't actually call
     summarize". We pass the full top-K candidate texts to
     :func:`call_summarizer` so the answer is a synthesised
     multi-article narrative rather than a single-article template.
     If the summariser returns ``null`` or raises, the handler falls
     back to the deterministic Arabic template used by every other
     handler.

  5. Citations = top-K candidates as-is (no verifier — long_context
     specifically benefits from broader recall, not tighter
     filtering, and the existing baselines all score HCR=0 on this
     stratum so the risk of fabricated citations is low).

Sub-LM call budget per query: **1 summariser call**, well under the
project ``max_sub_calls=12`` envelope and matches HANDOFF §3's
"others=2" budget.

The handler is self-contained (no LangGraph, no ``RootController``)
and is callable as a baseline-shaped pipeline.
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
from akn_rlm.rlm.sub_worker import call_summarizer
from akn_rlm.rlm.supervisor import SupervisorFn, should_supervise

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# HANDOFF §3 says "broad hybrid k=20". We interpret that as k_each=20
# per retriever (so the fused pool is up to 40 distinct chunks before
# dedup) and final_top_k=10 to feed the summariser the breadth it needs
# for a real narrative answer.
DEFAULT_K_EACH: int = 20
# R9.4: final_top_k 10 → 6 to lift Cite F1 by trading per-citation
# precision against the recall-side wins long_context already showed
# (MRR doc 0.833, Doc Cite F1 0.590, JIR 0.059 best-of-3). Six
# citations still exceeds typical gold-set size (4-6 per query) so
# the summariser still has the breadth it needs for a real narrative.
DEFAULT_FINAL_TOP_K: int = 6
#: Fix-LC: after RRF, expand each top-seed with up to N siblings in the
#: same AKN chapter/section. Long-context queries usually want a whole
#: section ("all articles about X in chapter Y"); pure relevance RRF
#: catches 2-3 of 4-6 gold articles, neighbors close the gap.
DEFAULT_CHAPTER_EXPANSION: bool = False   # opt-in via dispatcher / kwarg
DEFAULT_SEEDS_FOR_EXPANSION: int = 4
DEFAULT_NEIGHBORS_PER_SEED: int = 2
#: Final top_K is bumped when chapter expansion is on so the summariser
#: gets the broader section coverage.
DEFAULT_FINAL_TOP_K_WITH_EXPANSION: int = 12
#: When chapter expansion is on, cap the first-pass RRF dedup at this
#: many seeds. The remaining final_top_k - seed_cap slots are reserved
#: for chapter neighbors. Bug fix: without a cap the dedup loop fills
#: the entire final_top_k from raw RRF and the expansion adds 0 entries.
DEFAULT_RRF_SEED_CAP_WITH_EXPANSION: int = 8
DEFAULT_ROUTE_TOP_N: int = 3
SUPPORT_SPAN_LEN: int = 280

TELEMETRY_BASELINE: str = "rlm_long_context"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

SummarizerFn = Callable[[Any, str, list[dict], str], dict]


class LongContextHandler:
    """Typed long-context handler: route -> broad hybrid k=20 ->
    summariser sub-LM call -> top-K citations."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        k_each: int = DEFAULT_K_EACH,
        route_top_n: int = DEFAULT_ROUTE_TOP_N,
        summarizer_fn: Optional[SummarizerFn] = None,
        supervisor_fn: Optional[SupervisorFn] = None,
        # Fix-LC: AKN chapter/section neighbor expansion. When on, the
        # handler bumps final_top_k to 10 and unions each of the top-N
        # RRF seeds with their chapter siblings.
        enable_chapter_expansion: bool = DEFAULT_CHAPTER_EXPANSION,
        seeds_for_expansion: int = DEFAULT_SEEDS_FOR_EXPANSION,
        neighbors_per_seed: int = DEFAULT_NEIGHBORS_PER_SEED,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._sub_model = sub_model
        # Fix-LC: when chapter expansion is on, we widen final_top_k so the
        # summariser sees the full section. Caller-supplied final_top_k
        # overrides this default if explicitly passed.
        if enable_chapter_expansion and final_top_k == DEFAULT_FINAL_TOP_K:
            self._final_top_k = DEFAULT_FINAL_TOP_K_WITH_EXPANSION
        else:
            self._final_top_k = final_top_k
        self._k_each = k_each
        self._route_top_n = route_top_n
        self._summarizer_fn = summarizer_fn or call_summarizer
        # Fix-LC config + lazy-loaded article->ancestors map.
        self._enable_chapter_expansion = bool(enable_chapter_expansion)
        self._seeds_for_expansion = int(seeds_for_expansion)
        self._neighbors_per_seed = int(neighbors_per_seed)
        self._ancestor_map: Optional[dict[tuple[str, str], dict[str, str]]] = None
        # R9.5: optional gpt-oss-120b per-citation re-ranker. Trigger
        # is unlikely to fire on the LC path (citations are scored by
        # RRF, which sits well below 0.30), but the seam is wired so
        # an explicit override can opt in.
        self._supervisor_fn = supervisor_fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain(
                "empty_query", routed=[], top_score=0.0,
                candidates=0, sub_calls=0,
            )

        # 1. Doc-route
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)

        # 2. Broad RRF(BM25, Dense) at k_each=20, restricted to routed docs.
        candidates = self._fused_candidates(query, routed_ids)
        if not candidates:
            return self._abstain(
                "no_hits", routed=routed_ids, top_score=0.0,
                candidates=0, sub_calls=0,
            )

        # 3. Take top-K seeds. When chapter expansion is on, cap the
        # first-pass dedup at ``seed_cap`` so there are slots left for
        # neighbors. Without the cap, the loop fills the entire
        # final_top_k from raw RRF and expansion has nothing to add.
        if self._enable_chapter_expansion:
            seed_cap = DEFAULT_RRF_SEED_CAP_WITH_EXPANSION
        else:
            seed_cap = self._final_top_k
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for cand in candidates:
            key = (cand["doc_id"], cand["article_ref"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(cand)
            if len(deduped) >= seed_cap:
                break

        # Fix-LC: chapter-neighbor expansion. For each of the top-N RRF
        # seeds, find articles in the same AKN chapter/section and union
        # them into the pool. Long-context gold sets are usually whole
        # sections (4-6 articles); broad RRF alone tends to catch 2-3 of
        # them, the neighbors close the gap.
        if self._enable_chapter_expansion:
            expanded = self._expand_with_chapter_neighbors(deduped)
            for nbr in expanded:
                key = (nbr["doc_id"], nbr["article_ref"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(nbr)
                if len(deduped) >= self._final_top_k:
                    break

        top_score = float(deduped[0]["score"]) if deduped else 0.0

        # 4. Build citations from the top-K candidates (no verifier).
        final_citations = [self._build_citation(c) for c in deduped]

        sub_calls = 0
        # 4b. R9.5 supervisor (smart-trigger). Only fires if RRF top
        # confidence happens to fall in the [0.30, 0.70] band — rare
        # for LC, but the seam is wired for completeness.
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
                    "long_context supervisor failed (%s) — keeping pre-supervisor citations",
                    exc,
                )

        # 5. **Real** summariser call (the discriminator HANDOFF §3 names).
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
            log.warning(
                "long_context summariser failed (%s) — using template", exc
            )

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   [
                f"routed_doc_ids={routed_ids}",
                f"final_top_k={len(final_citations)}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 1,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed_ids,
                "top_score":       top_score,
                "candidate_count": len(deduped),
                "sub_call_count":  sub_calls,
                "supervisor_used": supervisor_used,
            },
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _fused_candidates(
        self, query: str, routed_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Broad RRF(BM25, Dense) restricted to routed docs."""
        try:
            bm25_hits: list[BM25Hit] = self._bm25.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("long_context BM25 failed: %s", exc)
            bm25_hits = []
        try:
            dense_hits: list[DenseHit] = self._dense.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("long_context dense failed: %s", exc)
            dense_hits = []

        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        dense_dicts = self._hits_to_dicts(dense_hits, retriever="dense")
        if not bm25_dicts and not dense_dicts:
            return []

        fused = rrf_fuse([bm25_dicts, dense_dicts])
        if routed_ids:
            allowed = set(routed_ids)
            filtered = [h for h in fused if h.get("doc_id") in allowed]
            if filtered:
                fused = filtered
        return fused

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

    # ------------------------------------------------------------------
    # Citation / answer assembly
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fix-LC — chapter neighbor expansion
    # ------------------------------------------------------------------

    def _load_ancestor_map(self) -> dict[tuple[str, str], dict[str, str]]:
        """Lazy-build (doc_id, article_ref) -> ancestors map by re-parsing
        the corpus. One-time ~0.7 s cost on first dispatched LC question.
        Subsequent dispatches reuse the cached map.
        """
        if self._ancestor_map is not None:
            return self._ancestor_map
        try:
            from akn_rlm.corpus.akn_parser import parse_all
            from akn_rlm.normalizers import canonical_article_ref as _canon
        except Exception as exc:
            log.warning("Fix-LC ancestor map load failed: %s", exc)
            self._ancestor_map = {}
            return self._ancestor_map

        amap: dict[tuple[str, str], dict[str, str]] = {}
        # Also keep an index of (doc_id, chapter, section) -> [refs] so the
        # neighbor lookup is O(siblings) not O(corpus). The "chapter+section"
        # key is the right grouping: long_context queries asking "all
        # provisions about X" usually want one section (or one chapter
        # when sections are absent).
        sibling_idx: dict[tuple[str, str, str], list[str]] = {}
        try:
            for art in parse_all():
                anc = dict(getattr(art, "ancestors", {}) or {})
                ref = _canon(art.article_ref) or art.article_ref
                key = (art.doc_id, ref)
                amap[key] = anc
                grp_key = (art.doc_id, anc.get("chapter", ""), anc.get("section", ""))
                sibling_idx.setdefault(grp_key, []).append(ref)
        except Exception as exc:
            log.warning("Fix-LC parse_all raised: %s", exc)

        self._ancestor_map = amap
        # Stash sibling_idx on self for the lookup path
        self._sibling_idx = sibling_idx
        return amap

    def _expand_with_chapter_neighbors(
        self,
        seeds: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return neighbor candidates for the top-N seeds in the same
        chapter / section, ordered by seed rank then by article_ref.
        """
        amap = self._load_ancestor_map()
        if not amap:
            return []
        sibling_idx = getattr(self, "_sibling_idx", {}) or {}
        if not sibling_idx:
            return []

        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = {(s["doc_id"], s["article_ref"]) for s in seeds}

        for seed in seeds[: self._seeds_for_expansion]:
            doc_id = seed.get("doc_id", "")
            ref = seed.get("article_ref", "")
            anc = amap.get((doc_id, ref))
            if not anc:
                continue
            grp_key = (doc_id, anc.get("chapter", ""), anc.get("section", ""))
            siblings = sibling_idx.get(grp_key, [])
            if len(siblings) <= 1:
                continue   # singleton chapter/section — no siblings to add
            added = 0
            for sib_ref in siblings:
                if added >= self._neighbors_per_seed:
                    break
                if (doc_id, sib_ref) in seen:
                    continue
                seen.add((doc_id, sib_ref))
                # Build a neighbor candidate with damped score so the
                # original RRF order is preserved when truncating.
                out.append({
                    "doc_id":      doc_id,
                    "article_ref": sib_ref,
                    "text":        self._lookup_chunk_text(doc_id, sib_ref),
                    "score":       float(seed.get("score", 0.5)) * 0.7,
                    "kg_first":    False,
                    "chapter_neighbor": True,
                })
                added += 1
        return out

    def _lookup_chunk_text(self, doc_id: str, ref: str) -> str:
        """Best-effort article text lookup via BM25 chunk meta. Falls back
        to empty string when the chunk isn't indexed (e.g. the article
        was empty / repealed at parse time and chunker dropped it).
        """
        # BM25Index stores chunks in self._chunks keyed by chunk_id.
        # chunk_id pattern: "{doc_id}#{eid}". We don't have eid here, so
        # scan by (doc_id, ref). This is O(chunks) but only fires on
        # ~2-4 added neighbors per LC question.
        try:
            chunks = getattr(self._bm25, "_chunks", None) or []
            for c in chunks:
                if getattr(c, "doc_id", "") == doc_id and \
                   canonical_article_ref(getattr(c, "article_ref", "")) == ref:
                    return getattr(c, "text", "") or getattr(c, "text_norm", "") or ""
        except Exception:
            pass
        return ""

    def _build_citation(self, candidate: dict[str, Any]) -> dict[str, Any]:
        doc_id = candidate.get("doc_id", "")
        ref = canonical_article_ref(candidate.get("article_ref", "")) or \
            candidate.get("article_ref", "")
        text = candidate.get("text", "") or ""
        return {
            "doc_id":          doc_id,
            "article_ref":     ref,
            "doc_title":       self._doc_title(doc_id),
            "supporting_span": text[:SUPPORT_SPAN_LEN],
            "text":            text,
            "confidence":      float(candidate.get("score", 0.0)),
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
    @staticmethod
    def _abstain(
        reason: str,
        *,
        routed: list[str],
        top_score: float,
        candidates: int,
        sub_calls: int,
    ) -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [
                f"routed_doc_ids={routed}",
                f"reason={reason}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed,
                "top_score":       top_score,
                "candidate_count": candidates,
                "sub_call_count":  sub_calls,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_long_context_handler(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> LongContextHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return LongContextHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
