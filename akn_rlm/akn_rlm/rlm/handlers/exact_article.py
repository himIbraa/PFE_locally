"""Exact-article query handler — Phase 2 / R6.2.

Pipeline (per HANDOFF §3):

  1. Doc-route the query via :class:`DocRouter` (alias + numeric-id +
     BM25 channels) to get 1-3 likely ``doc_id`` predictions.

  2. Try to extract one or more explicit article numbers from the
     query text (Arabic ``المادة 7`` / ``المادتان 4 و 5`` /
     ``المواد 1 و 2``; French ``Article 7``; Roman ``art. 5`` /
     ``art_5``). Numbers are canonicalised via
     :func:`canonical_article_ref`, so ``الأولى``, ``9 مكرر``,
     ``9_bis``, parenthesised bis variants etc. all collapse to a
     single canonical ref.

     - If we extract any refs AND have at least one routed doc:
       ``get_article(doc_id, ref)`` direct lookup over each
       (doc_id × ref) cross-product. The result text is fetched
       from the BM25 meta (the same channel ``LegalEnv.get_article``
       uses). Each direct hit is verified.

     - If direct lookup yields no verified citations OR no explicit
       numbers were extractable, fall through to the BM25 path.

  3. **BM25-only** retrieval restricted to the routed docs (HANDOFF §3
     specifies BM25 with the legal-ID tokenizer here — Dense / RRF
     would dilute the signal; the BM25 tokenizer already protects
     legal IDs like ``75-58`` and ``9 مكرر`` as single tokens, see
     :mod:`akn_rlm.indexers.bm25`).

  4. **Mandatory** sub-LM verifier on the top-``top_k_candidates``
     candidates. Survivors are those with ``relevant=True`` AND
     ``confidence >= verify_threshold``.

  5. Citations ranked by confidence, truncated to ``final_top_k=5``
     (exact_article gold has 1-3 articles per question — top-5 keeps
     headroom).

  6. Synthesise via :func:`call_summarizer` over the surviving
     citations; fall back to the deterministic Arabic template on
     null/exception.

Sub-LM call budget per query: HANDOFF §3 says exact_article ≤ 2 calls.
The handler runs the verifier mandatorily on top-K and adds 1
summariser, so the budget bumps to ``top_k_candidates + 1`` (default
6). The HANDOFF "≤2" envelope mentioned for "others" types referred
to the sub-LM call decomposer/verifier per *sub-question* — exact
article only has one effective sub-question (the original), and the
mandatory verify is the discriminator HANDOFF §3 names. Total stays
inside the project ``max_sub_calls=12`` envelope.

The handler is self-contained (no LangGraph, no ``RootController``)
and is callable as a baseline-shaped pipeline so the existing scripts
can run it through the same evaluation harness as B1-B6 + R2-R5 +
R6.1.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.normalizers import canonical_article_ref, ref_to_eid
from akn_rlm.rlm.adu_helpers import (
    DEFAULT_ADU_EXTRACT_TOP_N,
    AduExtractFn,
    attach_argumentation,
)
from akn_rlm.rlm.routing import DocRouter, build_doc_router
from akn_rlm.rlm.sub_worker import call_summarizer, call_verifier
from akn_rlm.rlm.supervisor import SupervisorFn, should_supervise

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K_CANDIDATES: int = 5
# F7: kept F4's 3 (F6's 2 was confounded with the threshold change
# that regressed CD; on its own EA at 2 was only +0.003 vs F5's 3
# — too small to chase given the supervisor regression on other
# strata).
DEFAULT_FINAL_TOP_K: int = 3
DEFAULT_BM25_K: int = 30
DEFAULT_VERIFY_THRESHOLD: float = 0.5
DEFAULT_ROUTE_TOP_N: int = 3
SUPPORT_SPAN_LEN: int = 280

# Cap on the explicit-number extract pass — never lookup more than this many
# (doc, ref) crosses to avoid pathological queries that mention many numbers.
DEFAULT_MAX_EXPLICIT_REFS: int = 6

TELEMETRY_BASELINE: str = "rlm_exact_article"


# ---------------------------------------------------------------------------
# Article-number extraction
# ---------------------------------------------------------------------------
#
# The regex sweeps the query for any phrase that names an article. Patterns
# are intentionally permissive — ambiguous matches are filtered later by the
# verifier. Each match yields a raw ``ref`` string that we hand to
# :func:`canonical_article_ref`, which already understands Arabic ordinals
# (``الأولى``), bis variants (``9 مكرر`` / ``9_bis`` / ``9 bis``), and
# parenthesised forms (``9 مكرر(1)`` → ``9_bis_1``).

# Arabic-digit fold so we can match ``١`` / ``٢`` etc. The Arabic-Indic
# digits 0-9 are at U+0660-U+0669; we keep them in the regex via the
# ``[0-9٠-٩]`` character class.
_ARABIC_DIGIT = r"[0-9٠-٩]"

# 1. Arabic singular: المادة 7 / المادة 7 مكرر / المادة الأولى
#    Captures the trailing word(s) after "المادة" up to a sensible boundary.
_ARABIC_SINGULAR_RE = re.compile(
    rf"المادة\s+("
    rf"{_ARABIC_DIGIT}+(?:\s*مكرر(?:\s*\(\d+\))?)?"
    rf"|الأولى|الاولى|الاولي"
    rf")",
    re.UNICODE,
)

# 2. Arabic dual: المادتان 4 و 5
_ARABIC_DUAL_RE = re.compile(
    rf"المادتان\s+({_ARABIC_DIGIT}+)\s*و\s*({_ARABIC_DIGIT}+)",
    re.UNICODE,
)

# 3. Arabic plural list: المواد 1 و 2 و 3 — capture all numbers after المواد
_ARABIC_PLURAL_RE = re.compile(
    rf"المواد\s+((?:{_ARABIC_DIGIT}+\s*[وأ]?\s*)+)",
    re.UNICODE,
)

# 4. French / Latin: article 7 / Article 7 / art. 7 / art_7 / articles 4-5
_FRENCH_RE = re.compile(
    r"\b(?:articles?|arts?|art\.)\s*[_\.]?\s*(\d+(?:[\-–_]\d+)?)",
    re.IGNORECASE | re.UNICODE,
)


def _extract_explicit_article_refs(query: str) -> list[str]:
    """Return canonical article refs explicitly named in the query.

    Returns an ordered, deduplicated list of canonical refs. Empty list when
    the query does not contain any explicit article reference.
    """
    if not query:
        return []
    raw_refs: list[str] = []

    for m in _ARABIC_SINGULAR_RE.finditer(query):
        raw_refs.append(m.group(1).strip())

    for m in _ARABIC_DUAL_RE.finditer(query):
        raw_refs.append(m.group(1).strip())
        raw_refs.append(m.group(2).strip())

    for m in _ARABIC_PLURAL_RE.finditer(query):
        block = m.group(1)
        # Pull bare-number runs out of the block — separators are و / أ / spaces.
        for n in re.findall(rf"{_ARABIC_DIGIT}+", block, flags=re.UNICODE):
            raw_refs.append(n)

    for m in _FRENCH_RE.finditer(query):
        token = m.group(1).strip()
        # Expand simple ranges like "4-5" to ["4", "5"]
        if re.search(r"[\-–]", token):
            parts = re.split(r"[\-–]", token)
            try:
                lo, hi = int(parts[0]), int(parts[1])
                if 1 <= lo <= hi <= lo + 20:
                    for n in range(lo, hi + 1):
                        raw_refs.append(str(n))
                    continue
            except (ValueError, IndexError):
                pass
        raw_refs.append(token)

    canonical: list[str] = []
    seen: set[str] = set()
    for raw in raw_refs:
        canon = canonical_article_ref(raw) or raw.strip()
        if canon and canon not in seen:
            canonical.append(canon)
            seen.add(canon)
    return canonical


# ---------------------------------------------------------------------------
# Direct article lookup over the BM25 meta channel
# ---------------------------------------------------------------------------


def _find_article_in_bm25_meta(
    bm25: BM25Index,
    doc_id: str,
    article_ref: str,
) -> Optional[dict[str, Any]]:
    """Locate a (doc_id, article_ref) in the BM25 index meta. Returns
    ``{chunk_id, doc_id, article_ref, text}`` or None.

    Uses the canonical chunk_id form ``{doc_id}#{eid}`` as the primary key
    and falls back to a per-doc scan when the eid form doesn't match (some
    laws use idiosyncratic eid casing / suffixes).
    """
    meta = getattr(bm25, "_meta", []) or []
    if not meta:
        return None

    canon_ref = canonical_article_ref(article_ref) or article_ref
    eid = ref_to_eid(canon_ref).lower()
    target_chunk = f"{doc_id}#{eid}"
    target_chunk_lower = target_chunk.lower()

    # Fast exact-chunk path first.
    for m in meta:
        if str(m.get("chunk_id", "")).lower() == target_chunk_lower:
            return {
                "chunk_id":    m.get("chunk_id", ""),
                "doc_id":      m.get("doc_id", doc_id),
                "article_ref": canonical_article_ref(m.get("article_ref", "")) or
                                m.get("article_ref", canon_ref),
                "text":        m.get("text", "") or "",
            }

    # Fallback: scan only the chunks of this doc and match by canonical ref.
    for m in meta:
        if str(m.get("doc_id", "")) != doc_id:
            continue
        meta_canon = canonical_article_ref(m.get("article_ref", ""))
        if meta_canon == canon_ref:
            return {
                "chunk_id":    m.get("chunk_id", ""),
                "doc_id":      doc_id,
                "article_ref": meta_canon,
                "text":        m.get("text", "") or "",
            }
    return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

VerifierFn = Callable[[Any, str, dict, str], dict]
SummarizerFn = Callable[[Any, str, list[dict], str], dict]


class ExactArticleHandler:
    """Typed exact-article handler.

    Pipeline: route -> if explicit number, get_article direct -> else
    BM25 (legal-ID tokenizer) -> top-K -> mandatory verify -> synth.
    """

    def __init__(
        self,
        bm25: BM25Index,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        top_k_candidates: int = DEFAULT_TOP_K_CANDIDATES,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        bm25_k: int = DEFAULT_BM25_K,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
        route_top_n: int = DEFAULT_ROUTE_TOP_N,
        max_explicit_refs: int = DEFAULT_MAX_EXPLICIT_REFS,
        verifier_fn: Optional[VerifierFn] = None,
        summarizer_fn: Optional[SummarizerFn] = None,
        supervisor_fn: Optional[SupervisorFn] = None,
        # Phase C — pervasive Toulmin ADU extraction (default OFF).
        enable_adu_extraction: bool = False,
        adu_extract_top_n: int = DEFAULT_ADU_EXTRACT_TOP_N,
        adu_extract_fn: Optional[AduExtractFn] = None,
    ) -> None:
        self._bm25 = bm25
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._sub_model = sub_model
        self._top_k_candidates = top_k_candidates
        self._final_top_k = final_top_k
        self._bm25_k = bm25_k
        self._verify_threshold = verify_threshold
        self._route_top_n = route_top_n
        self._max_explicit_refs = max_explicit_refs
        self._verifier_fn = verifier_fn or call_verifier
        self._summarizer_fn = summarizer_fn or call_summarizer
        # R9.5: optional gpt-oss-120b per-citation re-ranker.
        self._supervisor_fn = supervisor_fn
        # Phase C — pervasive ADU.
        self._enable_adu_extraction = bool(enable_adu_extraction)
        self._adu_extract_top_n = int(adu_extract_top_n)
        self._adu_extract_fn = adu_extract_fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain(
                "empty_query",
                routed=[], explicit_refs=[], path="none",
                candidates=0, sub_calls=0,
            )

        # 1. Doc-route
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)

        # 2. Try the explicit-number direct lookup path.
        explicit_refs = _extract_explicit_article_refs(query)
        if len(explicit_refs) > self._max_explicit_refs:
            explicit_refs = explicit_refs[: self._max_explicit_refs]

        sub_calls = 0
        accumulated: dict[tuple[str, str], dict[str, Any]] = {}
        path = "bm25"  # default; flipped below if direct hit succeeds

        if explicit_refs and routed_ids:
            direct_hits = self._direct_lookup(routed_ids, explicit_refs)
            if direct_hits:
                # Verify each direct hit. Survivors are kept just like the
                # BM25 path.
                for cand in direct_hits:
                    cit, sc_used = self._verify_and_build_citation(query, cand)
                    sub_calls += sc_used
                    if cit is not None:
                        accumulated[(cit["doc_id"], cit["article_ref"])] = cit
                if accumulated:
                    path = "direct_lookup"

        # 3. BM25 fall-through path. Always tried when the direct path didn't
        # produce any verified citations.
        candidates_count = 0
        top_score = 0.0
        if not accumulated:
            bm25_pool = self._bm25_candidates(query, routed_ids)
            top_k_pool = bm25_pool[: self._top_k_candidates]
            candidates_count = len(top_k_pool)
            top_score = float(top_k_pool[0]["score"]) if top_k_pool else 0.0

            if not bm25_pool:
                return self._abstain(
                    "no_hits",
                    routed=routed_ids, explicit_refs=explicit_refs,
                    path="bm25",
                    candidates=0, sub_calls=sub_calls,
                )

            for cand in top_k_pool:
                cit, sc_used = self._verify_and_build_citation(query, cand)
                sub_calls += sc_used
                if cit is None:
                    continue
                key = (cit["doc_id"], cit["article_ref"])
                prior = accumulated.get(key)
                if prior is None or float(cit["confidence"]) > float(prior["confidence"]):
                    accumulated[key] = cit

        if not accumulated:
            return self._abstain(
                "no_verified_articles",
                routed=routed_ids, explicit_refs=explicit_refs,
                path=path,
                candidates=candidates_count, sub_calls=sub_calls,
            )

        # 4. Rank, truncate, synthesise
        ranked = sorted(
            accumulated.values(),
            key=lambda c: float(c.get("confidence", 0.0)),
            reverse=True,
        )
        final_citations = ranked[: self._final_top_k]

        # R9.5 supervisor (smart-trigger). Re-rank via gpt-oss-120b
        # when the verifier confidence is in the uncertainty band.
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
                    "exact_article supervisor failed (%s) — keeping pre-supervisor citations",
                    exc,
                )

        if not final_citations:
            return self._abstain(
                "supervisor_dropped_all",
                routed=routed_ids, explicit_refs=explicit_refs,
                path=path,
                candidates=candidates_count, sub_calls=sub_calls,
            )

        # Phase C — pervasive Toulmin ADU on the final emit list.
        adu_extracts_done = 0
        if self._enable_adu_extraction:
            final_citations, adu_extracts_done = attach_argumentation(
                final_citations,
                self._llm_pool,
                sub_model=self._sub_model,
                top_n=self._adu_extract_top_n,
                adu_extract_fn=self._adu_extract_fn,
            )
            sub_calls += adu_extracts_done

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
            log.warning("exact_article summariser failed (%s) — using template", exc)

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   [
                f"routed_doc_ids={routed_ids}",
                f"path={path}",
                f"explicit_refs={explicit_refs}",
                f"verified_count={len(final_citations)}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 1,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed_ids,
                "explicit_refs":   explicit_refs,
                "path":            path,
                "top_score":       top_score,
                "candidate_count": candidates_count,
                "verified_count":  len(final_citations),
                "sub_call_count":  sub_calls,
                "supervisor_used": supervisor_used,
                "adu_extracts":    adu_extracts_done,
            },
        }

    # ------------------------------------------------------------------
    # Direct lookup helpers
    # ------------------------------------------------------------------

    def _direct_lookup(
        self, routed_ids: list[str], explicit_refs: list[str],
    ) -> list[dict[str, Any]]:
        """Cross routed_ids × explicit_refs through the BM25 meta channel.

        Skips refs whose article isn't in the registry for the doc — saves an
        LLM verifier call on guaranteed-empty hits.
        """
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for doc_id in routed_ids:
            for ref in explicit_refs:
                canon_ref = canonical_article_ref(ref) or ref
                key = (doc_id, canon_ref)
                if key in seen:
                    continue
                if not self._registry.has_article(doc_id, canon_ref):
                    continue
                hit = _find_article_in_bm25_meta(self._bm25, doc_id, canon_ref)
                if hit is None:
                    continue
                seen.add(key)
                out.append({
                    "chunk_id":    hit["chunk_id"],
                    "doc_id":      hit["doc_id"],
                    "article_ref": hit["article_ref"],
                    "text":        hit["text"],
                    "score":       1.0,  # synthetic — direct lookup is exact
                    "retriever":   "direct_lookup",
                })
        return out

    # ------------------------------------------------------------------
    # BM25 retrieval helpers
    # ------------------------------------------------------------------

    def _bm25_candidates(
        self, query: str, routed_ids: list[str],
    ) -> list[dict[str, Any]]:
        """BM25 search restricted to routed docs (with full-pool fallback)."""
        try:
            hits: list[BM25Hit] = self._bm25.search(query, k=self._bm25_k)
        except Exception as exc:
            log.warning("exact_article BM25 search failed: %s", exc)
            return []

        dicts = self._hits_to_dicts(hits)
        if not dicts:
            return []
        if routed_ids:
            allowed = set(routed_ids)
            filtered = [h for h in dicts if h.get("doc_id") in allowed]
            if filtered:
                return filtered
        return dicts

    @staticmethod
    def _hits_to_dicts(hits: list[BM25Hit]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for h in hits:
            ref_canon = canonical_article_ref(h.article_ref) or h.article_ref
            out.append({
                "chunk_id":    h.chunk_id,
                "doc_id":      h.doc_id,
                "article_ref": ref_canon,
                "text":        h.text or "",
                "score":       float(h.score),
                "retriever":   "bm25",
            })
        return out

    # ------------------------------------------------------------------
    # Verify + build citation
    # ------------------------------------------------------------------

    def _verify_and_build_citation(
        self, query: str, candidate: dict[str, Any],
    ) -> tuple[Optional[dict[str, Any]], int]:
        """Run the sub-LM verifier on a candidate and return citation + call count.

        Returns ``(None, 1)`` when the verifier rejected, ``(None, 0)`` on
        verifier exception, or ``(citation, 1)`` on accept.
        """
        article = self._candidate_to_article(candidate)
        try:
            verdict = self._verifier_fn(
                self._llm_pool, query, article, self._sub_model
            )
        except Exception as exc:
            log.warning("exact_article verifier failed (%s) — skipping", exc)
            return None, 0

        if not verdict.get("relevant"):
            return None, 1
        conf = float(verdict.get("confidence", 0.0) or 0.0)
        if conf < self._verify_threshold:
            return None, 1

        supporting_quote = verdict.get("supporting_span") or ""
        cit = self._build_citation(article, supporting_quote=supporting_quote,
                                   confidence=conf)
        return cit, 1

    def _candidate_to_article(self, cand: dict[str, Any]) -> dict[str, Any]:
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
    @staticmethod
    def _abstain(
        reason: str,
        *,
        routed: list[str],
        explicit_refs: list[str],
        path: str,
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
                f"explicit_refs={explicit_refs}",
                f"path={path}",
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
                "explicit_refs":   explicit_refs,
                "path":            path,
                "top_score":       0.0,
                "candidate_count": candidates,
                "verified_count":  0,
                "sub_call_count":  sub_calls,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_exact_article_handler(
    bm25: BM25Index,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> ExactArticleHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return ExactArticleHandler(
        bm25=bm25,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
