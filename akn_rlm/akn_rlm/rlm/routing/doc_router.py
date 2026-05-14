"""Doc-level routing — predict 1-3 relevant doc_ids for a query.

The router is the first step of every Phase-2 RLM handler. It restricts
subsequent retrieval to the predicted document set, drastically reducing
noise from the 51-law corpus.

Design (deterministic by default; LLM channel is optional):

  1. Alias scan — direct match against the static alias map maintained in
     :mod:`akn_rlm.corpus.article_registry` (e.g. "Family Code" /
     "قانون الأسرة" / "cciv" -> 75-58_1975-09-26).  Multi-word and Arabic
     aliases match by substring; short Latin abbreviations require word
     boundaries to avoid false positives.

  2. Numeric law-id scan — bare law identifiers like ``84-11`` or
     ``06-154`` are matched with word-boundary regex and resolved through
     :meth:`ArticleRegistry.resolve_alias`.  A four-digit-prefixed match
     (e.g. ``1984-06-09``) is a date, not a law id, so we cap the leading
     run at three digits.

  3. BM25 aggregation — sum the top-N BM25 scores per doc_id.  This is the
     'soft' channel that picks up the right document even when the law is
     not named explicitly.  Each doc's contribution is capped at
     ``bm25_per_doc_cap`` hits so a single law's many chunks cannot crowd
     out a less verbose competitor.  The resulting per-doc aggregate is
     normalised so its max is 1.0, making it directly comparable to the
     alias bonus across queries.

  4. Optional LLM tie-breaker — if an ``llm_call`` callable is injected,
     the top-N BM25 candidates are passed to the LLM with the query and
     the LLM's chosen ids are added with a fixed bonus.  Off by default
     so :class:`DocRouter` runs at zero LLM cost.

  5. Fusion + ranking — fused score = alias_bonus·(alias-hit count) +
     bm25_weight·normalised-BM25 + llm_bonus·(LLM hit).  Ties are broken
     by lexicographic order of doc_id so the result is fully
     deterministic.

Returns :class:`RouteResult` with the top-N doc_ids ranked by fused
score, the full per-doc score breakdown, and the channels that surfaced
each doc.  Empty queries / empty corpus return an empty result rather
than raising.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Index
from akn_rlm.normalizers import normalize_arabic

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_N: int = 3
DEFAULT_BM25_POOL: int = 100
DEFAULT_BM25_PER_DOC_CAP: int = 5
DEFAULT_ALIAS_BONUS: float = 1.0
DEFAULT_BM25_WEIGHT: float = 1.0
DEFAULT_LLM_BONUS: float = 0.5
#: Phase E.4 — bonus added to a doc_id when the KG channel surfaces
#: it. Matches the LLM tie-breaker's 0.5 — strong enough to flip a
#: BM25 tie but not strong enough to override an explicit alias hit
#: (alias bonus = 1.0).
DEFAULT_KG_BONUS: float = 0.5

# Bare law identifiers like "84-11", "06-154", "75-58".  The leading run is
# capped at three digits so a date such as 1984-06-09 does not match.
_LAW_ID_RE = re.compile(r"(?<![\d-])(\d{2,3}-\d{1,3})(?![\d-])")

# Short Latin aliases (≤5 chars, all-ASCII) need word boundaries to avoid
# matching inside unrelated words like "applicants" or "ample".
_LATIN_RE = re.compile(r"^[a-z0-9._-]{1,5}$")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteResult:
    """Output of :meth:`DocRouter.route`.

    Attributes:
      doc_ids:    ranked doc_ids, length ≤ ``top_n``.
      scores:     full per-doc fused score breakdown for every doc that
                  received any signal (not just the top-N).
      sources:    doc_id → list of channel names ("alias", "bm25", "llm").
      confidence: 1.0 when at least one returned doc came from the alias
                  channel; 0.6 when the result is BM25-only; 0.0 when
                  empty.
    """
    doc_ids: list[str]
    scores: dict[str, float]
    sources: dict[str, list[str]]
    confidence: float


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

LLMCall = Callable[[str, list[str]], list[str]]
KGCall = Callable[[str], list[str]]


class DocRouter:
    """Predict 1-3 relevant doc_ids for a query."""

    def __init__(
        self,
        registry: ArticleRegistry,
        bm25: Optional[BM25Index] = None,
        *,
        top_n: int = DEFAULT_TOP_N,
        bm25_pool: int = DEFAULT_BM25_POOL,
        bm25_per_doc_cap: int = DEFAULT_BM25_PER_DOC_CAP,
        alias_bonus: float = DEFAULT_ALIAS_BONUS,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        llm_bonus: float = DEFAULT_LLM_BONUS,
        llm_call: Optional[LLMCall] = None,
        # Phase E.4 — KG-derived channel. ``kg_call`` returns the
        # doc_ids whose articles mention the query concept; the
        # router adds ``kg_bonus`` to each matched doc. Both default
        # ``None`` / 0.5 so existing callers see no behaviour change.
        kg_bonus: float = DEFAULT_KG_BONUS,
        kg_call: Optional[KGCall] = None,
    ) -> None:
        self._registry = registry
        self._bm25 = bm25
        self._top_n = top_n
        self._bm25_pool = bm25_pool
        self._bm25_per_doc_cap = bm25_per_doc_cap
        self._alias_bonus = alias_bonus
        self._bm25_weight = bm25_weight
        self._llm_bonus = llm_bonus
        self._llm_call = llm_call
        # Phase E.4 — KG channel.
        self._kg_bonus = kg_bonus
        self._kg_call = kg_call
        self._sorted_aliases = self._build_alias_lookup()

    # ------------------------------------------------------------------
    def set_kg_call(self, kg_call: Optional[KGCall]) -> None:
        """Phase E.4 — attach (or replace) the KG channel after init.

        The dispatcher lazy-loads the KG on first MH/TF/CD dispatch, so
        the router's KG channel is built only when the KG is already in
        memory. Tests can also call this to swap in a stub.
        """
        self._kg_call = kg_call

    # ------------------------------------------------------------------
    def _build_alias_lookup(self) -> list[tuple[str, str]]:
        """Cache the alias map sorted longest-first.

        Sorting longest-first means a multi-word alias like
        "civil procedure" is checked before "civil", which prevents the
        substring "civil" inside "civil procedure" from masking the more
        specific match.
        """
        aliases = getattr(self._registry, "_aliases", {}) or {}
        items: list[tuple[str, str]] = []
        for k, v in aliases.items():
            if not k or v is None:
                continue
            items.append((k, v))
        items.sort(key=lambda kv: (-len(kv[0]), kv[0]))
        return items

    # ------------------------------------------------------------------
    def route(self, query: str, top_n: Optional[int] = None) -> RouteResult:
        """Return the top-N predicted doc_ids for *query*."""
        if top_n is None:
            top_n = self._top_n
        if top_n <= 0:
            return RouteResult(doc_ids=[], scores={}, sources={}, confidence=0.0)
        if not query or not query.strip():
            return RouteResult(doc_ids=[], scores={}, sources={}, confidence=0.0)

        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, list[str]] = defaultdict(list)

        # 1. Alias scan
        for doc_id in self._scan_aliases(query):
            scores[doc_id] += self._alias_bonus
            if "alias" not in sources[doc_id]:
                sources[doc_id].append("alias")

        # 2. Numeric law-id scan
        for doc_id in self._scan_numeric_ids(query):
            if "alias" not in sources[doc_id]:
                scores[doc_id] += self._alias_bonus
                sources[doc_id].append("alias")

        # 3. BM25 aggregation
        for doc_id, s in self._bm25_scores(query).items():
            scores[doc_id] += self._bm25_weight * s
            if "bm25" not in sources[doc_id]:
                sources[doc_id].append("bm25")

        # 4. Phase E.4 — KG channel. ``kg_call`` returns doc_ids whose
        # articles mention the query's concept phrases; we bump each
        # matched doc by ``kg_bonus`` (default 0.5). Bigram-based
        # CONTAINS is conservative — if it returns 0 docs the channel
        # is a no-op and the router falls back to alias+BM25.
        if self._kg_call is not None:
            try:
                kg_docs = self._kg_call(query) or []
            except Exception as exc:
                log.debug("doc_router KG channel skipped (%s)", exc)
                kg_docs = []
            for doc_id in kg_docs:
                scores[doc_id] += self._kg_bonus
                if "kg" not in sources[doc_id]:
                    sources[doc_id].append("kg")

        # 5. Optional LLM tie-breaker
        if self._llm_call is not None and scores:
            try:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                candidate_ids = [d for d, _ in ranked[: max(top_n * 3, 5)]]
                llm_picks = self._llm_call(query, candidate_ids) or []
                for doc_id in llm_picks:
                    if doc_id in scores:
                        scores[doc_id] += self._llm_bonus
                        if "llm" not in sources[doc_id]:
                            sources[doc_id].append("llm")
            except Exception as exc:
                log.debug("doc_router LLM channel skipped (%s)", exc)

        if not scores:
            return RouteResult(doc_ids=[], scores={}, sources={}, confidence=0.0)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ranked[:top_n]
        doc_ids = [d for d, _ in top]
        confidence = (
            1.0 if any("alias" in sources[d] for d in doc_ids) else 0.6
        )
        return RouteResult(
            doc_ids=doc_ids,
            scores={d: float(s) for d, s in ranked},
            sources={d: list(sources[d]) for d in scores},
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    def _scan_aliases(self, query: str) -> list[str]:
        """Return doc_ids whose alias appears in the query."""
        q_lower = query.lower()
        q_norm = normalize_arabic(query).lower()
        seen: set[str] = set()
        out: list[str] = []
        for alias, doc_id in self._sorted_aliases:
            if doc_id in seen:
                continue
            # Bare law-IDs are handled by _scan_numeric_ids with regex
            # word boundaries; skip them here to avoid false positives
            # like "75-58" matching inside the date "1975-58-...".
            if _LAW_ID_RE.fullmatch(alias):
                continue
            if _LATIN_RE.fullmatch(alias):
                # Word-boundary match for short Latin abbreviations
                pat = r"\b" + re.escape(alias) + r"\b"
                if re.search(pat, q_lower):
                    out.append(doc_id)
                    seen.add(doc_id)
                continue
            # Multi-word phrases or Arabic — substring match is reliable
            if alias in q_lower or alias in q_norm:
                out.append(doc_id)
                seen.add(doc_id)
        return out

    def _scan_numeric_ids(self, query: str) -> list[str]:
        """Return doc_ids resolved from bare law-id patterns like '84-11'."""
        seen: set[str] = set()
        out: list[str] = []
        for m in _LAW_ID_RE.finditer(query):
            tok = m.group(1)
            doc_id = self._registry.resolve_alias(tok)
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
        return out

    def _bm25_scores(self, query: str) -> dict[str, float]:
        """Aggregate normalised BM25 scores per doc_id (max → 1.0)."""
        if self._bm25 is None:
            return {}
        try:
            hits = self._bm25.search(query, k=self._bm25_pool)
        except Exception as exc:
            log.debug("doc_router BM25 channel skipped (%s)", exc)
            return {}
        if not hits:
            return {}
        per_doc: dict[str, list[float]] = defaultdict(list)
        for h in hits:
            per_doc[h.doc_id].append(float(h.score))
        agg: dict[str, float] = {}
        for d, scs in per_doc.items():
            top = sorted(scs, reverse=True)[: self._bm25_per_doc_cap]
            agg[d] = float(sum(top))
        max_agg = max(agg.values()) if agg else 0.0
        if max_agg <= 0.0:
            return {}
        return {d: s / max_agg for d, s in agg.items()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_doc_router(
    registry: ArticleRegistry,
    bm25: Optional[BM25Index] = None,
    **kwargs: Any,
) -> DocRouter:
    """Factory mirroring the :mod:`akn_rlm.baselines` ``build_*`` helpers."""
    return DocRouter(registry=registry, bm25=bm25, **kwargs)
