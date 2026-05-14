"""E1-E4 enhancers — post-HPC ablation experiments.

Each enhancer is a small, injectable callable that augments one
specific stage of the F5 dispatcher pipeline. Enabled via env flags
so the F5 baseline path is unchanged when off.

  E1 — concept_amendment_fn  : CD handler unions concept->amendment
                               SPARQL hits with the KG-bias set so
                               definitions that live in amending
                               decrees become retrievable. Env:
                               AKN_E1_CONCEPT_AMENDMENT=1
  E2 — nli_v2_verifier_fn    : reverse-direction NLI verifier. Gemma
                               rewrites the sub-question as a
                               declarative claim, then NLI scores
                               entailment(article, claim). Replaces
                               the F5 LLM verifier in RA/MH/EA.
                               Env: AKN_E2_NLI_REVERSE=1
  E3 — query_paraphrase_fn   : Gemma paraphrases the query before
                               retrieval; RRF fuses over [original,
                               paraphrases]. Applied to MH/TF/CD
                               (the hard types). Env: AKN_E3_PARAPHRASE=1
  E4 — hyde_query_enhancer   : Qwen drafts a hypothetical answer;
                               we embed query+answer instead of
                               just query. DenseIndex wrapper.
                               Env: AKN_E4_HYDE=1

Master flag: AKN_ENHANCERS=all enables E1+E2+E3+E4 simultaneously.

All enhancers fail open: an LLM exception, a model load failure, or
an empty response degrades to the F5 behaviour silently, so a flag
being on never makes things worse than a clean failure.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, List, Optional

from akn_rlm.gates.faithfulness_nli import CLAIM_THRESHOLD, entailment_score
from akn_rlm.normalizers import canonical_article_ref

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation flags
# ---------------------------------------------------------------------------

def _env_flag(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    master = os.getenv("AKN_ENHANCERS", "").strip().lower()
    return master in {"all", "1", "true", "yes"}


def is_e1_enabled() -> bool: return _env_flag("AKN_E1_CONCEPT_AMENDMENT")
def is_e2_enabled() -> bool: return _env_flag("AKN_E2_NLI_REVERSE")
def is_e3_enabled() -> bool: return _env_flag("AKN_E3_PARAPHRASE")
def is_e4_enabled() -> bool: return _env_flag("AKN_E4_HYDE")


#: Phase E.2 — KG topology disambiguator activation flag. When on, the
#: multi_hop handler post-consensus disambiguates pairs of similar-score
#: candidates by their KG structural distance + chapter-title concept
#: match. Default OFF for back-compat. Env: AKN_E5_KG_TOPOLOGY=1
def is_e5_enabled() -> bool: return _env_flag("AKN_E5_KG_TOPOLOGY")


#: Phase E.3 — concept-KG retrieval channel for multi_hop / rule_
#: application. When on, MH/RA handlers get a SPARQL-CONTAINS channel
#: that ADDS articles (alongside the BM25/Dense/HyDE RRF pool) whose
#: any-version text contains a query phrase. Same pattern as Fix-TF's
#: KG-first channel but without the date filter — works on any-version
#: text, useful for queries like "حدود الإقرار في القانون المدني" where
#: the gold article's text literally contains the concept phrase but
#: BM25/Dense miss it. Default OFF. Env: AKN_E6_CONCEPT_KG=1
def is_e6_enabled() -> bool: return _env_flag("AKN_E6_CONCEPT_KG")


#: Phase E.4 — KG-derived doc-router channel. When on, the DocRouter
#: gets a 4th channel that queries the KG for docs containing
#: articles whose text mentions a query phrase, then adds a bonus to
#: the matched doc_ids. Helps lift recall@3 above the existing 82.9%
#: baseline by surfacing docs that BM25 missed because the query's
#: concept phrase doesn't lexically match the doc's title or
#: high-IDF tokens. Default OFF. Env: AKN_E7_KG_DOC_ROUTER=1
def is_e7_enabled() -> bool: return _env_flag("AKN_E7_KG_DOC_ROUTER")


GEMMA_MODEL = "google/gemma-4-31B"


# ===========================================================================
# E1 — concept_amendment_fn  (re-export from ceiling_breakers for one home)
# ===========================================================================

from akn_rlm.rlm.ceiling_breakers import (   # noqa: E402
    make_concept_amendment_search as make_concept_amendment_fn,
)


# ===========================================================================
# E5 — KG structural distance + topology disambiguator (Phase E.2)
# ===========================================================================
#
# Two purposes, one helper:
#
#   1. Multi-hop disambiguator (`make_kg_topology_disambiguator`).
#      When the consensus-boost ranking leaves two candidates with
#      similar final_score in the same doc, we ask the KG which one
#      lives in a chapter/section whose *title* contains the query
#      concept. The matched candidate gets promoted, dethroning the
#      adjacent-but-wrong sibling that Fix-MH's confidence-uniform
#      verifier couldn't discriminate (HANDOFF §R2: civ_409 vs gold
#      civ_408).
#
#   2. Chapter-title channel for temporal_factual (folded from E.1).
#      Phase E.1 falsified the "URI/merge bug" hypothesis: SPARQL
#      CONTAINS over versionText simply misses scope-defining art_1
#      on 4/7 TF questions because art_1 *defines* the concept rather
#      than repeating its exact phrasing. The chapter-title channel
#      surfaces art_1 of any chapter whose URI/title fragment contains
#      the concept — orthogonal retrieval signal to text-CONTAINS.
#
# Failure semantics: every helper degrades to the empty result on KG
# absence, SPARQL errors, or unresolvable URIs. The MH/TF handlers
# keep their F5/Phase D ranking intact when the disambiguator returns
# nothing.

# Walk the ``directlyContainedIn`` chain transitively to enumerate
# every Chapter / Section / Book ancestor an article belongs to.
# NOTE: ``dzdoc:containedIn`` in the AKN-RLM KG only links each
# entity to its enclosing *document* (not its chapter/section), so we
# CANNOT use it for structural distance — we must follow
# ``directlyContainedIn`` recursively via SPARQL property path ``+``.
_E5_ANCESTORS_SPARQL = """\
PREFIX dzdoc: <https://legal.dz/ontology/document#>

SELECT DISTINCT ?ancestor ?type WHERE {
  <%s> dzdoc:directlyContainedIn+ ?ancestor .
  ?ancestor a ?type .
  FILTER(?type IN (dzdoc:Chapter, dzdoc:Section, dzdoc:Book))
}
"""

# Every Chapter / Section in a single doc — used by the chapter-title
# channel to enumerate candidate containers whose URI fragment
# contains the query concept.
_E5_CHAPTERS_IN_DOC_SPARQL = """\
PREFIX dzdoc: <https://legal.dz/ontology/document#>

SELECT DISTINCT ?container ?type WHERE {
  ?container a ?type .
  FILTER(?type IN (dzdoc:Chapter, dzdoc:Section, dzdoc:Book))
  FILTER(STRSTARTS(STR(?container), "%s"))
}
"""

# Articles directly contained in a given container — used to surface
# art_1 (and friends) when a chapter title matches a query concept.
_E5_ARTICLES_IN_CONTAINER_SPARQL = """\
PREFIX dzdoc: <https://legal.dz/ontology/document#>

SELECT DISTINCT ?article WHERE {
  ?article a dzdoc:Article .
  ?article dzdoc:directlyContainedIn <%s> .
}
ORDER BY ASC(STR(?article))
"""


SparqlFn = Callable[[str], list]


def _ask_to_bool(result: Any) -> bool:
    """Normalise an ASK-result to a bool (mirrors temporal_factual)."""
    if isinstance(result, bool):
        return result
    if isinstance(result, list):
        if not result:
            return False
        first = result[0]
        if isinstance(first, dict):
            for v in first.values():
                if isinstance(v, bool):
                    return v
                if str(v).strip().lower() in ("true", "1"):
                    return True
            return True
        return bool(first)
    return False


def _kg_article_ancestors(
    sparql_fn: Optional[SparqlFn], article_uri: str,
) -> set[str]:
    """Return the set of ancestor URIs (Chapter / Section / Book) for
    ``article_uri``. Empty when sparql_fn is None / URI is empty /
    the query errors. Used by :func:`kg_structural_distance`."""
    if not sparql_fn or not article_uri:
        return set()
    try:
        rows = sparql_fn(_E5_ANCESTORS_SPARQL % article_uri) or []
    except Exception as exc:
        log.debug("E5 ancestors SPARQL failed for %s: %s", article_uri, exc)
        return set()
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        a = row.get("ancestor") or row.get("?ancestor")
        if a:
            out.add(str(a))
    return out


def kg_structural_distance(
    sparql_fn: Optional[SparqlFn],
    article_uri_a: str,
    article_uri_b: str,
    *,
    doc_uri_a: Optional[str] = None,
    doc_uri_b: Optional[str] = None,
) -> float:
    """Phase E.2 — KG structural distance between two articles.

    Returns:
      * ``0.0`` when both articles share their deepest ancestor — i.e.
        same Section (and therefore same Chapter and same Book).
      * ``1.0`` when they share a Chapter but not a Section.
      * ``2.0`` when they share a doc/Book but not a Chapter.
      * ``float('inf')`` when they don't share an ancestor at all.

    The implementation does **not** care whether the ancestor is a
    Chapter or Section per-se — it just enumerates the ancestor set
    of each article and uses the size of the intersection as the
    inverse-distance proxy:

      * |intersection| >= 2  → distance 0 (chapter AND section overlap)
      * |intersection| == 1  → distance 1 (chapter only)
      * |intersection| == 0 but same doc URI → distance 2
      * otherwise            → infinity

    Pass ``doc_uri_a`` / ``doc_uri_b`` to allow the "same doc, no
    chapter overlap" case to resolve to ``2.0`` even when both
    articles' chapter URIs failed to load.
    """
    a = _kg_article_ancestors(sparql_fn, article_uri_a)
    b = _kg_article_ancestors(sparql_fn, article_uri_b)
    intersection = a & b
    if len(intersection) >= 2:
        return 0.0
    if len(intersection) == 1:
        return 1.0
    # No shared chapter/section. Check doc equality via supplied
    # ``doc_uri_*`` OR by stripping the fragment from each URI. This
    # branch handles BOTH the "no ancestors either side" case AND the
    # "ancestors exist but don't overlap" case uniformly.
    def _doc(uri: str) -> str:
        return uri.split("#", 1)[0] if uri else ""
    d_a = doc_uri_a or _doc(article_uri_a)
    d_b = doc_uri_b or _doc(article_uri_b)
    if d_a and d_b and d_a == d_b:
        return 2.0
    return float("inf")


def kg_containers_matching_phrase(
    sparql_fn: Optional[SparqlFn],
    doc_uri_prefix: str,
    phrases: list[str],
) -> list[tuple[str, str]]:
    """Phase E.2 — find Chapter / Section / Book URIs in a single doc
    whose URI fragment (i.e. structural title) contains any of
    ``phrases``. Returns a list of ``(container_uri, matched_phrase)``
    tuples ordered by URI.

    Why URI-fragment matching: half of the Algerian Legal KG's
    Chapter URIs encode their human-readable title in the fragment
    (e.g. ``#chp_الثاني_السلطة_و_تنظيمها``); the other half are bare
    numeric (e.g. ``#chp_1``). The bare-numeric case adds zero signal
    via this channel, so this helper only fires when the human title
    is available — which is the case where TF's CONTAINS-versionText
    channel struggles the most (questions framed around legal
    concepts like *labor relations* or *family-law amendments*).
    """
    if not sparql_fn or not doc_uri_prefix or not phrases:
        return []
    try:
        rows = sparql_fn(_E5_CHAPTERS_IN_DOC_SPARQL % doc_uri_prefix) or []
    except Exception as exc:
        log.debug("E5 chapters-in-doc SPARQL failed for %s: %s",
                  doc_uri_prefix, exc)
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        uri = row.get("container") or row.get("?container") or ""
        uri_s = str(uri)
        if not uri_s or uri_s in seen:
            continue
        # Take the fragment after '#'. Chapter / Section URIs in the
        # Algerian Legal KG encode their title in the fragment using
        # underscores as word separators (e.g. ``chp_الثاني_السلطة_و_
        # تنظيمها``). Normalise to spaces so the bigram phrases the
        # TF tokenizer emits can hit them as substrings.
        fragment = uri_s.split("#", 1)[1] if "#" in uri_s else uri_s
        fragment_norm = fragment.replace("_", " ")
        for phrase in phrases:
            phrase = (phrase or "").strip()
            if len(phrase) < 3:
                continue
            if phrase in fragment_norm:
                out.append((uri_s, phrase))
                seen.add(uri_s)
                break
    return sorted(out, key=lambda kv: kv[0])


def kg_articles_in_container(
    sparql_fn: Optional[SparqlFn],
    container_uri: str,
    *,
    limit: Optional[int] = None,
) -> list[str]:
    """Phase E.2 — list every Article URI directly contained in
    ``container_uri`` (a Chapter / Section / Book), ordered by URI.
    Empty list on KG absence or SPARQL error.

    The natural ordering is by URI fragment, which for AKN articles
    means "art_1 first" — exactly the position where scope-defining
    foundational articles live (the failure mode E.1 documents).
    """
    if not sparql_fn or not container_uri:
        return []
    try:
        rows = sparql_fn(_E5_ARTICLES_IN_CONTAINER_SPARQL % container_uri) or []
    except Exception as exc:
        log.debug("E5 articles-in-container SPARQL failed for %s: %s",
                  container_uri, exc)
        return []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        a = row.get("article") or row.get("?article")
        if a:
            out.append(str(a))
        if limit is not None and len(out) >= limit:
            break
    return out


KGTopologyDisambiguatorFn = Callable[
    [str, list[dict[str, Any]]],
    list[dict[str, Any]],
]


def make_kg_topology_disambiguator(
    sparql_fn: Optional[SparqlFn],
    *,
    resolve_uri: Callable[[str, str], Optional[str]],
    similarity_window: float = 0.10,
    promote_bonus: float = 0.30,
) -> KGTopologyDisambiguatorFn:
    """Phase E.2 (Fix-MH v2) — return a function that re-ranks a
    list of citation dicts using KG structural distance + chapter-
    title concept match.

    Args:
      sparql_fn: A callable matching the rdflib SPARQL signature used
        by temporal_factual / ceiling_breakers. ``None`` → the
        returned function is a no-op identity.
      resolve_uri: A ``(doc_id, article_ref) -> Optional[str]`` URI
        resolver, typically ``temporal_factual._resolve_article_uri``
        bound to the same ``sparql_fn``. Lets the disambiguator
        operate on the (doc_id, article_ref) shape MH already uses.
      similarity_window: Two candidates with ``abs(final_score
        difference) <= similarity_window`` are considered "tied" and
        eligible for disambiguation. Default 0.10 mirrors the
        Fix-MH calibration: consensus boost is 0.25 per extra sub-Q,
        so 0.10 catches "one extra sub-Q + some confidence drift".
      promote_bonus: Amount added to the winning candidate's
        confidence/score field when the disambiguator fires. Default
        0.30 — large enough to flip the ranking but small enough to
        leave a multi-consensus winner untouched.

    Returns:
      A function ``(query, citations) -> reordered citations`` that
      can be invoked from MH or any other handler that wants
      structural disambiguation post-consensus.

      The returned function NEVER reduces the citation set; it only
      promotes via ``promote_bonus`` so a downstream truncate
      preserves the ones the disambiguator favoured. Citations not
      involved in any tie pass through unchanged.
    """
    if sparql_fn is None:
        def _identity(_query: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return list(citations)
        return _identity

    def _phrases(query: str) -> list[str]:
        # Reuse the same phrase extraction TF uses, plus single-token
        # content words (length>=5, stopwords filtered). Multi-token
        # phrases catch composite concepts; long unigrams catch
        # single-noun concepts (e.g. "السلطة" / "العقوبة") that
        # routinely appear as the chapter-title concept on their own.
        # Local import avoids a top-level cycle (enhancers <- TF).
        from akn_rlm.rlm.handlers.temporal_factual import (
            _TF_AR_STOP,
            _TF_TOKEN_RE,
            _tf_query_phrases,
        )
        multi = _tf_query_phrases(query, max_n=8)
        unis: list[str] = []
        if query:
            for tok in _TF_TOKEN_RE.split(query):
                if not tok or len(tok) < 5 or tok in _TF_AR_STOP:
                    continue
                if tok not in unis:
                    unis.append(tok)
        return multi + unis[:6]

    def _disambiguate(
        query: str, citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not citations or len(citations) < 2:
            return list(citations)
        phrases = _phrases(query)
        if not phrases:
            return list(citations)

        # Pre-resolve URIs once per candidate so we don't hit the KG
        # twice for the same article inside the O(n^2) tie loop.
        uris: list[Optional[str]] = []
        for c in citations:
            doc = str(c.get("doc_id", ""))
            ref = str(c.get("article_ref", ""))
            try:
                uris.append(resolve_uri(doc, ref))
            except Exception:
                uris.append(None)

        # Cache of chapter-title matches per (doc URI prefix).
        title_cache: dict[str, list[tuple[str, str]]] = {}

        promoted_indices: set[int] = set()
        for i in range(len(citations)):
            uri_i = uris[i]
            if not uri_i:
                continue
            score_i = float(citations[i].get("confidence", 0.0))
            for j in range(i + 1, len(citations)):
                if j in promoted_indices:
                    continue
                uri_j = uris[j]
                if not uri_j:
                    continue
                if citations[i].get("doc_id") != citations[j].get("doc_id"):
                    continue
                score_j = float(citations[j].get("confidence", 0.0))
                if abs(score_i - score_j) > similarity_window:
                    continue
                # Tie + same doc. Ask the KG which one is in a
                # title-matched chapter/section.
                doc_prefix = uri_i.split("#", 1)[0]
                if doc_prefix not in title_cache:
                    title_cache[doc_prefix] = kg_containers_matching_phrase(
                        sparql_fn, doc_prefix, phrases,
                    )
                matched_containers = {c[0] for c in title_cache[doc_prefix]}
                if not matched_containers:
                    continue
                anc_i = _kg_article_ancestors(sparql_fn, uri_i)
                anc_j = _kg_article_ancestors(sparql_fn, uri_j)
                i_match = bool(anc_i & matched_containers)
                j_match = bool(anc_j & matched_containers)
                if i_match and not j_match:
                    promoted_indices.add(i)
                    citations[i]["kg_topology_promoted"] = True
                elif j_match and not i_match:
                    promoted_indices.add(j)
                    citations[j]["kg_topology_promoted"] = True
                # Both or neither match → no action (no clear winner).

        if not promoted_indices:
            return list(citations)

        # Apply the bonus and re-sort.
        out = []
        for idx, c in enumerate(citations):
            new_c = dict(c)
            if idx in promoted_indices:
                new_c["confidence"] = float(new_c.get("confidence", 0.0)) + promote_bonus
            out.append(new_c)
        out.sort(key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
        return out

    return _disambiguate


# ===========================================================================
# E6 — Concept-KG retrieval channel for MH/RA (Phase E.3)
# ===========================================================================
#
# A SPARQL-CONTAINS retrieval channel that surfaces articles whose any
# version text contains a query phrase. Returns candidate dicts in the
# same shape as ``akn_rlm.retrievers.hybrid_fusion.rrf_fuse`` output so
# MH / RA handlers can union them into their existing per-sub-q (or
# per-query) fused pool before verification — the Fix-TF KG-first
# pattern, but without the date filter and with a slightly lower base
# score so it doesn't dominate the consensus boost on MH.
#
# Why a separate helper instead of reusing TF's ``_tf_kg_first_candidates``:
# TF anchors on a date target and uses a high base_score=0.6 to
# guarantee its hits survive the top-K cap. MH/RA queries are *not*
# version-specific; injecting a 0.6-anchored channel would dominate
# the BM25/Dense/HyDE RRF pool and break the consensus boost. We use
# ``base_score=0.35`` here so concept-KG candidates compete fairly
# inside the per-sub-q top-K but don't dethrone strong RRF winners.

#: Default base score for an E6 bigram/trigram hit before phrase-bonus.
#: Calibrated below the strongest single-channel BM25/Dense RRF score
#: (~0.5 typical for a high-precision hit) and above the typical
#: filler hit (~0.1).
DEFAULT_E6_BASE_SCORE: float = 0.35

#: Lower base score for the E6 unigram-fallback pass. Real-KG probe
#: (2026-05-14 against 90-11 art_1 / "علاقات العمل") shows the multi-
#: token CONTAINS misses scope-defining articles that use the *same*
#: words in a different word order. Unigrams of length>=6 surface
#: those articles (33-68 hits in the test case) but at much lower
#: precision, so we score them lower and ONLY fire the unigram pass
#: when bigrams returned 0 candidates.
DEFAULT_E6_UNIGRAM_BASE_SCORE: float = 0.22

#: Per-query cap on unigram-fallback candidates. The unigram pass can
#: return 60+ hits on common Arabic content tokens; we keep only the
#: highest-scoring ones to bound noise.
DEFAULT_E6_UNIGRAM_TOP_N: int = 5

#: Same per-phrase SPARQL row cap as Fix-TF's KG-first channel.
DEFAULT_E6_LIMIT_PER_PHRASE: int = 30

#: Minimum unigram length for the fallback pass. <6 chars produces
#: stopword-grade noise (matches thousands of articles per token).
DEFAULT_E6_UNIGRAM_MIN_LEN: int = 6


ConceptKgChannelFn = Callable[[str, Optional[list[str]]], list[dict[str, Any]]]


def make_concept_kg_channel(
    sparql_fn: Optional[SparqlFn],
    *,
    base_score: float = DEFAULT_E6_BASE_SCORE,
    limit_per_phrase: int = DEFAULT_E6_LIMIT_PER_PHRASE,
    # Unigram-fallback kwargs preserved for back-compat with the
    # interim 2026-05-14 implementation; ``enable_unigram_fallback``
    # defaults to ``False`` because the full-244 measurement showed
    # the fallback flooded MH/RA verifier slots (top-K=8 with
    # verify_top_n=4) with unigram-precision noise, displacing the
    # actual hybrid hits and regressing MH/RA −0.038/−0.037 Cite F1.
    # The kwargs stay so a future operator with a fixed score-merge
    # path (or a corpus-tuned embedder, Phase F) can re-enable it.
    enable_unigram_fallback: bool = False,
    unigram_base_score: float = DEFAULT_E6_UNIGRAM_BASE_SCORE,
    unigram_top_n: int = DEFAULT_E6_UNIGRAM_TOP_N,
    unigram_min_len: int = DEFAULT_E6_UNIGRAM_MIN_LEN,
) -> ConceptKgChannelFn:
    """Phase E.3 — return a callable that produces concept-KG candidate
    dicts for a query.

    The returned function takes ``(query, routed_ids)`` and returns a
    list of dicts shaped like ``rrf_fuse`` output:
      ``{chunk_id, doc_id, article_ref, text, score, retriever}``

    Phrase extraction matches Fix-TF: trigrams + bigrams of content
    tokens, stopwords stripped. ``score`` starts at ``base_score`` and
    grows by ``+0.05`` per additional phrase that hits the same
    article — multi-phrase agreement is a stronger signal than a
    single hit.

    Empty / no-KG / SPARQL-error inputs silently return ``[]`` so the
    handler keeps its hybrid-only path.
    """
    if sparql_fn is None:
        def _noop(_query: str, _routed_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
            return []
        return _noop

    def _run_contains(
        phrase: str,
        base: float,
        existing: dict[tuple[str, str], dict[str, Any]],
        routed_set: Optional[set[str]],
        from_uri: Callable[[str], Optional[tuple[str, str]]],
    ) -> None:
        """Run one CONTAINS pass and merge hits into ``existing``."""
        safe = phrase.replace("\\", "\\\\").replace('"', '\\"')
        sparql = (
            'PREFIX dzdoc: <https://legal.dz/ontology/document#>\n'
            'SELECT DISTINCT ?article ?text WHERE {\n'
            '  ?article a dzdoc:Article .\n'
            '  {\n'
            '    ?article dzdoc:hasVersion ?v .\n'
            '    ?v dzdoc:versionText ?text .\n'
            '  } UNION {\n'
            '    ?article dzdoc:fullText ?text .\n'
            '  }\n'
            f'  FILTER(CONTAINS(STR(?text), "{safe}"))\n'
            '}\n'
            f'LIMIT {limit_per_phrase}'
        )
        try:
            rows = sparql_fn(sparql) or []
        except Exception as exc:
            log.debug("E6 concept-KG SPARQL failed for %r: %s", phrase, exc)
            return
        for row in rows:
            uri = str(row.get("article", "")) if isinstance(row, dict) else ""
            text = str(row.get("text", "")) if isinstance(row, dict) else ""
            parsed = from_uri(uri)
            if not parsed:
                continue
            doc_id, ref = parsed
            if routed_set and doc_id not in routed_set:
                continue
            key = (doc_id, ref)
            entry = existing.get(key)
            if entry is None:
                existing[key] = {
                    "chunk_id":    f"{doc_id}#art_{ref}",
                    "doc_id":      doc_id,
                    "article_ref": ref,
                    "text":        text,
                    "score":       float(base),
                    "retriever":   "kg_concept",
                    "kg_concept_match": True,
                }
            else:
                entry["score"] = min(1.0, entry["score"] + 0.05)

    def _channel(
        query: str,
        routed_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        # Reuse TF's phrase extractor for parity with the KG-first
        # channel — local import avoids the top-level cycle.
        from akn_rlm.rlm.handlers.temporal_factual import (
            _TF_AR_STOP,
            _TF_TOKEN_RE,
            _tf_query_phrases,
            _tf_uri_to_doc_ref,
        )

        phrases = _tf_query_phrases(query)
        routed_set: Optional[set[str]] = set(routed_ids) if routed_ids else None
        results: dict[tuple[str, str], dict[str, Any]] = {}

        # Bigram / trigram pass.
        for phrase in phrases:
            _run_contains(phrase, base_score, results, routed_set,
                          _tf_uri_to_doc_ref)

        # Unigram fallback (opt-in only). Surfaces scope-defining
        # articles whose text uses the same content tokens as the
        # query in a different word order — the failure mode E.1
        # documented. **Disabled by default** because the 2026-05-14
        # full-244 measurement showed the fallback regressed MH/RA
        # by ~0.038 Cite F1: the unigram base_score (0.22) is far
        # above typical hybrid RRF scores (~0.03-0.07), so unigram
        # candidates fill the verifier top-K and displace actual
        # hybrid hits, then get rejected by the verifier as
        # irrelevant.
        if enable_unigram_fallback and not results and query:
            seen_unis: list[str] = []
            for tok in _TF_TOKEN_RE.split(query):
                if (not tok
                        or len(tok) < unigram_min_len
                        or tok in _TF_AR_STOP
                        or tok in seen_unis):
                    continue
                seen_unis.append(tok)
            unigram_results: dict[tuple[str, str], dict[str, Any]] = {}
            for tok in seen_unis:
                _run_contains(tok, unigram_base_score, unigram_results,
                              routed_set, _tf_uri_to_doc_ref)
            ranked = sorted(
                unigram_results.values(),
                key=lambda c: float(c.get("score", 0.0)),
                reverse=True,
            )
            for cand in ranked[: unigram_top_n]:
                results[(cand["doc_id"], cand["article_ref"])] = cand

        return list(results.values())

    return _channel


def merge_hybrid_with_concept_kg(
    hybrid: list[dict[str, Any]],
    concept_kg: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Phase E.3 — additive union of the hybrid (RRF) pool with the
    concept-KG channel. Keys candidates by ``(doc_id, article_ref)``,
    keeps the higher-scoring entry, and preserves the ``kg_concept_match``
    flag whenever either source had it. Used by both MH and RA handlers
    so the merge logic is shared.

    Ordering: returned list sorted by ``score`` desc (matches the
    RRF-fuse convention).
    """
    if not concept_kg:
        return list(hybrid)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for src in (hybrid, concept_kg):
        for cand in src:
            key = (str(cand.get("doc_id", "")),
                   str(cand.get("article_ref", "")))
            prior = by_key.get(key)
            if prior is None:
                by_key[key] = dict(cand)
                continue
            new_score = float(cand.get("score", 0.0))
            prior_score = float(prior.get("score", 0.0))
            if new_score > prior_score:
                merged = dict(cand)
                if not merged.get("text") and prior.get("text"):
                    merged["text"] = prior["text"]
                # Preserve concept-match flag from either side.
                if prior.get("kg_concept_match") or merged.get("kg_concept_match"):
                    merged["kg_concept_match"] = True
                by_key[key] = merged
            else:
                if not prior.get("text") and cand.get("text"):
                    prior["text"] = cand["text"]
                if cand.get("kg_concept_match"):
                    prior["kg_concept_match"] = True
    return sorted(by_key.values(),
                  key=lambda c: float(c.get("score", 0.0)), reverse=True)


# ===========================================================================
# E7 — KG-derived doc-router channel (Phase E.4)
# ===========================================================================
#
# A thin wrapper that produces a ``list[doc_id]`` for the DocRouter to
# union with its alias / numeric / BM25 channels. Internally runs the
# same SPARQL CONTAINS pattern as the E.3 concept-KG channel but only
# emits *distinct doc_ids* (no article-level info, no scoring). The
# DocRouter then applies its own ``kg_bonus`` weight per matched doc.
#
# Why a separate helper instead of reusing ``make_concept_kg_channel``:
# the DocRouter doesn't care about article-level scores or text — it
# just wants the set of doc_ids that the KG associates with the query
# concept. Skipping article-level bookkeeping makes the SPARQL faster
# and keeps the integration surface narrow.

KGDocRouterCallFn = Callable[[str], list[str]]


def make_kg_doc_router_call(
    sparql_fn: Optional[SparqlFn],
    *,
    limit_per_phrase: int = 30,
    max_docs: int = 10,
) -> KGDocRouterCallFn:
    """Phase E.4 — return a callable producing doc_ids for the
    DocRouter's KG channel.

    Args:
      sparql_fn: rdflib SPARQL caller; ``None`` returns a no-op.
      limit_per_phrase: per-phrase SPARQL LIMIT (matches E.3 channel).
      max_docs: cap on the returned doc_id list. Calibrated so the
        KG channel can't dominate routing — if 10+ docs match, the
        signal is too weak to be useful as a router-level bonus.

    Returns a function ``(query) -> list[doc_id]``. Failure-open:
    SPARQL errors or empty phrases → empty list.
    """
    if sparql_fn is None:
        def _noop(_query: str) -> list[str]:
            return []
        return _noop

    def _call(query: str) -> list[str]:
        if not query:
            return []
        from akn_rlm.rlm.handlers.temporal_factual import (
            _tf_query_phrases,
            _tf_uri_to_doc_ref,
        )
        phrases = _tf_query_phrases(query)
        if not phrases:
            return []
        # Order-preserving distinct doc_ids.
        seen: list[str] = []
        seen_set: set[str] = set()
        for phrase in phrases:
            if len(seen) >= max_docs:
                break
            safe = phrase.replace("\\", "\\\\").replace('"', '\\"')
            sparql = (
                'PREFIX dzdoc: <https://legal.dz/ontology/document#>\n'
                'SELECT DISTINCT ?article WHERE {\n'
                '  ?article a dzdoc:Article .\n'
                '  {\n'
                '    ?article dzdoc:hasVersion ?v .\n'
                '    ?v dzdoc:versionText ?text .\n'
                '  } UNION {\n'
                '    ?article dzdoc:fullText ?text .\n'
                '  }\n'
                f'  FILTER(CONTAINS(STR(?text), "{safe}"))\n'
                '}\n'
                f'LIMIT {limit_per_phrase}'
            )
            try:
                rows = sparql_fn(sparql) or []
            except Exception as exc:
                log.debug("E7 KG doc-router SPARQL failed for %r: %s",
                          phrase, exc)
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                uri = str(row.get("article", ""))
                parsed = _tf_uri_to_doc_ref(uri)
                if not parsed:
                    continue
                doc_id = parsed[0]
                if doc_id in seen_set:
                    continue
                seen_set.add(doc_id)
                seen.append(doc_id)
                if len(seen) >= max_docs:
                    break
        return seen

    return _call


# ===========================================================================
# E2 — reverse NLI verifier (article entails declarative-claim form of Q)
# ===========================================================================

_Q_TO_CLAIM_PROMPT = (
    "Convert this Arabic legal question into a SINGLE declarative claim "
    "that a legal article could entail. The claim should state what the "
    "answer would look like as a fact. Output ONLY the claim in Arabic, "
    "no preamble, no quotes.\n\n"
    "Question: {question}\n\n"
    "Claim:"
)


def _strip_label(s: str) -> str:
    s = s.strip()
    # Drop common leading labels Gemma sometimes emits.
    for prefix in ("Claim:", "claim:", "الادعاء:", "العبارة:", "الإجابة:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # Drop wrapping quotes.
    if len(s) >= 2 and s[0] in "\"'“«" and s[-1] in "\"'”»":
        s = s[1:-1].strip()
    return s


def make_nli_v2_verifier_fn(
    llm_pool: Any,
    *,
    claim_model: str = GEMMA_MODEL,
    threshold: float = CLAIM_THRESHOLD,
    fall_back_to_llm: bool = True,
) -> Callable:
    """Build a verifier_fn matching :func:`call_verifier`'s signature.

    Gemma rewrites the sub-question once per unique question (cached),
    then NLI scores entailment(article_text, claim). This is the
    correct direction for NLI: an article entails a claim, never a
    question.

    Failures fall back to the F5 LLM verifier (or to a {relevant=False}
    verdict if even that fails).
    """
    from akn_rlm.rlm.sub_worker import call_verifier as _llm_verifier

    claim_cache: dict[str, str] = {}

    def _question_to_claim(question: str) -> str:
        if not question:
            return ""
        if question in claim_cache:
            return claim_cache[question]
        try:
            raw = llm_pool.call(
                _Q_TO_CLAIM_PROMPT.format(question=question[:1200]),
                model=claim_model, max_tokens=200, temperature=0.0,
            )
            claim = _strip_label(raw) or question
        except Exception as exc:
            log.debug("E2 question->claim failed: %s", exc)
            claim = question
        claim_cache[question] = claim
        return claim

    def _verify(
        llm_pool_inner: Any,
        sub_question: str,
        article: dict,
        model: str,
    ) -> dict:
        article_text = article.get("text", "") or article.get("supporting_span", "")
        if not article_text or not sub_question:
            return {
                "relevant": False,
                "supporting_span": None,
                "contradicting_span": None,
                "confidence": 0.0,
            }
        claim = _question_to_claim(sub_question)
        score = entailment_score(article_text, claim)
        if score == 0.5 and fall_back_to_llm:
            # NLI model unavailable / threw — defer to F5 LLM verifier.
            try:
                return _llm_verifier(llm_pool_inner, sub_question, article, model)
            except Exception as exc:
                log.warning("E2 LLM-fallback verifier raised: %s", exc)
        relevant = bool(score >= threshold)
        return {
            "relevant": relevant,
            "supporting_span": article_text[:280] if relevant else None,
            "contradicting_span": None,
            "confidence": float(score),
        }

    return _verify


# ===========================================================================
# E3 — Gemma query paraphrase for hard types
# ===========================================================================

_PARAPHRASE_PROMPT = (
    "Generate {n} Arabic paraphrases of this Algerian legal question. "
    "Preserve meaning exactly but vary the surface phrasing (use synonyms, "
    "different word order, formal MSA legal terms). Each paraphrase on its "
    "own line. NO numbering, NO commentary, NO original question.\n\n"
    "Question: {question}\n\n"
    "Paraphrases:"
)


def make_query_paraphrase_fn(
    llm_pool: Any,
    *,
    model: str = GEMMA_MODEL,
    n: int = 2,
) -> Callable[[str], List[str]]:
    """Return ``paraphrase(query) -> list[str]`` of up to N MSA paraphrases.

    Cached per query so a handler calling it twice doesn't pay twice.
    Empty / failed paraphrasing returns ``[]`` (caller falls back to
    original-only retrieval).
    """
    cache: dict[str, List[str]] = {}

    def _paraphrase(query: str) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []
        if query in cache:
            return cache[query]
        try:
            raw = llm_pool.call(
                _PARAPHRASE_PROMPT.format(n=n, question=query[:1200]),
                model=model, max_tokens=400, temperature=0.0,
            )
        except Exception as exc:
            log.debug("E3 paraphrase failed: %s", exc)
            cache[query] = []
            return []
        out: List[str] = []
        for line in (raw or "").splitlines():
            stripped = line.strip().lstrip("-•*1234567890.) ").strip()
            stripped = stripped.strip("\"'“”«»")
            if stripped and stripped != query and stripped not in out:
                out.append(stripped)
            if len(out) >= n:
                break
        cache[query] = out
        return out

    return _paraphrase


# ===========================================================================
# E4 — HyDE: hypothetical-answer-augmented retrieval
# ===========================================================================

_HYDE_PROMPT = (
    "You are an Algerian legal expert. Answer this question briefly in MSA "
    "Arabic, 2-3 sentences, citing the type of article that would apply. "
    "If you don't know the exact answer, write the SHAPE of the answer that "
    "the correct article would have. NO preamble.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def make_hyde_query_enhancer(
    llm_pool: Any,
    *,
    model: str = "Qwen3-30B-A3B-Thinking",
    cache: bool = True,
) -> Callable[[str], str]:
    """Return ``enhance(query) -> "query  HYDE_SEP  hypothetical_answer"``.

    Used to augment the *query side* of dense retrieval. The original
    BM25 channel still sees the bare query so token-exact matches are
    not diluted; only the dense channel gets the HyDE expansion.

    Caches per query so a multi-handler dispatch doesn't pay twice.
    """
    _cache: Optional[dict[str, str]] = {} if cache else None

    def _enhance(query: str) -> str:
        if not query or not query.strip():
            return query or ""
        if _cache is not None and query in _cache:
            return _cache[query]
        try:
            answer = llm_pool.call(
                _HYDE_PROMPT.format(question=query[:1200]),
                model=model, max_tokens=250, temperature=0.0,
            ) or ""
            answer = answer.strip()
        except Exception as exc:
            log.debug("E4 HyDE generation failed: %s", exc)
            answer = ""
        if not answer:
            augmented = query
        else:
            # Concatenate with a soft separator. The dense encoder treats
            # this as one passage; the model attends across both.
            augmented = f"{query}\n\n{answer}"
        if _cache is not None:
            _cache[query] = augmented
        return augmented

    return _enhance


class HyDEDenseIndex:
    """Wraps a DenseIndex so .search(query) augments the query first.

    The wrapped index keeps the original DenseIndex unchanged — every
    other handler that calls ``.search`` gets the HyDE augmentation
    automatically when this wrapper is plugged in.
    """

    def __init__(self, inner, enhancer: Callable[[str], str]) -> None:
        self._inner = inner
        self._enhance = enhancer

    def search(self, query: str, k: int = 20):
        return self._inner.search(self._enhance(query), k=k)

    # Delegate any other attribute access (e.g. ``_model`` introspection
    # used by tests) to the wrapped index.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ===========================================================================
# E3 helper — RRF-merge wrapper for multi-query retrieval
# ===========================================================================

def _hit_key(h: Any) -> str:
    """Stable de-dup key for both DenseHit and BM25Hit shapes."""
    cid = getattr(h, "chunk_id", None)
    if cid is not None:
        return cid
    doc = getattr(h, "doc_id", "")
    ref = getattr(h, "article_ref", "")
    return f"{doc}#{ref}"


def _rrf_merge_hits(hits_per_query: List[List[Any]], k: int, rrf_k: int = 60) -> List[Any]:
    """Reciprocal Rank Fusion across N hit lists. Returns top-k merged."""
    score: dict[str, float] = {}
    canon: dict[str, Any] = {}
    for hits in hits_per_query:
        for rank, h in enumerate(hits, 1):
            key = _hit_key(h)
            score[key] = score.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in canon:
                canon[key] = h
    ranked_keys = sorted(score.keys(), key=lambda kk: -score[kk])
    return [canon[k_] for k_ in ranked_keys[:k]]


class MultiQueryRetrieverWrapper:
    """Wraps a BM25 or Dense index to run search(original + paraphrases) and
    RRF-merge the result lists. Returns native hit objects (DenseHit /
    BM25Hit) so downstream handlers receive the same shape they expect.

    Paraphrase_fn(query) -> list[str].
    """

    def __init__(
        self,
        inner: Any,
        paraphrase_fn: Callable[[str], List[str]],
        *,
        k_each: int = 20,
    ) -> None:
        self._inner = inner
        self._paraphrase_fn = paraphrase_fn
        self._k_each = k_each

    def search(self, query: str, k: int = 20):
        try:
            paras = self._paraphrase_fn(query) or []
        except Exception as exc:
            log.debug("E3 paraphrase fn raised: %s", exc)
            paras = []
        if not paras:
            return self._inner.search(query, k=k)
        per = [self._inner.search(query, k=self._k_each)]
        for pq in paras:
            try:
                per.append(self._inner.search(pq, k=self._k_each))
            except Exception as exc:
                log.debug("E3 inner search raised on paraphrase: %s", exc)
        return _rrf_merge_hits(per, k=k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ===========================================================================
# Summary helper for telemetry
# ===========================================================================

def active_enhancers() -> dict[str, bool]:
    return {
        "E1_concept_amendment": is_e1_enabled(),
        "E2_nli_reverse":       is_e2_enabled(),
        "E3_paraphrase":        is_e3_enabled(),
        "E4_hyde":              is_e4_enabled(),
    }
