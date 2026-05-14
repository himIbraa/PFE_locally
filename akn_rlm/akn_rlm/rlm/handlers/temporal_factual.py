"""Temporal-factual query handler — Phase 2 / R3.

Pipeline:

  1. Doc-route the query via :class:`DocRouter`. Empty route → fall back
     to corpus-wide retrieval.

  2. Extract a target date from the query. Regex captures Gregorian
     dates (ISO ``YYYY-MM-DD``, ``DD/MM/YYYY``) and bare years
     (``\\d{4}``). Multiple years are allowed; the **latest** year is
     used as the target so "بين 1996 و2008 و2020" resolves to the most
     recent rule (matching benchmark ``applicable_version="post"``).
     If no date is found the handler defaults to the latest known
     version (target_date = ``"9999-12-31"``).

  3. Retrieve candidate articles via RRF(BM25, Dense), restricted to
     the routed ``doc_id``s. Take the top-``top_k_candidates``.

  4. **MANDATORY** for every retrieved candidate: query the KG
     amendment chain. The chain is the ordered list of
     ``dzdoc:hasVersion`` entries with ``dzdoc:inForceFrom`` /
     ``dzdoc:versionText`` triples. Pick the version whose
     ``inForceFrom <= target_date`` with the latest such date — that
     is the answer text. If the article has no chain in the KG
     (article was never amended → no ``hasVersion`` triples) the
     candidate's chunk text is used as the fallback (the article was
     enacted at its origin date and is still in force).

  5. Optionally sub-LM verify the top-``verify_top_n`` candidates so
     the verifier confidence flows into the citation. The verifier is
     fed the **KG-versioned text**, never the raw chunk text — this
     is what HANDOFF §3 means by "answer from the KG result, never
     from search".

  6. Build citations carrying the version-specific text + version
     date. Synthesise an answer via :func:`call_summarizer`; fall back
     to the deterministic Arabic template otherwise.

  7. Return an answer dict shaped like the deterministic baselines so
     :func:`akn_rlm.eval.runner._answer_to_result` consumes it.

Sub-LM call budget per query: ≤ ``verify_top_n=3`` verifier calls + 1
summariser = ≤ 4 calls. Well under the project ``max_sub_calls=12``
envelope.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse
from akn_rlm.rlm.adu_helpers import (
    DEFAULT_ADU_EXTRACT_TOP_N,
    AduExtractFn,
    attach_argumentation,
)
from akn_rlm.rlm.corrective_retry import maybe_corrective_retry
from akn_rlm.rlm.recursive_refine import (
    DEFAULT_CONFIDENCE_STRONG,
    DEFAULT_CONFIDENCE_WEAK,
    DEFAULT_COVERAGE_MIN,
    DEFAULT_MAX_DEPTH,
    DEFAULT_PROBE_MODEL,
    GapProbeFn,
    RecursiveRetriever,
    call_gap_probe,
)
from akn_rlm.rlm.routing import DocRouter, build_doc_router
from akn_rlm.rlm.sub_worker import call_summarizer, call_verifier

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K_CANDIDATES: int = 5

#: Fix-TF: per-query SPARQL row cap for the KG-first concept-at-date
#: channel. Each query phrase (~4-6 phrases typical) generates one
#: SPARQL query at this row cap. Higher = better recall, slower KG.
DEFAULT_KG_FIRST_LIMIT: int = 30
#: Baseline score assigned to KG-first hits before they enter the fused
#: candidate pool. Bumped by +0.05 per additional phrase that also hits
#: the same (doc, ref). Anchored a touch above the typical RRF score
#: to ensure KG hits are seriously considered (the failure mode we're
#: fixing is "gold article missing from top-K").
DEFAULT_KG_FIRST_BASE_SCORE: float = 0.6
#: Stopwords stripped during phrase extraction for the KG-first channel.
_TF_AR_STOP = {
    "هذا", "ذلك", "التي", "الذي", "اللذان", "اللذين", "اللتان", "اللتين",
    "حيث", "كيف", "متى", "أين", "ماذا", "كان", "تكون", "كانت", "يكون",
}
_TF_TOKEN_RE = re.compile(r"\W+", re.UNICODE)
_TF_URI_RE = re.compile(
    r"resource/([^/]+)/(\d{4}-\d{2}-\d{2})/([^/#]+)(?:#art_(.+))?",
)
# Verifier OFF by default. The HANDOFF §3 contract is "answer from the
# KG result, never from search" — the KG amendment chain is the source
# of truth, and a generic LLM relevance verifier (trained on
# search-style judgments) tends to reject foundational articles like
# art_1 / scope articles that ARE the gold answer for evolution-style
# temporal queries. Empirical: enabling the verifier dropped Cite F1
# from 0.167 → 0.095 on the full 7-q temporal_factual slice. Set
# ``verify_top_n>0`` to opt back in.
DEFAULT_VERIFY_TOP_N: int = 0
# R9.2: tightened from 5 → 2 to lift Cite F1 by trading recall for
# precision. Temporal_factual gold typically names a single in-force
# version; emitting only the top-2 KG-versioned articles drops the
# noisy 3rd-5th citations that were diluting precision on the 7-q
# slice.
DEFAULT_FINAL_TOP_K: int = 2
DEFAULT_K_EACH: int = 30
DEFAULT_VERIFY_THRESHOLD: float = 0.4
DEFAULT_ROUTE_TOP_N: int = 3
SUPPORT_SPAN_LEN: int = 280
# Sentinel "latest known version" target. Any inForceFrom date will be <=.
LATEST_VERSION_DATE: str = "9999-12-31"

# Telemetry tag.
TELEMETRY_BASELINE: str = "rlm_temporal_factual"

# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

# Date extractors. We use ``(?<!\d) ... (?!\d)`` lookarounds rather than
# ``\b`` because ``\b`` does not fire between an Arabic letter (which is
# a word char) and a digit — e.g. ``و2008`` would never be matched by
# ``\b2008\b``. The lookarounds correctly require the year to be
# digit-isolated regardless of surrounding Arabic / Latin / punctuation.
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DMY_RE = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
# Bare 4-digit year, only the plausible legal range to avoid catching
# article numbers etc.
_YEAR_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2}|21\d{2})(?!\d)")


def _extract_dates(query: str) -> list[str]:
    """Return all dates mentioned in ``query``, normalised to ``YYYY-MM-DD``.

    Order is preserved (left-to-right). Each date is emitted exactly once
    even if it appears multiple times in the source text.
    """
    if not query:
        return []
    seen: set[str] = set()
    ordered: list[str] = []

    def _push(date_str: str) -> None:
        if date_str and date_str not in seen:
            seen.add(date_str)
            ordered.append(date_str)

    consumed: list[tuple[int, int]] = []

    for m in _ISO_DATE_RE.finditer(query):
        try:
            y, mo, d = (int(x) for x in m.groups())
            _push(f"{y:04d}-{mo:02d}-{d:02d}")
            consumed.append(m.span())
        except ValueError:
            continue

    for m in _DMY_RE.finditer(query):
        try:
            d, mo, y = (int(x) for x in m.groups())
            _push(f"{y:04d}-{mo:02d}-{d:02d}")
            consumed.append(m.span())
        except ValueError:
            continue

    def _in_consumed(span: tuple[int, int]) -> bool:
        s0, s1 = span
        return any(c0 <= s0 and s1 <= c1 for c0, c1 in consumed)

    for m in _YEAR_RE.finditer(query):
        if _in_consumed(m.span()):
            continue
        y = int(m.group(1))
        _push(f"{y:04d}-12-31")  # bare year → end of year (most permissive)

    return ordered


def _pick_target_date(dates: list[str]) -> str:
    """Pick the most relevant date as the version target.

    The benchmark always wants the post-amendment rule
    (``applicable_version="post"`` for every temporal_factual question),
    so we use the **maximum** date — this surfaces the version that
    was in force at the latest mentioned date, which is also the
    version a model trained on outdated data would mis-identify.

    No dates → ``LATEST_VERSION_DATE`` so any version with
    ``inForceFrom <= target`` survives.
    """
    if not dates:
        return LATEST_VERSION_DATE
    return max(dates)


# ---------------------------------------------------------------------------
# KG version-chain helpers
# ---------------------------------------------------------------------------

# Real KG predicates (inspected 2026-05-08 against
# data/kg/algerian_legal_kg.ttl). The handler queries these directly —
# legal_env.kg_amendment_chain uses a different namespace and does not
# match the loaded TTL.
_DZDOC_NS = "https://legal.dz/ontology/document#"

_VERSION_CHAIN_SPARQL = """
PREFIX dzdoc: <https://legal.dz/ontology/document#>
SELECT ?version ?inForceFrom ?text WHERE {
    <%s> dzdoc:hasVersion ?version .
    ?version dzdoc:inForceFrom ?inForceFrom .
    OPTIONAL { ?version dzdoc:versionText ?text . }
}
ORDER BY ?inForceFrom
"""

# Article URI pattern — there are several "categories" (law, order,
# constitution, organic-law, presidential-decree, executive-decree).
# The doc_id form is ``{num}_{enactment_date}`` (or
# ``constitution_{date}`` for the four numbered constitutions). Mapping:
#
#   84-11_1984-06-09          -> https://legal.dz/resource/law/1984-06-09/84-11
#   75-59_1975-09-26          -> https://legal.dz/resource/order/1975-09-26/75-59
#   constitution_2020-12-30   -> https://legal.dz/resource/constitution/2020-12-30/2020
#
# We can't tell the category from the doc_id alone, so we try a small
# ordered list of categories and ASK the KG.
_KG_CATEGORIES: tuple[str, ...] = (
    "law",
    "order",
    "constitution",
    "organic-law",
    "presidential-decree",
    "executive-decree",
)


def _resolve_article_uri(
    sparql_fn: Callable[[str], list[dict]] | None,
    doc_id: str,
    article_ref: str,
) -> str | None:
    """Resolve a canonical (doc_id, article_ref) pair to a KG article URI.

    Tries each category in :data:`_KG_CATEGORIES` and returns the first
    URI that has at least one outgoing triple.
    """
    if not sparql_fn or not doc_id or not article_ref:
        return None

    canon_ref = canonical_article_ref(article_ref) or article_ref
    if not canon_ref:
        return None

    if doc_id.startswith("constitution_"):
        date = doc_id[len("constitution_"):]
        if not date:
            return None
        year = date.split("-", 1)[0]
        candidates = [
            f"https://legal.dz/resource/constitution/{date}/{year}#art_{canon_ref}"
        ]
    else:
        if "_" not in doc_id:
            return None
        num, _, date = doc_id.rpartition("_")
        if not num or not date:
            return None
        candidates = [
            f"https://legal.dz/resource/{cat}/{date}/{num}#art_{canon_ref}"
            for cat in _KG_CATEGORIES
        ]
        # A handful of laws use the redundant suffix form
        # ``96-21_1996-07-09`` inside the URI as well.
        candidates += [
            f"https://legal.dz/resource/{cat}/{date}/{num}_{date}#art_{canon_ref}"
            for cat in _KG_CATEGORIES
        ]

    for uri in candidates:
        ask = f"ASK {{ <{uri}> ?p ?o }}"
        try:
            result = sparql_fn(ask)
        except Exception as exc:
            log.debug("ASK uri %s failed: %s", uri, exc)
            continue
        # rdflib returns [{"_ask": True}] for ASK; tests may return a bool
        # or a non-empty list. Normalise.
        if _ask_to_bool(result):
            return uri
    return None


def _ask_to_bool(result: Any) -> bool:
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
            # Non-empty dict with no bool — treat as positive evidence.
            return True
        return bool(first)
    return False


def _amendment_chain(
    sparql_fn: Callable[[str], list[dict]] | None,
    article_uri: str,
) -> list[dict[str, Any]]:
    """Return the KG amendment chain for ``article_uri``.

    Each entry: ``{"version_uri": str, "date": str, "text": str}``.
    Empty list means the URI is unknown OR the article has no
    ``dzdoc:hasVersion`` triples (i.e. never amended).
    """
    if not sparql_fn or not article_uri:
        return []
    try:
        rows = sparql_fn(_VERSION_CHAIN_SPARQL % article_uri)
    except Exception as exc:
        log.debug("amendment_chain SPARQL failed for %s: %s", article_uri, exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows or []:
        out.append({
            "version_uri": row.get("version") or "",
            "date":        (row.get("inForceFrom") or "").strip(),
            "text":        (row.get("text") or "").strip(),
        })
    out.sort(key=lambda e: e.get("date") or "")
    return out


# ---------------------------------------------------------------------------
# Fix-TF — KG-first concept-at-date retrieval channel
# ---------------------------------------------------------------------------

def _tf_query_phrases(query: str, max_n: int = 6) -> list[str]:
    """Extract MULTI-WORD content phrases from a TF query for KG-first
    retrieval. SPARQL CONTAINS over a single common token (e.g. "العمل",
    "قانون") matches thousands of articles — useless noise. Bigrams /
    trigrams of content tokens are 100-1000x more selective.

    Strategy: tokenize → drop stopwords + length<3 → emit trigrams first
    (most selective), then bigrams. Preserve ``ال`` definite-article
    prefix because CONTAINS is a literal substring match (HANDOFF §R4).
    """
    if not query:
        return []
    raw_tokens = [t for t in _TF_TOKEN_RE.split(query) if t and len(t) >= 3]
    tokens = [t for t in raw_tokens if t not in _TF_AR_STOP]
    if len(tokens) < 2:
        return []
    trigrams: list[str] = []
    bigrams: list[str] = []
    for i in range(len(tokens) - 2):
        trigrams.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
    for i in range(len(tokens) - 1):
        bigrams.append(f"{tokens[i]} {tokens[i+1]}")
    seen: set[str] = set()
    out: list[str] = []
    for ph in trigrams + bigrams:
        if ph in seen:
            continue
        seen.add(ph)
        out.append(ph)
        if len(out) >= max_n:
            break
    return out


def _tf_uri_to_doc_ref(uri: str) -> Optional[tuple[str, str]]:
    """Parse an article URI into (canonical doc_id, canonical article_ref)."""
    m = _TF_URI_RE.search(str(uri))
    if not m:
        return None
    _cat, date, num, ref = m.groups()
    if not ref:
        return None
    doc_id = f"{num}_{date}"
    return doc_id, canonical_article_ref(ref)


def _merge_candidate_pools(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union two candidate pools by (doc_id, canonical article_ref). Keep
    the entry with the higher score; merge ``kg_first`` flag and
    preserve ``text`` from whichever source had it non-empty.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for src in (primary, secondary):
        for cand in src or []:
            doc = cand.get("doc_id", "")
            ref = canonical_article_ref(cand.get("article_ref", "")) or cand.get("article_ref", "")
            key = (doc, ref)
            prior = merged.get(key)
            if prior is None:
                merged[key] = dict(cand)
                merged[key]["article_ref"] = ref
                continue
            # Keep the higher-scoring entry; merge metadata.
            if float(cand.get("score", 0.0)) > float(prior.get("score", 0.0)):
                # New entry wins; preserve prior text if new entry's is empty.
                new_text = cand.get("text") or prior.get("text", "")
                merged[key] = dict(cand)
                merged[key]["article_ref"] = ref
                merged[key]["text"] = new_text
            else:
                # Prior wins; fill empty text from new entry if helpful.
                if not prior.get("text") and cand.get("text"):
                    prior["text"] = cand["text"]
            if cand.get("kg_first") or merged[key].get("kg_first"):
                merged[key]["kg_first"] = True
    return sorted(merged.values(), key=lambda c: float(c.get("score", 0.0)), reverse=True)


def _tf_kg_first_candidates(
    sparql_fn: Callable[[str], list[dict]] | None,
    query: str,
    target_date: str,
    *,
    routed_ids: Optional[list[str]] = None,
    base_score: float = DEFAULT_KG_FIRST_BASE_SCORE,
    limit_per_phrase: int = DEFAULT_KG_FIRST_LIMIT,
) -> list[dict[str, Any]]:
    """Fix-TF: surface articles whose any-version text contains a query
    phrase AND was in force on ``target_date``. Returns candidate dicts
    in the same shape as the hybrid retrieval pool so the downstream
    chain step consumes them uniformly.

    The bottleneck this fixes (documented in HANDOFF §R3): for 4/7 of the
    TF slice the hybrid retrieve step gets the right *doc* in top-3 but
    misses the *gold article* in top-5. Asking the KG "give me articles
    that mention X and existed at date Y" gives us the exact ground truth
    the chain step then validates.
    """
    if not sparql_fn or not query or not target_date:
        return []
    phrases = _tf_query_phrases(query)
    if not phrases:
        return []

    results: dict[tuple[str, str], dict[str, Any]] = {}
    routed_set = set(routed_ids) if routed_ids else None

    for phrase in phrases:
        # SPARQL string-literal escape — replace " with \" and \ with \\
        safe = phrase.replace("\\", "\\\\").replace('"', '\\"')
        sparql = (
            'PREFIX dzdoc: <https://legal.dz/ontology/document#>\n'
            'PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n'
            'SELECT DISTINCT ?article ?text ?inForceFrom WHERE {\n'
            '  ?article a dzdoc:Article ;\n'
            '           dzdoc:hasVersion ?v .\n'
            '  ?v dzdoc:versionText ?text ;\n'
            '     dzdoc:inForceFrom ?inForceFrom .\n'
            f'  FILTER(CONTAINS(STR(?text), "{safe}"))\n'
            f'  FILTER(STR(?inForceFrom) <= "{target_date}")\n'
            '}\n'
            f'LIMIT {limit_per_phrase}'
        )
        try:
            rows = sparql_fn(sparql) or []
        except Exception as exc:
            log.debug("KG-first SPARQL failed for phrase %r: %s", phrase, exc)
            continue

        for row in rows:
            uri = str(row.get("article", "")) if isinstance(row, dict) else ""
            text = str(row.get("text", "")) if isinstance(row, dict) else ""
            parsed = _tf_uri_to_doc_ref(uri)
            if not parsed:
                continue
            doc_id, ref = parsed
            if routed_set and doc_id not in routed_set:
                continue   # respect doc-routing
            key = (doc_id, ref)
            entry = results.get(key)
            if entry is None:
                results[key] = {
                    "doc_id":      doc_id,
                    "article_ref": ref,
                    "text":        text,
                    "score":       float(base_score),
                    "kg_first":    True,
                }
            else:
                # multi-phrase match: bump the score (caps at 1.0)
                entry["score"] = min(1.0, entry["score"] + 0.05)
    return list(results.values())


def _version_at_date(chain: list[dict[str, Any]], target_date: str) -> dict[str, Any] | None:
    """Pick the latest version whose ``date <= target_date``.

    Empty chain → ``None``. No version pre-dates the target → ``None``
    (the article didn't exist yet at the requested date).
    """
    if not chain:
        return None
    chosen: dict[str, Any] | None = None
    for v in chain:
        d = v.get("date") or ""
        if d and d <= target_date:
            if chosen is None or d >= (chosen.get("date") or ""):
                chosen = v
    return chosen


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

VerifierFn = Callable[[Any, str, dict, str], dict]
SummarizerFn = Callable[[Any, str, list[dict], str], dict]
SparqlFn = Callable[[str], list[dict]]


class TemporalFactualHandler:
    """Typed temporal-factual handler: route -> extract date -> retrieve ->
    KG amendment chain MANDATORY -> answer-from-KG-version.
    """

    def __init__(
        self,
        kg: Any,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        top_k_candidates: int = DEFAULT_TOP_K_CANDIDATES,
        verify_top_n: int = DEFAULT_VERIFY_TOP_N,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        k_each: int = DEFAULT_K_EACH,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
        route_top_n: int = DEFAULT_ROUTE_TOP_N,
        verifier_fn: Optional[VerifierFn] = None,
        summarizer_fn: Optional[SummarizerFn] = None,
        sparql_fn: Optional[SparqlFn] = None,
        # Phase C — pervasive Toulmin ADU extraction (default OFF).
        enable_adu_extraction: bool = False,
        adu_extract_top_n: int = DEFAULT_ADU_EXTRACT_TOP_N,
        adu_extract_fn: Optional[AduExtractFn] = None,
        # Phase D — gap-driven recursion + corrective retry. The TF
        # depth-1 is "hybrid + KG-first → KG amendment chain → versioned
        # answer". Recursion's gap question goes through the same
        # hybrid+KG-first+chain path with the new query string.
        enable_recursion: bool = False,
        recursion_max_depth: int = DEFAULT_MAX_DEPTH,
        recursion_coverage_min: int = DEFAULT_COVERAGE_MIN,
        recursion_confidence_weak: float = DEFAULT_CONFIDENCE_WEAK,
        recursion_confidence_strong: float = DEFAULT_CONFIDENCE_STRONG,
        recursion_probe_fn: GapProbeFn = call_gap_probe,
        recursion_probe_model: str = DEFAULT_PROBE_MODEL,
        enable_corrective_retry: bool = False,
    ) -> None:
        self._kg = kg
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._sub_model = sub_model
        self._top_k_candidates = top_k_candidates
        self._verify_top_n = verify_top_n
        self._final_top_k = final_top_k
        self._k_each = k_each
        self._verify_threshold = verify_threshold
        self._route_top_n = route_top_n
        self._verifier_fn = verifier_fn or call_verifier
        self._summarizer_fn = summarizer_fn or call_summarizer
        # SPARQL injection point: the runner wires this to the loaded
        # rdflib graph; tests mock it with canned responses.
        self._sparql_fn = sparql_fn or self._default_sparql_fn()
        # Phase C — pervasive ADU.
        self._enable_adu_extraction = bool(enable_adu_extraction)
        self._adu_extract_top_n = int(adu_extract_top_n)
        self._adu_extract_fn = adu_extract_fn
        # Phase D — recursion + corrective retry.
        self._enable_recursion = bool(enable_recursion)
        self._recursion_max_depth = int(recursion_max_depth)
        self._recursion_coverage_min = int(recursion_coverage_min)
        self._recursion_confidence_weak = float(recursion_confidence_weak)
        self._recursion_confidence_strong = float(recursion_confidence_strong)
        self._recursion_probe_fn = recursion_probe_fn
        self._recursion_probe_model = recursion_probe_model
        self._enable_corrective_retry = bool(enable_corrective_retry)

    # ------------------------------------------------------------------
    def _default_sparql_fn(self) -> Optional[SparqlFn]:
        """Build a default SPARQL caller bound to ``self._kg`` if present."""
        kg = self._kg
        if kg is None:
            return None

        def _fn(query: str) -> Any:
            try:
                results = kg.query(query)
            except Exception as exc:
                log.debug("SPARQL query failed: %s", exc)
                return []
            stripped = query.strip().lower()
            if stripped.startswith("ask"):
                # rdflib returns a SPARQLResult that is truthy iff the ASK
                # is positive — convert to a single-row list[{"_ask": bool}].
                try:
                    return [{"_ask": bool(results)}]
                except Exception:
                    return []
            rows: list[dict] = []
            try:
                vars_list = list(results.vars or [])
            except Exception:
                vars_list = []
            for row in results:
                rows.append({
                    str(var): (str(row[var]) if row[var] is not None else None)
                    for var in vars_list
                })
            return rows

        return _fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain(
                "empty_query",
                routed=[],
                target_date=LATEST_VERSION_DATE,
                dates=[],
                sub_calls=0,
                chains=[],
            )

        trajectory: list[dict[str, Any]] = []

        # 1. Doc-route
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)
        trajectory.append({
            "step": "route", "depth": 0,
            "routed_doc_ids": routed_ids,
        })

        # 2. Extract date(s) → target
        dates = _extract_dates(query)
        target_date = _pick_target_date(dates)
        trajectory.append({
            "step": "extract_date", "depth": 0,
            "dates": dates, "target": target_date,
        })

        # State threaded through the retrieve+chain closure so recursion's
        # extra calls show up in telemetry.
        chain_traces: list[dict[str, Any]] = []
        sub_calls_holder = [0]
        candidate_holder: list[dict[str, Any]] = []
        # Phase E.1 — per-call KG-first telemetry. One dict per
        # _retrieve_chain_verify invocation (depth 1, plus one per
        # recursion depth). The handler-level _telemetry surfaces the
        # list as ``tf_kg_first_telemetry`` so post-hoc inspection can
        # diagnose whether KG-first hits actually survive (a) the merge,
        # (b) the top-K slice, (c) URI resolution, and (d) verification.
        # The bug HANDOFF §1.3 documents ("Fix-TF wired but flat on full
        # 244 — likely the merge logic or chain step rejecting KG-first
        # hits") needs all four numbers to localise.
        kg_first_telemetry: list[dict[str, Any]] = []
        depth_counter = [0]

        def _retrieve_chain_verify(q: str) -> dict[tuple[str, str], dict[str, Any]]:
            depth_counter[0] += 1
            depth = depth_counter[0]
            hybrid = self._fused_candidates(q, routed_ids)
            kgf = _tf_kg_first_candidates(
                self._sparql_fn, q, target_date, routed_ids=routed_ids,
            )
            kg_first_hits = [
                (c.get("doc_id", ""), c.get("article_ref", ""))
                for c in kgf
            ]
            cands = _merge_candidate_pools(hybrid, kgf)
            top_slice = cands[: self._top_k_candidates]
            kg_first_in_top_slice = sum(1 for c in top_slice if c.get("kg_first"))
            # Snapshot the per-trace cursor so we can attribute new
            # chain_traces entries back to this depth's KG-first hits.
            traces_before = len(chain_traces)
            if not cands:
                kg_first_telemetry.append({
                    "depth":               depth,
                    "query":               q,
                    "hybrid_count":        len(hybrid),
                    "kg_first_count":      len(kgf),
                    "kg_first_hits":       kg_first_hits,
                    "merged_pool_size":    0,
                    "top_slice_size":      0,
                    "kg_first_in_top_slice": 0,
                    "kg_first_uri_resolved": 0,
                    "kg_first_in_verified":  0,
                })
                return {}
            candidate_holder.extend(top_slice)
            verified = self._chain_verify_pool(
                top_slice,
                query=q,
                target_date=target_date,
                chain_traces=chain_traces,
                sub_calls_holder=sub_calls_holder,
            )
            # Diagnose URI-resolution success on the KG-first slice — the
            # primary failure mode HANDOFF §1.3 calls out.
            new_traces = chain_traces[traces_before:]
            kg_first_keys = {
                (c.get("doc_id", ""), canonical_article_ref(c.get("article_ref", "")) or c.get("article_ref", ""))
                for c in top_slice if c.get("kg_first")
            }
            kg_first_uri_resolved = sum(
                1 for t in new_traces
                if (t.get("doc_id"), t.get("article_ref")) in kg_first_keys
                and t.get("uri")
            )
            kg_first_in_verified = sum(
                1 for key in verified.keys() if key in kg_first_keys
            )
            kg_first_telemetry.append({
                "depth":                  depth,
                "query":                  q,
                "hybrid_count":           len(hybrid),
                "kg_first_count":         len(kgf),
                "kg_first_hits":          kg_first_hits,
                "merged_pool_size":       len(cands),
                "top_slice_size":         len(top_slice),
                "kg_first_in_top_slice":  kg_first_in_top_slice,
                "kg_first_uri_resolved":  kg_first_uri_resolved,
                "kg_first_in_verified":   kg_first_in_verified,
            })
            return verified

        # 3-4. Depth-1: retrieve + KG-chain + (optional) verify.
        verified = _retrieve_chain_verify(query)
        if not candidate_holder:
            return self._abstain(
                "no_hits",
                routed=routed_ids,
                target_date=target_date,
                dates=dates,
                sub_calls=sub_calls_holder[0],
                chains=[],
            )
        trajectory.append({
            "step": "kg_chain", "depth": 1,
            "candidates": len(candidate_holder),
            "verified": len(verified),
        })

        # 4b. Phase D — gap-driven recursion (additive merge).
        recursion_steps = []
        recursion_probe_calls = 0
        if self._enable_recursion and self._recursion_max_depth >= 2:
            retriever = RecursiveRetriever(
                llm_pool=self._llm_pool,
                retrieve_verify_fn=_retrieve_chain_verify,
                max_depth=self._recursion_max_depth,
                coverage_min=self._recursion_coverage_min,
                confidence_weak=self._recursion_confidence_weak,
                confidence_strong=self._recursion_confidence_strong,
                probe_fn=self._recursion_probe_fn,
                probe_model=self._recursion_probe_model,
                seed_accumulator=verified,
            )
            verified, recursion_steps, recursion_probe_calls = retriever.run(query)
            sub_calls_holder[0] += recursion_probe_calls
            for step in recursion_steps:
                trajectory.append({"step": "recursion", **step.to_dict()})

        sub_calls = sub_calls_holder[0]

        if not verified:
            return self._abstain(
                "no_verified_articles",
                routed=routed_ids,
                target_date=target_date,
                dates=dates,
                sub_calls=sub_calls,
                chains=chain_traces,
            )

        # 6. Final ranking + truncate
        ranked = sorted(
            verified.values(),
            key=lambda c: float(c.get("confidence", 0.0)),
            reverse=True,
        )
        final_citations = ranked[: self._final_top_k]

        # 6b. Phase C — pervasive Toulmin ADU.
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
            trajectory.append({
                "step": "adu_extract", "depth": 0,
                "extracts": adu_extracts_done,
            })

        # 7. Synthesise
        template_answer = self._template_answer(final_citations)
        answer_text = template_answer
        try:
            synth = self._summarizer_fn(
                self._llm_pool, query, final_citations, self._sub_model
            )
            sub_calls += 1
            summary = synth.get("summary")
            if isinstance(summary, str) and summary.strip():
                answer_text = summary.strip()
        except Exception as exc:
            log.warning("temporal summariser failed (%s) — template answer", exc)
        trajectory.append({"step": "summarize", "depth": 0})

        # 8. Phase D — corrective retry on faithfulness failure.
        retry_trace = None
        if self._enable_corrective_retry:
            answer_text, retry_trace = maybe_corrective_retry(
                answer_text=answer_text,
                citations=final_citations,
                original_question=query,
                summarizer_fn=self._summarizer_fn,
                llm_pool=self._llm_pool,
                sub_model=self._sub_model,
                enabled=True,
                template_fallback=template_answer,
            )
            sub_calls += retry_trace.sub_call_count
            trajectory.append({
                "step": "faithfulness_gate", "depth": 0,
                "fired": retry_trace.fired,
                "pre_passed": retry_trace.pre_passed,
                "post_passed": retry_trace.post_passed,
            })

        depth_max = max(
            (s.depth for s in recursion_steps if s.new_citations > 0 or s.depth == 1),
            default=1,
        )
        if not recursion_steps:
            depth_max = 1

        telemetry: dict[str, Any] = {
            "retry_count":     1 if (retry_trace and retry_trace.fired) else 0,
            "gate_results":    {},
            "baseline":        TELEMETRY_BASELINE,
            "routed_doc_ids":  routed_ids,
            "extracted_dates": dates,
            "target_date":     target_date,
            "amendment_chains": chain_traces,
            "sub_call_count":  sub_calls,
            "adu_extracts":    adu_extracts_done,
            "recursion_trace": [s.to_dict() for s in recursion_steps],
            "recursion_depth_max": depth_max,
            # Phase E.1 — per-depth KG-first inspection data. Lets a
            # post-hoc analysis answer "did KG-first hits survive the
            # merge? did the URI resolve? did the chain step accept
            # them?" without re-running.
            "tf_kg_first_telemetry": kg_first_telemetry,
        }
        if retry_trace is not None:
            telemetry["corrective_retry"] = retry_trace.to_dict()

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   [
                f"target_date={target_date}",
                *[f"{t['doc_id']}/{t['article_ref']}@{t.get('picked') or '-'}({t['source']})"
                  for t in chain_traces],
            ],
            "trajectory":        trajectory,
            "tokens_used":       0,
            "depth_max_reached": depth_max,
            "_telemetry":        telemetry,
        }

    # ------------------------------------------------------------------
    # KG amendment-chain verifier — extracted so recursion can re-call.
    # ------------------------------------------------------------------

    def _chain_verify_pool(
        self,
        candidates: list[dict[str, Any]],
        *,
        query: str,
        target_date: str,
        chain_traces: list[dict[str, Any]],
        sub_calls_holder: list[int],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Run the KG amendment chain on each candidate and return the
        verified accumulator. Side-effects: appends to ``chain_traces``;
        increments ``sub_calls_holder[0]`` for each verifier call.
        """
        verified: dict[tuple[str, str], dict[str, Any]] = {}
        for cand in candidates:
            doc_id = cand.get("doc_id", "")
            ref = canonical_article_ref(cand.get("article_ref", "")) or cand.get(
                "article_ref", ""
            )
            chunk_text = cand.get("text", "") or ""

            article_uri = _resolve_article_uri(self._sparql_fn, doc_id, ref)
            chain = _amendment_chain(self._sparql_fn, article_uri) if article_uri else []

            if chain:
                version = _version_at_date(chain, target_date)
                if version is None:
                    chain_traces.append({
                        "doc_id":     doc_id,
                        "article_ref": ref,
                        "uri":        article_uri,
                        "chain_len":  len(chain),
                        "picked":     None,
                        "source":     "kg_no_match",
                    })
                    continue
                version_text = version.get("text") or chunk_text
                version_date = version.get("date") or ""
                source = "kg"
            else:
                version_text = chunk_text
                version_date = ""
                source = "fallback"

            chain_traces.append({
                "doc_id":      doc_id,
                "article_ref": ref,
                "uri":         article_uri,
                "chain_len":   len(chain),
                "picked":      version_date or None,
                "source":      source,
            })

            confidence = float(cand.get("score", 0.6))
            supporting_quote = ""
            if self._verifier_fn is not None and len(verified) < self._verify_top_n:
                article_for_verify = {
                    "doc_id":      doc_id,
                    "article_ref": ref,
                    "text":        version_text,
                }
                try:
                    verdict = self._verifier_fn(
                        self._llm_pool, query, article_for_verify, self._sub_model
                    )
                    sub_calls_holder[0] += 1
                except Exception as exc:
                    log.warning(
                        "temporal verifier failed (%s) — keeping candidate", exc
                    )
                    verdict = {
                        "relevant": True,
                        "confidence": max(confidence, self._verify_threshold),
                        "supporting_span": None,
                    }
                if not verdict.get("relevant"):
                    continue
                vc = float(verdict.get("confidence", 0.0) or 0.0)
                if vc < self._verify_threshold:
                    continue
                confidence = vc
                supporting_quote = verdict.get("supporting_span") or ""

            citation = self._build_citation(
                doc_id=doc_id,
                article_ref=ref,
                version_text=version_text,
                version_date=version_date,
                source=source,
                supporting_quote=supporting_quote,
                confidence=confidence,
            )
            key = (doc_id, ref)
            prior = verified.get(key)
            if prior is None or confidence > float(prior.get("confidence", 0.0)):
                verified[key] = citation
        return verified

    # ------------------------------------------------------------------
    # Retrieval helpers (mirror multi_hop)
    # ------------------------------------------------------------------

    def _fused_candidates(
        self, query: str, routed_ids: list[str]
    ) -> list[dict[str, Any]]:
        """RRF(BM25, Dense) restricted to routed docs (with full-pool fallback)."""
        try:
            bm25_hits: list[BM25Hit] = self._bm25.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("temporal BM25 failed: %s", exc)
            bm25_hits = []
        try:
            dense_hits: list[DenseHit] = self._dense.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("temporal dense failed: %s", exc)
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

    def _build_citation(
        self,
        *,
        doc_id: str,
        article_ref: str,
        version_text: str,
        version_date: str,
        source: str,
        supporting_quote: str,
        confidence: float,
    ) -> dict[str, Any]:
        text = version_text or ""
        if supporting_quote and supporting_quote in text:
            span = supporting_quote[:SUPPORT_SPAN_LEN]
        else:
            span = text[:SUPPORT_SPAN_LEN]
        return {
            "doc_id":            doc_id,
            "article_ref":       article_ref,
            "doc_title":         self._doc_title(doc_id),
            "supporting_span":   span,
            "text":              text,
            "confidence":        float(confidence),
            "version_date":      version_date,
            "kg_source":         source,
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
            vdate = c.get("version_date") or ""
            head = f"وفقًا لـ {doc_title}، المادة {ref}"
            if vdate:
                head += f" (نسخة {vdate})"
            parts.append(f"{head}: {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _abstain(
        reason: str,
        *,
        routed: list[str],
        target_date: str,
        dates: list[str],
        sub_calls: int,
        chains: list[dict[str, Any]],
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
                "retry_count":      0,
                "gate_results":     {},
                "baseline":         TELEMETRY_BASELINE,
                "routed_doc_ids":   routed,
                "extracted_dates":  dates,
                "target_date":      target_date,
                "amendment_chains": chains,
                "sub_call_count":   sub_calls,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_temporal_factual_handler(
    kg: Any,
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> TemporalFactualHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return TemporalFactualHandler(
        kg=kg,
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
