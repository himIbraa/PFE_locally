"""Conceptual-definitional query handler — Phase 2 / R4.

Pipeline:

  1. Doc-route the query via :class:`DocRouter`. Empty route → fall back
     to corpus-wide retrieval.

  2. Extract concept phrases from the query — multi-word noun phrases
     between stopwords/question words first, single content tokens as
     fallback. The conceptual_definitional benchmark always names the
     concept explicitly ("ما الفرق بين الاتفاقية الجماعية...",
     "ما مفهوم الدفع بعدم الدستورية"), so phrase-level lookup is
     reliable when it works.

  3. **kg_entity_lookup first** (HANDOFF §3): for every concept phrase
     run a CONTAINS-style SPARQL against ``dzdoc:directlyContainedIn`` /
     ``dzdoc:text`` (the real KG schema — the existing
     ``graphrag.entity_lookup`` queries ``rdfs:label`` which doesn't
     match the loaded TTL, so we do this directly here). Article URIs
     are scored by phrase coverage and resolved back to canonical
     ``(doc_id, article_ref)`` pairs.

  4. **Fallback when KG misses**: if the KG returned no usable hits OR
     fewer than ``min_kg_hits`` articles, generate up to
     ``paraphrase_count`` paraphrases of the query via a single LLM
     call (one inline prompt → JSON list of paraphrases), search dense
     for each paraphrase, and RRF-fuse the results with the original
     BM25 + dense pools. This widens recall on questions where the
     concept is phrased unusually.

  5. Filter the candidate pool by the routed ``doc_id``s. If filtering
     wipes everything, fall back to the unrestricted pool (better than
     returning empty).

  6. **extract_adu** (Toulmin claim/ground/warrant/...) on the top-
     ``adu_extract_top_n`` candidates via :func:`akn_rlm.adu.extract`.
     The extracted claim+ground becomes the citation's
     ``supporting_span`` so the answer carries the *defining claim*,
     not a leading prefix of the article.

  7. Optionally sub-LM verify the top-``verify_top_n`` candidates so
     verifier confidence flows into the citation. The verifier is OFF
     by default — empirically, generic relevance verifiers reject
     foundational/scope articles that ARE the gold definition (same
     finding as R3).

  8. Synthesise a final answer over the ADU-enriched citations via
     :func:`call_summarizer`. Fall back to the deterministic Arabic
     template otherwise.

  9. Return an answer dict shaped like the deterministic baselines so
     :func:`akn_rlm.eval.runner._answer_to_result` consumes it.

Sub-LM call budget per query:

  ≤ 1 paraphraser (only on KG miss) +
  ≤ ``adu_extract_top_n`` ADU extractors (default 2) +
  ≤ ``verify_top_n`` verifiers (default 0) +
  1 summariser
  = ≤ 4 calls in the typical case (KG miss + 2 ADU + 1 summary)
  = ≤ 3 calls when KG hits (2 ADU + 1 summary).

Well under the project ``max_sub_calls=12`` envelope and aligned with
HANDOFF §3 "conceptual=4".
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from akn_rlm.adu import extract as adu_extract
from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse
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
# Verifier OFF by default (R3 lesson). The Toulmin ADU extraction is the
# discriminator — when claim/ground come back well-formed they are
# stronger evidence the article defines the concept than a generic
# search-relevance verdict. Set ``verify_top_n>0`` to opt back in.
DEFAULT_VERIFY_TOP_N: int = 0
# F4: reverted from R9.2's 2 back to F2's 5. CD gold often names 2-3
# articles (definition + ground + edge case); top-2 was dropping the
# 3rd article and regressed Cite F1 from 0.107 (F2) → 0.097 (R9.2 F3).
# TF stays at 2 (R9.2 there worked).
DEFAULT_FINAL_TOP_K: int = 5
DEFAULT_K_EACH: int = 30
DEFAULT_VERIFY_THRESHOLD: float = 0.4
DEFAULT_ROUTE_TOP_N: int = 3
DEFAULT_PARAPHRASE_COUNT: int = 3
DEFAULT_ADU_EXTRACT_TOP_N: int = 2
DEFAULT_MIN_KG_HITS: int = 1
DEFAULT_PHRASE_MIN_LEN: int = 4
DEFAULT_KG_LIMIT: int = 50
# KG bias: small additive boost applied to the post-RRF score of any
# fused candidate whose (doc_id, canonical article_ref) appears in the
# KG concept-search result set. Same magnitude / role as the B6
# KG-Hybrid baseline's bias — the B5 KG path on its own is a doc-level
# baseline that surfaces too many adjacent articles for definitional
# precision; keeping retrieval anchored on RRF and treating KG as a
# secondary signal is what beats B3 here.
DEFAULT_KG_BOOST: float = 0.05
SUPPORT_SPAN_LEN: int = 280

TELEMETRY_BASELINE: str = "rlm_conceptual_definitional"

# Article URIs in the KG follow:
#   https://legal.dz/resource/<type>/<date>/<num>[_<date>]#art_<X>[_<sub>...]
# Same shape as B5 KG baseline.
_URI_RE = re.compile(
    r"https://legal\.dz/resource/[^/]+/(?P<date>[0-9-]+)/(?P<num>[A-Za-z0-9_-]+)#"
    r"(?P<frag>art_[A-Za-z0-9_]+)"
)
_ART_SUFFIX_RE = re.compile(
    r"_(para|right|obligation|condition|permission|prohibition|action|role)"
    r"_[A-Za-z0-9_]+$"
)

# CONTAINS substring search using the REAL KG schema (dzdoc, not dznorm).
# Same approach as B5 but operates on multi-word phrases instead of
# single tokens — phrase-level matches are dramatically more selective
# for definitional queries.
_KG_PHRASE_SEARCH = """
PREFIX dzdoc:  <https://legal.dz/ontology/document#>

SELECT DISTINCT ?article ?text WHERE {{
    ?node dzdoc:directlyContainedIn ?article ;
          dzdoc:text                ?text .
    FILTER(CONTAINS(?text, "{phrase}"))
}}
LIMIT {limit}
"""

# Splits on anything that isn't a Unicode word character. With re.UNICODE
# flag, ``\w`` covers Arabic letters (U+0621-U+064A …) and digits but
# correctly excludes Arabic punctuation (``؟ ، ؛``) — important here
# because B5's wider regex preserves ``؟`` inside the Arabic block,
# which would propagate "الجماعية؟" into bigram literals and break
# CONTAINS lookups.
_TOKEN_SPLIT_RE = re.compile(r"\W+", flags=re.UNICODE)

# Conservative Arabic + Latin stopword list (subset of B5's; kept
# narrower to preserve more concept tokens for phrase building).
_STOPWORDS: frozenset[str] = frozenset({
    # Arabic question/connective words
    "ما", "هي", "هو", "هل", "ماذا", "لماذا", "كيف", "أين", "متى", "من",
    "ومن", "إلى", "عن", "على", "في", "مع", "كل", "بعض", "هذا", "هذه",
    "ذلك", "تلك", "الذي", "التي", "الذين", "كان", "كانت", "أن", "إن",
    "أو", "أم", "ثم", "بين", "حتى", "لكن", "غير", "بدون", "ضد",
    "وفقا", "وفقًا", "وفق", "بموجب", "نفس", "الفرق", "الجزائري",
    "الجزائر", "التشريع",
    # Latin
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "was",
    "were", "for", "on", "with", "as", "at", "by", "this", "that",
    "from", "what", "which", "who", "how", "why", "when", "where",
})

# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------


def _content_tokens(query: str) -> list[str]:
    """Tokenise ``query`` and drop stopwords/short tokens.

    Surface forms are preserved (``ال`` definite-article kept). Arabic
    KG text literals carry the ``ال`` prefix verbatim, and CONTAINS is a
    literal substring match — stripping the prefix would break phrase
    matches across "الاتفاقية الجماعية" because the embedded ``ال`` of
    the second token breaks the substring "اتفاقية جماعية".
    """
    out: list[str] = []
    for raw in _TOKEN_SPLIT_RE.split(query or ""):
        tok = raw.strip()
        if not tok or len(tok) < 3:
            continue
        if tok in _STOPWORDS:
            continue
        # Drop the ``ال``-stripped form too, to filter "الفرق" / "الجزائري"
        # without burning a slot on them.
        if tok.startswith("ال") and len(tok) > 4 and tok[2:] in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _extract_concept_phrases(
    query: str,
    *,
    max_phrases: int = 8,
    bigram_cap: int = 4,
    trigram_cap: int = 3,
    min_len: int = DEFAULT_PHRASE_MIN_LEN,
) -> list[str]:
    """Pull concept phrases out of a definitional query.

    Strategy: emit up to ``bigram_cap`` bigrams (most semantically
    meaningful for definitional concepts: "الاتفاقية الجماعية",
    "المسؤولية المدنية"), then up to ``trigram_cap`` trigrams (catch
    3-word concepts like "الدفع بعدم الدستورية"), then single content
    tokens (≥ ``min_len`` chars) as a recall fallback. Caps the
    individual categories so a query with many tokens doesn't drown
    trigrams in bigram noise. Phrases are deduped by surface form
    (first-seen wins). Returns up to ``max_phrases`` phrases.
    """
    tokens = _content_tokens(query)
    if not tokens:
        return []

    seen: set[str] = set()
    out: list[str] = []

    def _push(p: str) -> bool:
        """Append ``p`` if new; return True if appended."""
        if p and p not in seen:
            seen.add(p)
            out.append(p)
            return True
        return False

    bigrams = 0
    for i in range(len(tokens) - 1):
        if bigrams >= bigram_cap:
            break
        if _push(f"{tokens[i]} {tokens[i + 1]}"):
            bigrams += 1
        if len(out) >= max_phrases:
            return out

    trigrams = 0
    for i in range(len(tokens) - 2):
        if trigrams >= trigram_cap:
            break
        if _push(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"):
            trigrams += 1
        if len(out) >= max_phrases:
            return out

    for tok in tokens:
        if len(tok) < min_len:
            continue
        _push(tok)
        if len(out) >= max_phrases:
            break
    return out


# ---------------------------------------------------------------------------
# KG entity lookup
# ---------------------------------------------------------------------------


def _kg_phrase_lookup(
    sparql_fn: Callable[[str], list[dict]] | None,
    phrases: list[str],
    *,
    limit: int = DEFAULT_KG_LIMIT,
) -> tuple[dict[str, float], dict[str, str]]:
    """For each phrase, find articles whose text CONTAINS it.

    Returns ``(scores, spans)`` where ``scores[uri]`` is the number of
    distinct phrases matched and ``spans[uri]`` is the first matched
    text seen for that URI (used as the fallback supporting span).
    """
    scores: dict[str, float] = {}
    spans: dict[str, str] = {}
    if not sparql_fn or not phrases:
        return scores, spans
    for phrase in phrases:
        if not phrase:
            continue
        # Escape double-quote / backslash so the SPARQL literal stays well-formed.
        safe = phrase.replace("\\", "").replace('"', "'")
        q = _KG_PHRASE_SEARCH.format(phrase=safe, limit=limit)
        try:
            rows = sparql_fn(q) or []
        except Exception as exc:
            log.debug("kg phrase lookup failed for %r: %s", phrase, exc)
            continue
        seen_for_phrase: set[str] = set()
        for row in rows:
            uri = (row.get("article") or "").strip()
            if not uri or uri in seen_for_phrase:
                continue
            seen_for_phrase.add(uri)
            scores[uri] = scores.get(uri, 0.0) + 1.0
            if uri not in spans:
                spans[uri] = (row.get("text") or "").strip()
    return scores, spans


def _uri_to_doc_ref(
    uri: str,
    registry: ArticleRegistry,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve an article URI to canonical ``(doc_id, article_ref)``.

    Mirrors the parser used by B5; collapses sub-node fragments
    (``art_X_para_1`` → ``art_X``) and runs ``num`` through the
    registry's alias map so a URI's bare law number resolves to the
    canonical ``num_date`` doc_id.
    """
    if not uri:
        return (None, None)
    m = _URI_RE.match(uri)
    if not m:
        return (None, None)
    date = m.group("date")
    num = m.group("num")
    frag = m.group("frag")
    art_eid = _ART_SUFFIX_RE.sub("", frag)
    if not art_eid.startswith("art_"):
        return (None, None)
    article_ref = art_eid[len("art_"):]
    try:
        canonical = (
            registry.resolve_alias(num)
            or registry.resolve_alias(f"{num}_{date}")
            or num
        )
    except Exception:
        canonical = num
    return (canonical, article_ref)


# ---------------------------------------------------------------------------
# Paraphrase fallback
# ---------------------------------------------------------------------------


_PARAPHRASE_PROMPT = """\
You will receive an Arabic legal question that asks for the meaning, definition,
distinction, or scope of a legal concept. Generate {n} short paraphrases of the
question that preserve the legal concept(s) verbatim but vary the
phrasing of the question itself (use synonyms for question words and
restructure if helpful). Each paraphrase must remain a question and
must mention the same concept(s).

Return ONLY a JSON object of the form:
{{"paraphrases": ["...", "...", "..."]}}

Do not add prose, comments, or markdown.

Original question:
{question}
"""


def _generate_paraphrases(
    llm_pool, query: str, n: int, sub_model: str,
) -> list[str]:
    """Single LLM call returning up to ``n`` paraphrases of ``query``."""
    if n <= 0 or not query.strip():
        return []
    prompt = _PARAPHRASE_PROMPT.format(n=n, question=query.strip())
    try:
        raw = llm_pool.call(
            prompt=prompt, model=sub_model, max_tokens=384, temperature=0.0,
        )
    except Exception as exc:
        log.warning("paraphraser LLM call failed: %s", exc)
        return []
    # parse first JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return []
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return []
    items = obj.get("paraphrases") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for s in items:
        if isinstance(s, str) and s.strip() and s.strip() != query.strip():
            out.append(s.strip())
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

VerifierFn = Callable[[Any, str, dict, str], dict]
SummarizerFn = Callable[[Any, str, list[dict], str], dict]
SparqlFn = Callable[[str], list[dict]]
ParaphraseFn = Callable[[Any, str, int, str], list[str]]
AduExtractFn = Callable[[str, Any, str], dict]


class ConceptualDefinitionalHandler:
    """Typed conceptual-definitional handler:

    route → concept-phrase extract → KG entity lookup → paraphrase
    fallback (only on miss) → ADU claim/ground extraction → synthesise.
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
        paraphrase_count: int = DEFAULT_PARAPHRASE_COUNT,
        adu_extract_top_n: int = DEFAULT_ADU_EXTRACT_TOP_N,
        min_kg_hits: int = DEFAULT_MIN_KG_HITS,
        kg_limit: int = DEFAULT_KG_LIMIT,
        kg_boost: float = DEFAULT_KG_BOOST,
        verifier_fn: Optional[VerifierFn] = None,
        summarizer_fn: Optional[SummarizerFn] = None,
        sparql_fn: Optional[SparqlFn] = None,
        paraphrase_fn: Optional[ParaphraseFn] = None,
        adu_extract_fn: Optional[AduExtractFn] = None,
        # E1: union concept->amendment SPARQL hits into the KG-bias set.
        # Catches definitions that live inside amending decrees (the
        # documented R4 ceiling case — e.g. lab_cd_q01 art_114 lives
        # inside 96-21#art_17). Callable: phrases -> set[(doc_id, ref)].
        concept_amendment_fn: Optional[Callable[[List[str]], Any]] = None,
        # Phase D — gap-driven recursion + corrective retry.
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
        self._paraphrase_count = paraphrase_count
        self._adu_extract_top_n = adu_extract_top_n
        self._min_kg_hits = min_kg_hits
        self._kg_limit = kg_limit
        self._kg_boost = kg_boost
        self._verifier_fn = verifier_fn or call_verifier
        self._summarizer_fn = summarizer_fn or call_summarizer
        self._sparql_fn = sparql_fn or self._default_sparql_fn()
        self._paraphrase_fn = paraphrase_fn or _generate_paraphrases
        self._adu_extract_fn = adu_extract_fn or adu_extract
        self._concept_amendment_fn = concept_amendment_fn
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
        kg = self._kg
        if kg is None:
            return None

        def _fn(query: str) -> list[dict]:
            try:
                results = kg.query(query)
            except Exception as exc:
                log.debug("SPARQL failed: %s", exc)
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
                routed=[], phrases=[], kg_hits=0, paraphrases=[], sub_calls=0,
            )

        trajectory: list[dict[str, Any]] = []

        # 1. Doc-route
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)
        trajectory.append({
            "step": "route", "depth": 0,
            "routed_doc_ids": routed_ids,
        })

        # 2. Concept phrases (used for KG entity lookup)
        phrases = _extract_concept_phrases(query)

        sub_calls = 0
        paraphrases: list[str] = []

        # 3. Generate paraphrases (single LLM call). Paraphrase-driven
        # dense retrieval widens recall on definitional questions where
        # the original phrasing might miss — and unlike a "fallback",
        # always running them costs only one extra LLM call but keeps the
        # candidate pool from collapsing when the surface concept is
        # named differently in the article body.
        if self._paraphrase_count > 0:
            try:
                paraphrases = self._paraphrase_fn(
                    self._llm_pool, query, self._paraphrase_count,
                    self._sub_model,
                )
                if paraphrases:
                    sub_calls += 1
            except Exception as exc:
                log.warning("paraphraser failed: %s", exc)
                paraphrases = []

        # 4. RRF-fuse(BM25, Dense-orig, Dense-paraphrase[s]) restricted
        # to routed docs (with full-pool fallback only if filtering
        # would wipe everything — same contract as multi_hop / temporal).
        fused = self._fused_candidates(query, paraphrases, routed_ids)

        # 5. KG entity lookup — surface-form CONTAINS over phrases.
        # The KG is a *secondary* signal: any fused candidate that ALSO
        # has a KG match for at least one concept phrase gets a small
        # additive score boost. This is the B6 KG-Hybrid pattern, which
        # empirically beats KG-only on conceptual_definitional.
        kg_scores, kg_spans = _kg_phrase_lookup(
            self._sparql_fn, phrases, limit=self._kg_limit,
        )
        kg_keys = self._kg_keys(kg_scores)

        # E1 — union concept->amendment SPARQL hits into the bias set so
        # definitions inside amending decrees (lab_cd_q01 pattern) receive
        # the same kg_boost as direct KG hits.
        if self._concept_amendment_fn is not None and phrases:
            try:
                amend_hits = self._concept_amendment_fn(phrases) or set()
                if amend_hits:
                    kg_keys = set(kg_keys) | set(amend_hits)
            except Exception as exc:
                log.debug("E1 concept_amendment_fn raised: %s", exc)

        # 6. Apply KG bias and re-sort.
        candidates = self._apply_kg_bias(fused, kg_keys)

        if not candidates:
            return self._abstain(
                "no_hits",
                routed=routed_ids, phrases=phrases, kg_hits=len(kg_scores),
                paraphrases=paraphrases, sub_calls=sub_calls,
            )

        candidates = candidates[: self._top_k_candidates]
        kg_used = bool(kg_keys)

        # 6. ADU extract on top-N — claim+ground becomes the supporting span.
        adu_outputs: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates):
            if i < self._adu_extract_top_n:
                try:
                    adu = self._adu_extract_fn(
                        cand.get("text", "") or "",
                        self._llm_pool,
                        model=self._sub_model,
                    )
                    sub_calls += 1
                except Exception as exc:
                    log.warning("ADU extract failed: %s", exc)
                    adu = {}
            else:
                adu = {}
            adu_outputs.append(adu)

        # 7. Optional verifier (off by default) + citation assembly
        accumulated: dict[tuple[str, str], dict[str, Any]] = {}
        for cand, adu in zip(candidates, adu_outputs):
            doc_id = cand.get("doc_id", "")
            ref = canonical_article_ref(cand.get("article_ref", "")) or cand.get(
                "article_ref", ""
            )
            confidence = float(cand.get("score", 0.6))
            supporting_quote = ""

            if self._verifier_fn is not None and len(accumulated) < self._verify_top_n:
                try:
                    verdict = self._verifier_fn(
                        self._llm_pool,
                        query,
                        {
                            "doc_id":      doc_id,
                            "article_ref": ref,
                            "text":        cand.get("text", "") or "",
                        },
                        self._sub_model,
                    )
                    sub_calls += 1
                except Exception as exc:
                    log.warning("conceptual verifier failed: %s", exc)
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
                cand=cand,
                adu=adu,
                supporting_quote=supporting_quote,
                confidence=confidence,
                kg_hit=cand.get("kg_hit", False),
            )
            key = (doc_id, ref)
            prior = accumulated.get(key)
            if prior is None or confidence > float(prior.get("confidence", 0.0)):
                accumulated[key] = citation

        if not accumulated:
            return self._abstain(
                "no_verified_articles",
                routed=routed_ids, phrases=phrases, kg_hits=len(kg_scores),
                paraphrases=paraphrases, sub_calls=sub_calls,
            )

        trajectory.append({
            "step": "candidate_pool", "depth": 1,
            "phrases": phrases,
            "paraphrases": paraphrases,
            "kg_hits": len(kg_scores),
            "verified": len(accumulated),
        })

        # Phase D — gap-driven recursion (additive merge).
        recursion_steps = []
        recursion_probe_calls = 0
        if self._enable_recursion and self._recursion_max_depth >= 2:
            sub_calls_holder = [0]

            def _gap_retrieve(gap_q: str) -> dict[tuple[str, str], dict[str, Any]]:
                # Gap question feeds straight back through fused + KG-bias
                # without generating new paraphrases (cost-control: each
                # recursion depth already costs a probe call). Phrases
                # extracted from the gap-question itself drive a fresh
                # KG lookup so adjacent definitional articles surface.
                gap_phrases = _extract_concept_phrases(gap_q)
                gap_fused = self._fused_candidates(gap_q, [], routed_ids)
                gap_kg_scores, _ = _kg_phrase_lookup(
                    self._sparql_fn, gap_phrases, limit=self._kg_limit,
                )
                gap_kg_keys = self._kg_keys(gap_kg_scores)
                if self._concept_amendment_fn is not None and gap_phrases:
                    try:
                        amend_hits = self._concept_amendment_fn(gap_phrases) or set()
                        if amend_hits:
                            gap_kg_keys = set(gap_kg_keys) | set(amend_hits)
                    except Exception:
                        pass
                gap_cands = self._apply_kg_bias(gap_fused, gap_kg_keys)[: self._top_k_candidates]
                local_acc: dict[tuple[str, str], dict[str, Any]] = {}
                for cand in gap_cands:
                    doc_id = cand.get("doc_id", "")
                    ref = canonical_article_ref(cand.get("article_ref", "")) or cand.get(
                        "article_ref", ""
                    )
                    confidence = float(cand.get("score", 0.6))
                    citation = self._build_citation(
                        doc_id=doc_id,
                        article_ref=ref,
                        cand=cand,
                        adu={},
                        supporting_quote="",
                        confidence=confidence,
                        kg_hit=cand.get("kg_hit", False),
                    )
                    local_acc[(doc_id, ref)] = citation
                return local_acc

            retriever = RecursiveRetriever(
                llm_pool=self._llm_pool,
                retrieve_verify_fn=_gap_retrieve,
                max_depth=self._recursion_max_depth,
                coverage_min=self._recursion_coverage_min,
                confidence_weak=self._recursion_confidence_weak,
                confidence_strong=self._recursion_confidence_strong,
                probe_fn=self._recursion_probe_fn,
                probe_model=self._recursion_probe_model,
                seed_accumulator=accumulated,
            )
            accumulated, recursion_steps, recursion_probe_calls = retriever.run(query)
            sub_calls += recursion_probe_calls + sub_calls_holder[0]
            for step in recursion_steps:
                trajectory.append({"step": "recursion", **step.to_dict()})

        ranked = sorted(
            accumulated.values(),
            key=lambda c: float(c.get("confidence", 0.0)),
            reverse=True,
        )
        final_citations = ranked[: self._final_top_k]

        # 8. Synthesise
        template_answer = self._template_answer(final_citations)
        answer_text = template_answer
        try:
            synth = self._summarizer_fn(
                self._llm_pool, query, final_citations, self._sub_model,
            )
            sub_calls += 1
            summary = synth.get("summary")
            if isinstance(summary, str) and summary.strip():
                answer_text = summary.strip()
        except Exception as exc:
            log.warning("conceptual summariser failed: %s — template", exc)
        trajectory.append({"step": "summarize", "depth": 0})

        # 9. Phase D — corrective retry on faithfulness failure.
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
            "retry_count":      1 if (retry_trace and retry_trace.fired) else 0,
            "gate_results":     {},
            "baseline":         TELEMETRY_BASELINE,
            "routed_doc_ids":   routed_ids,
            "concept_phrases":  phrases,
            "kg_hits":          len(kg_scores),
            "kg_used":          kg_used,
            "paraphrases":      paraphrases,
            "sub_call_count":   sub_calls,
            "recursion_trace":  [s.to_dict() for s in recursion_steps],
            "recursion_depth_max": depth_max,
        }
        if retry_trace is not None:
            telemetry["corrective_retry"] = retry_trace.to_dict()

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   phrases,
            "trajectory":        trajectory,
            "tokens_used":       0,
            "depth_max_reached": depth_max,
            "_telemetry":        telemetry,
        }

    # ------------------------------------------------------------------
    # KG bias helpers
    # ------------------------------------------------------------------

    def _kg_keys(self, kg_scores: dict[str, float]) -> set[tuple[str, str]]:
        """Convert KG-search URIs into the canonical ``(doc_id, ref)``
        keys used by the fused candidate pool. Used to flag KG-confirmed
        articles for boosting.
        """
        out: set[tuple[str, str]] = set()
        for uri in kg_scores:
            doc_id, art_ref = _uri_to_doc_ref(uri, self._registry)
            if not doc_id or not art_ref:
                continue
            canon_ref = canonical_article_ref(art_ref) or art_ref
            out.add((doc_id, canon_ref))
        return out

    def _apply_kg_bias(
        self,
        fused: list[dict[str, Any]],
        kg_keys: set[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Add ``kg_boost`` to candidates whose key is KG-confirmed and re-sort.

        Mutates the candidate dicts to mark ``kg_hit`` so downstream
        citations can record provenance.
        """
        if not fused:
            return []
        for c in fused:
            key = (c.get("doc_id", ""), c.get("article_ref", ""))
            if key in kg_keys:
                c["kg_hit"] = True
                c["score"] = float(c.get("score", 0.0)) + self._kg_boost
            else:
                c["kg_hit"] = c.get("kg_hit", False)
        return sorted(
            fused, key=lambda c: float(c.get("score", 0.0)), reverse=True,
        )

    # ------------------------------------------------------------------
    # Hybrid retrieval (with paraphrase widening)
    # ------------------------------------------------------------------

    def _fused_candidates(
        self,
        query: str,
        paraphrases: list[str],
        routed_ids: list[str],
    ) -> list[dict[str, Any]]:
        """RRF-fuse(BM25-on-original, Dense-on-original ∪ paraphrases)."""
        try:
            bm25_hits: list[BM25Hit] = self._bm25.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("conceptual BM25 failed: %s", exc)
            bm25_hits = []
        try:
            dense_hits_orig: list[DenseHit] = self._dense.search(
                query, k=self._k_each
            )
        except Exception as exc:
            log.warning("conceptual dense (original) failed: %s", exc)
            dense_hits_orig = []

        pools: list[list[dict[str, Any]]] = []
        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        if bm25_dicts:
            pools.append(bm25_dicts)
        dense_orig_dicts = self._hits_to_dicts(dense_hits_orig, retriever="dense")
        if dense_orig_dicts:
            pools.append(dense_orig_dicts)

        for p in paraphrases:
            try:
                ph_hits = self._dense.search(p, k=self._k_each)
            except Exception as exc:
                log.warning("paraphrase dense failed for %r: %s", p, exc)
                continue
            ph_dicts = self._hits_to_dicts(ph_hits, retriever="dense_pp")
            if ph_dicts:
                pools.append(ph_dicts)

        if not pools:
            return []
        fused = rrf_fuse(pools)

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
                "kg_hit":      False,
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
        cand: dict[str, Any],
        adu: dict[str, Any],
        supporting_quote: str,
        confidence: float,
        kg_hit: bool,
    ) -> dict[str, Any]:
        text = cand.get("text", "") or ""
        # Pick the supporting span — claim+ground from ADU is the strongest
        # evidence for a definitional answer, but only if the extractor
        # actually returned both. Otherwise fall back to the verifier
        # quote (when substring) or text[:280].
        claim = (adu.get("claim") or "").strip() if isinstance(adu, dict) else ""
        ground = (adu.get("ground") or "").strip() if isinstance(adu, dict) else ""
        adu_span = ""
        if claim and ground:
            adu_span = f"{claim}\n{ground}"
        elif claim:
            adu_span = claim

        if adu_span:
            span = adu_span[:SUPPORT_SPAN_LEN]
        elif supporting_quote and supporting_quote in text:
            span = supporting_quote[:SUPPORT_SPAN_LEN]
        else:
            span = text[:SUPPORT_SPAN_LEN]

        warrant_str = (
            (adu.get("warrant") or "").strip() if isinstance(adu, dict) else ""
        )
        rebuttal_str = (
            (adu.get("rebuttal") or "").strip() if isinstance(adu, dict) else ""
        )
        backing_str = (
            (adu.get("backing") or "").strip() if isinstance(adu, dict) else ""
        )
        argumentation = {
            "claim":    claim,
            "ground":   ground,
            "warrant":  warrant_str,
            "rebuttal": rebuttal_str,
            "backing":  backing_str,
        }
        return {
            "doc_id":            doc_id,
            "article_ref":       article_ref,
            "doc_title":         self._doc_title(doc_id),
            "supporting_span":   span,
            "text":              text,
            "confidence":        float(confidence),
            "kg_hit":            bool(kg_hit),
            # Legacy key — kept for backwards compatibility with the
            # CD-specific test suite. ``argumentation`` is the canonical
            # Phase-C key consumed by am_faithfulness_score / thesis
            # artifacts.
            "adu":               {
                "claim":    claim,
                "ground":   ground,
                "warrant":  warrant_str,
                "rebuttal": rebuttal_str,
            },
            "argumentation":     argumentation,
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
        phrases: list[str],
        kg_hits: int,
        paraphrases: list[str],
        sub_calls: int,
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
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed,
                "concept_phrases": phrases,
                "kg_hits":         kg_hits,
                "kg_used":         False,
                "paraphrases":     paraphrases,
                "sub_call_count":  sub_calls,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_conceptual_definitional_handler(
    kg: Any,
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> ConceptualDefinitionalHandler:
    """Factory mirroring the other handler ``build_*`` helpers."""
    return ConceptualDefinitionalHandler(
        kg=kg,
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
