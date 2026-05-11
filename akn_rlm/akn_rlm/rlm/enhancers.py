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


GEMMA_MODEL = "google/gemma-4-31B"


# ===========================================================================
# E1 — concept_amendment_fn  (re-export from ceiling_breakers for one home)
# ===========================================================================

from akn_rlm.rlm.ceiling_breakers import (   # noqa: E402
    make_concept_amendment_search as make_concept_amendment_fn,
)


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
