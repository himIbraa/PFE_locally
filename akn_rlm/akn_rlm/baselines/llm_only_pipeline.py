"""LLM-only baseline pipeline — no retrieval, no KG, no AM.

This is the **floor** baseline for the thesis: sends the query directly
to a configured LLM and asks it to answer + emit citations in a
structured format. We parse the citations and shape the output dict
identically to the other Phase-1 baselines so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged.

Two variants:

  raw          : send only the query → parse citations from response
  with_context : prepend dense top-5 article texts as context

Citations the LLM emits that don't exist in the registry are kept
(they show up as HCR — that's the point of this baseline).

Telemetry baseline tag examples:
  llm_only_<model>            (raw)
  llm_only_<model>_dense5     (with retrieved context)
  llm_only_<model>_dense5_kg  (with retrieved + KG amendment chain text)
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.normalizers import canonical_article_ref

log = logging.getLogger(__name__)

SUPPORT_SPAN_LEN: int = 280

# ---------------------------------------------------------------------------
# Citation extraction from free-form LLM output
# ---------------------------------------------------------------------------

# Structured citation block we ask the LLM to emit at the end of its answer.
# Example match:  "- doc_id: 84-11, article: 5"
#                 "- doc_id: 84-11_1984-06-09, article: 9 مكرر"
#                 "- doc_id: الدستور 2020-12-30, article: 88"
_STRUCTURED_LINE_RE = re.compile(
    r"^\s*[-•*]?\s*doc[_\s]*id\s*[:：]\s*([^\s,،;؛]+(?:[\s_-][^\s,،;؛]+)*?)\s*[,،;؛]\s*"
    r"article\s*[:：]\s*([^\n,،;؛]+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Loose fallback: "<doc_id_hint>" near "المادة N" within a 120-char window.
_LAW_ID_RE = re.compile(r"(?<!\d)(\d{2,3}-\d{1,4})(?!\d)")
_AR_ARTICLE_RE = re.compile(
    r"(?:المادة|المادتان|المواد)\s+(\d+(?:\s*مكرر(?:\s*\d+)?)?|الأولى|الاولي|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة)",
    re.UNICODE,
)
_FR_ARTICLE_RE = re.compile(r"[Aa]rticle\s+(\d+(?:\s*bis)?)|art\.\s*(\d+(?:\s*bis)?)", re.UNICODE)


def _resolve_doc_hint(hint: str, registry: ArticleRegistry) -> Optional[str]:
    """Try to canonicalise an LLM-emitted doc identifier."""
    if not hint:
        return None
    hint = hint.strip().strip(".,;()[]")
    # Direct alias / numeric-id match
    resolved = registry.resolve_alias(hint)
    if resolved:
        return resolved
    # Try the numeric-id substring
    m = _LAW_ID_RE.search(hint)
    if m:
        resolved = registry.resolve_alias(m.group(1))
        if resolved:
            return resolved
    return None


def _extract_structured_citations(
    text: str, registry: ArticleRegistry,
) -> List[dict]:
    """Pull citations from the structured CITATIONS block we ask for."""
    out: List[dict] = []
    seen: set[tuple[str, str]] = set()
    for m in _STRUCTURED_LINE_RE.finditer(text or ""):
        doc_hint = m.group(1).strip()
        article = m.group(2).strip()
        doc_id = _resolve_doc_hint(doc_hint, registry)
        ref = canonical_article_ref(article) or article
        if not ref:
            continue
        # Keep the citation even if doc_id is unresolved — that's a
        # legitimate "the LLM hallucinated a doc name" signal that
        # should show up in HCR.
        if not doc_id:
            doc_id = doc_hint   # raw hint, will be flagged as not-in-corpus
        key = (doc_id, ref)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "doc_id":          doc_id,
            "article_ref":     ref,
            "doc_title":       _safe_doc_title(registry, doc_id),
            "supporting_span": "",
            "text":            "",
            "confidence":      0.5,
        })
    return out


def _extract_loose_citations(
    text: str, registry: ArticleRegistry,
) -> List[dict]:
    """Loose fallback: find article-number mentions and pair with the
    nearest doc-id mention in a 120-char window. Useful when the LLM
    didn't follow the CITATIONS block format.

    Doc-id mentions come from two sources:
    1. Numeric law-IDs like "84-11"
    2. Arabic alias substrings from the registry's alias map
       ("الدستور", "قانون الأسرة", "القانون المدني", ...)
    """
    if not text:
        return []
    out: List[dict] = []
    seen: set[tuple[str, str]] = set()

    # Find all doc-id positions: numeric law-IDs + alias substrings.
    doc_positions: list[tuple[int, str]] = [
        (m.start(), m.group(1)) for m in _LAW_ID_RE.finditer(text)
    ]
    # Add Arabic / French aliases from the registry. Sort longest-first
    # so multi-word aliases win over substrings (same trick as DocRouter).
    text_lower = text.lower()
    aliases = getattr(registry, "_aliases", {}) or {}
    alias_items = sorted(
        ((k, v) for k, v in aliases.items() if k and v),
        key=lambda kv: (-len(kv[0]), kv[0]),
    )
    # Only consider non-trivial aliases (length >= 4) to avoid false hits
    # like "حق" matching inside every word.
    for alias, doc_id in alias_items:
        if len(alias) < 4:
            continue
        start = 0
        while True:
            idx = text_lower.find(alias, start)
            if idx < 0:
                break
            doc_positions.append((idx, doc_id))
            start = idx + len(alias)
    # Arabic article numbers
    for m in _AR_ARTICLE_RE.finditer(text):
        art = m.group(1).strip()
        ref = canonical_article_ref(art) or art
        if not ref:
            continue
        # Find the nearest doc-id mention (within 120 chars)
        nearest_doc = None
        for pos, raw_id in doc_positions:
            if abs(pos - m.start()) <= 120:
                if nearest_doc is None or abs(pos - m.start()) < abs(nearest_doc[0] - m.start()):
                    nearest_doc = (pos, raw_id)
        if nearest_doc:
            doc_id = registry.resolve_alias(nearest_doc[1]) or nearest_doc[1]
        else:
            doc_id = "UNKNOWN_DOC"
        key = (doc_id, ref)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "doc_id":          doc_id,
            "article_ref":     ref,
            "doc_title":       _safe_doc_title(registry, doc_id),
            "supporting_span": "",
            "text":            "",
            "confidence":      0.3,
        })
    # French articles
    for m in _FR_ARTICLE_RE.finditer(text):
        art = (m.group(1) or m.group(2) or "").strip()
        ref = canonical_article_ref(art) or art
        if not ref:
            continue
        nearest_doc = None
        for pos, raw_id in doc_positions:
            if abs(pos - m.start()) <= 120:
                if nearest_doc is None or abs(pos - m.start()) < abs(nearest_doc[0] - m.start()):
                    nearest_doc = (pos, raw_id)
        if nearest_doc:
            doc_id = registry.resolve_alias(nearest_doc[1]) or nearest_doc[1]
        else:
            doc_id = "UNKNOWN_DOC"
        key = (doc_id, ref)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "doc_id":          doc_id,
            "article_ref":     ref,
            "doc_title":       _safe_doc_title(registry, doc_id),
            "supporting_span": "",
            "text":            "",
            "confidence":      0.3,
        })
    return out


def _safe_doc_title(registry: ArticleRegistry, doc_id: str) -> str:
    try:
        entry = registry.get_doc(doc_id)
        return getattr(entry, "doc_title", "") or doc_id
    except Exception:
        return doc_id


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PROMPT_RAW = """\
أنت خبير قانوني جزائري. أجب على السؤال التالي بناءً على معرفتك بالقوانين الجزائرية.
أنه إجابتك بقائمة المواد القانونية ذات الصلة بالضبط بهذا الشكل:

CITATIONS:
- doc_id: <رمز القانون مثل 84-11 أو 75-58 أو الدستور 2020-12-30>, article: <رقم المادة>
- doc_id: ..., article: ...

إذا لم تكن متأكداً من المواد أو إذا لم يكن للسؤال إجابة في القانون الجزائري، اكتب:
CITATIONS:
(لا توجد مواد ذات صلة)

السؤال: {query}

الإجابة:"""

_PROMPT_WITH_CONTEXT = """\
أنت خبير قانوني جزائري. لديك المواد القانونية التالية كسياق:

{context}

---

استند إلى هذه المواد فقط لتجيب على السؤال. لا تذكر مواد غير موجودة في السياق أعلاه.
أنه إجابتك بقائمة المواد المُستشهد بها بالضبط بهذا الشكل:

CITATIONS:
- doc_id: <doc_id من السياق>, article: <article_ref من السياق>
- ...

إذا لم يحوِ السياق ما يجيب عن السؤال، اكتب:
CITATIONS:
(لا توجد مواد ذات صلة)

السؤال: {query}

الإجابة:"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LLMOnlyBaselinePipeline:
    """Raw-LLM baseline. No retrieval unless ``context_provider`` is set."""

    def __init__(
        self,
        llm_pool,
        registry: ArticleRegistry,
        *,
        model: str = "gpt-oss-120b",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        # If supplied, called as context_provider(query) -> list of
        # {doc_id, article_ref, text} dicts to prepend to the prompt.
        context_provider=None,
        # Telemetry tag suffix so different invocations are distinguishable.
        telemetry_suffix: str = "raw",
    ) -> None:
        self._llm = llm_pool
        self._registry = registry
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_provider = context_provider
        self._telemetry_suffix = telemetry_suffix

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain("empty_query", llm_response="")

        context_items: List[dict] = []
        if self._context_provider is not None:
            try:
                context_items = self._context_provider(query) or []
            except Exception as exc:
                log.warning("context_provider raised: %s", exc)
                context_items = []

        if context_items:
            ctx = "\n\n".join(
                f"[doc_id: {a.get('doc_id', '')}, article: {a.get('article_ref', '')}]\n"
                f"{(a.get('text') or '')[:1500]}"
                for a in context_items
            )
            prompt = _PROMPT_WITH_CONTEXT.format(context=ctx, query=query)
        else:
            prompt = _PROMPT_RAW.format(query=query)

        try:
            llm_response = self._llm.call(
                prompt, model=self._model,
                max_tokens=self._max_tokens, temperature=self._temperature,
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            return self._abstain("llm_error", llm_response=str(exc))

        if not llm_response or not str(llm_response).strip():
            return self._abstain("empty_response", llm_response="")

        # Extract citations: structured block first, loose fallback if empty.
        citations = _extract_structured_citations(llm_response, self._registry)
        if not citations:
            citations = _extract_loose_citations(llm_response, self._registry)

        # Cap to a reasonable number; if zero, we still return an answer
        # but with the abstention flag false so the eval treats the
        # empty-citation case as a "answered without grounding" — that's
        # what raw LLMs do and the metric should reflect it.
        citations = citations[:8]

        return {
            "answer_text":       str(llm_response).strip(),
            "abstention":        False,
            "abstention_reason": None,
            "citations":         citations,
            "reasoning_chain":   [],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count": 0,
                "gate_results": {},
                "baseline":    self._baseline_tag(),
                "model":       self._model,
                "raw_response_preview": str(llm_response)[:600],
            },
        }

    # ------------------------------------------------------------------
    def _baseline_tag(self) -> str:
        return f"llm_only_{self._model.replace('/', '_').replace('-', '_')}_{self._telemetry_suffix}"

    def _abstain(self, reason: str, llm_response: str) -> dict[str, Any]:
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
                "retry_count": 0,
                "gate_results": {},
                "baseline":    self._baseline_tag(),
                "model":       self._model,
                "raw_response_preview": str(llm_response)[:600],
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_llm_only_pipeline(
    llm_pool,
    registry: ArticleRegistry,
    *,
    model: str = "gpt-oss-120b",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    context_provider=None,
    telemetry_suffix: str = "raw",
) -> LLMOnlyBaselinePipeline:
    return LLMOnlyBaselinePipeline(
        llm_pool=llm_pool, registry=registry,
        model=model, max_tokens=max_tokens, temperature=temperature,
        context_provider=context_provider, telemetry_suffix=telemetry_suffix,
    )
