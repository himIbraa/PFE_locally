"""Intent classifier for AlgerianLegalBench query types.

Maps a raw query string to an IntentResult covering all 8 benchmark types:
  rule_application | exact_article | multi_hop | unanswerable |
  layman | long_context | conceptual_definitional | temporal_factual

Classification strategy:
  1. Answerability pre-check — if foreign-law signals detected in the query
     itself, label answerability_hint = "probably_unanswerable".
  2. Regex routing — deterministic keyword patterns covering 7 of the 8 types.
     Default (no match) → rule_application.
  3. Long-context heuristic — queries mentioning 3+ legal entities or spanning
     multiple procedural stages are promoted to long_context.

For Phase B we also expose an LLM-backed classifier (``llm_classify``)
that uses Gemma-4-31B with few-shot examples. The regex classifier only
hit 29.92% on full 244 because it has no rule for ``unanswerable`` and
defaults to ``rule_application`` for ~91% of queries; the LLM path is
the production deployable variant when ``--no-gold-type`` is set.

IntentResult is a frozen dataclass so it is safe to store in pipeline state.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from akn_rlm.normalizers import normalize_query

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns (ordered — first match wins; checked in order below)
# ---------------------------------------------------------------------------

_TEMPORAL_FACTUAL_KW = re.compile(
    r"(?:منذ متى|تاريخ صدور|سنة.*إصدار|"
    r"متى.*صدر|متى.*نشر|متى.*أُصدر|متى.*صار نافذ|"
    r"when.*promulgated|when.*published|date.*enacted|year.*issued)",
    re.IGNORECASE | re.UNICODE,
)

_MULTI_HOP_KW = re.compile(
    r"(?:و(?:القانون|الق[اأ]نون)|both.*and|"
    r"العلاقة بين|interaction between|combined|"
    r"متعدد القوانين|multi.?law|"
    r"بين.*وبين|كلا القانونين)",
    re.IGNORECASE | re.UNICODE,
)

_EXACT_ARTICLE_KW = re.compile(
    r"(?:(?:الم[اآ]دة|art(?:icle)?\.?)\s*\d+|"
    r"نص الم[اآ]دة\s*\d+|"
    r"\bالم[اآ]دة\s+(?:الأولى|الثانية|الثالثة|\d+)\b)",
    re.IGNORECASE | re.UNICODE,
)

_CONCEPTUAL_KW = re.compile(
    r"(?:ما هو تعريف|عرّف|ما المقصود|ماذا يُقصد|"
    r"define|definition of|what is meant by|"
    r"ما مفهوم|مفهوم|ما معنى)",
    re.IGNORECASE | re.UNICODE,
)

_LAYMAN_KW = re.compile(
    r"(?:بكلمات بسيطة|بشكل مبسط|للمواطن العادي|"
    r"in simple terms|simply|للعموم|باختصار غير قانوني|"
    r"اشرح ببساطة|بلغة بسيطة)",
    re.IGNORECASE | re.UNICODE,
)

# Long-context: procedural / multi-stage questions
_LONG_CONTEXT_KW = re.compile(
    r"(?:خطوات|إجراءات.*تفصيلية|كيفية.*كاملة|"
    r"step.?by.?step|full procedure|detailed process|"
    r"شرح.*مراحل|مراحل.*تفصيلية|الإجراءات الكاملة)",
    re.IGNORECASE | re.UNICODE,
)

# Heuristic: how many distinct legal entities appear (law numbers, article refs)
_LEGAL_ENTITY_RE = re.compile(
    r"(?:\d{2,4}-\d{1,3}|\bالمادة\s+\d+|art(?:icle)?\s*\d+)",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# IntentResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentResult:
    """Output of the intent classifier."""
    query_type: str           # one of the 8 benchmark query types
    language: str             # "ar" | "fr"
    answerability_hint: str   # "probably_answerable" | "probably_unanswerable"
    normalized_query: str     # Arabic/French normalized form of the query
    confidence: float         # 1.0 for regex matches; 0.8 for heuristic; 0.6 default


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify(query: str) -> IntentResult:
    """Classify a raw query string into an IntentResult."""
    normalized, lang = normalize_query(query)

    # Answerability pre-check via jurisdiction infection signals
    from akn_rlm.gates.jurisdiction import is_infected
    answerability = (
        "probably_unanswerable" if is_infected(query)
        else "probably_answerable"
    )

    query_type, confidence = _classify_type(query)

    return IntentResult(
        query_type=query_type,
        language=lang,
        answerability_hint=answerability,
        normalized_query=normalized,
        confidence=confidence,
    )


def _classify_type(query: str) -> tuple[str, float]:
    """Return (query_type, confidence) from regex rules."""
    # Temporal factual (must precede generic temporal patterns)
    if _TEMPORAL_FACTUAL_KW.search(query):
        return "temporal_factual", 1.0

    # Multi-hop (cross-law interactions)
    if _MULTI_HOP_KW.search(query):
        return "multi_hop", 1.0

    # Exact article lookup
    if _EXACT_ARTICLE_KW.search(query):
        return "exact_article", 1.0

    # Conceptual / definitional
    if _CONCEPTUAL_KW.search(query):
        return "conceptual_definitional", 1.0

    # Layman / plain-language
    if _LAYMAN_KW.search(query):
        return "layman", 1.0

    # Long-context (procedural keyword OR 3+ distinct legal entity references)
    if _LONG_CONTEXT_KW.search(query):
        return "long_context", 1.0
    if len(_LEGAL_ENTITY_RE.findall(query)) >= 3:
        return "long_context", 0.8

    # Default
    return "rule_application", 0.6


# ---------------------------------------------------------------------------
# LLM-backed classifier (Phase B — production deployable path)
# ---------------------------------------------------------------------------

#: Valid query_type labels in canonical order — used for prompt rendering
#: and output parsing.
VALID_QUERY_TYPES: tuple[str, ...] = (
    "rule_application",
    "exact_article",
    "multi_hop",
    "unanswerable",
    "layman",
    "long_context",
    "conceptual_definitional",
    "temporal_factual",
)

#: Default Gemma model routed via LLMPool's "gemma" / "google" slot.
DEFAULT_LLM_CLASSIFIER_MODEL: str = "google/gemma-4-31B"
DEFAULT_LLM_CLASSIFIER_MAX_TOKENS: int = 32

# Few-shot examples ground each label in a concrete query so Gemma can
# distinguish the close pairs (TF vs CD, RA vs MH, RA vs layman, EA vs
# RA). Kept short so the prompt fits comfortably under 1k tokens.
_LLM_CLASSIFIER_PROMPT = """\
أنت مصنّف نوع الأسئلة في معيار AlgerianLegalBench. \
صنّف السؤال التالي إلى نوع واحد فقط من الأنواع الثمانية المعرّفة أدناه. \
أعد الإجابة كاسم التصنيف فقط بالإنجليزية (snake_case)، بدون أي شرح إضافي.

الأنواع الثمانية:

1. rule_application — يطلب شرح قاعدة أو مبدأ قانوني عام أو تطبيقه على حالة. مثال:
   - "بيّن الأساس القانوني لمبدأ خضوع القاضي للنص التشريعي"
   - "اشرح مبدأ عدم رجعية القوانين من حيث سريانها الزمني"

2. exact_article — يستفسر عن محتوى مادة محددة أو حكم منصوص عليه نصاً، \
حتى ولو لم يذكر رقم المادة صراحة. مثال:
   - "نص المادة 408 من القانون المدني"
   - "حدّد سن الأهلية القانونية للزواج"
   - "اشرح المبدأ الدستوري للشرعية الجنائية"

3. multi_hop — يحتاج للجمع بين قانونين أو نصين متعددين (تفاعل / تكامل / علاقة). مثال:
   - "كيف تتفاعل أحكام المادة 408 من القانون المدني مع قانون الأسرة؟"
   - "ما العلاقة بين قانون العقوبات وقانون الإجراءات الجزائية؟"

4. unanswerable — يسأل عن نظام قانوني أجنبي أو مفهوم غير موجود في القانون الجزائري، \
ويُذكر فيه غالباً اسم أجنبي بين قوسين (at-will, ISF, DIC, …). مثال:
   - "هل تفرض الجزائر ضريبة التضامن على الثروة (ISF) كالنظام الفرنسي؟"
   - "هل يحق لصاحب العمل في الجزائر فصل العامل دون إبداء أسباب (at-will employment)؟"

5. layman — مكتوب بالدارجة الجزائرية أو بلغة عامية (مرتي، نقدر، وقتاش، …). مثال:
   - "أنا طلقت مرتي في المحكمة، هل نقدر نرجعها بلا ما نديرو عقد جديد؟"
   - "واش هي العقوبة اللي يستناها واحد سرقلي تليفوني؟"

6. long_context — يطلب عرضاً شاملاً تفصيلياً (جميع، تفصيلياً، قارن بين كل…). مثال:
   - "اشرح تفصيلياً جميع أحكام تكوين عقد البيع في القانون المدني"
   - "اعرض تفصيلياً جميع أنواع الشركات التجارية"

7. conceptual_definitional — يسأل عن تعريف أو فرق بين مفهومين أو مدلول مصطلح. مثال:
   - "ما الفرق بين الاتفاقية الجماعية واتفاقية المؤسسة؟"
   - "ما مفهوم الدفع بعدم الدستورية؟"

8. temporal_factual — يسأل عن تطور قاعدة عبر الزمن أو يذكر فترات / سنوات محددة \
(كيف تطور، بين سنة وسنة، 1996/2008/2020…). مثال:
   - "كيف تطورت قاعدة 51/49 بين 2016 و2022؟"
   - "كيف تطور نظام علاقات العمل بين التسيير الاشتراكي وقانون 90-11؟"

السؤال:
{query}

نوع السؤال (اسم واحد فقط):"""


def _parse_llm_label(raw: str) -> Optional[str]:
    """Pull the first valid label out of an LLM completion.

    Accepts noise around the label (quotes, trailing punctuation, label
    prefixes like "Type:"). Returns None if no valid label is matched.
    """
    if not raw:
        return None
    text = raw.strip().lower()
    # Strip common chat-completion artifacts (Gemma sometimes prepends a
    # label or wraps in quotes/asterisks).
    for prefix in ("type:", "label:", "answer:", "نوع السؤال:", "نوع:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = text.strip(" \t\n\r'\"`*.,;:()[]{}<>")
    # First, exact match on full label
    for label in VALID_QUERY_TYPES:
        if text == label:
            return label
    # Substring match — Gemma sometimes writes "the type is multi_hop"
    matches = [l for l in VALID_QUERY_TYPES if l in text]
    if len(matches) == 1:
        return matches[0]
    if matches:
        # Prefer the longest match so e.g. "conceptual_definitional"
        # beats nothing else, and "exact_article" beats "exact".
        return max(matches, key=len)
    return None


def llm_classify(
    query: str,
    llm_pool,
    *,
    model: str = DEFAULT_LLM_CLASSIFIER_MODEL,
    max_tokens: int = DEFAULT_LLM_CLASSIFIER_MAX_TOKENS,
    fallback_to_regex: bool = True,
) -> IntentResult:
    """LLM-backed Phase-B classifier.

    Calls ``llm_pool`` with a few-shot Gemma prompt that lists the 8
    AlgerianLegalBench query types with representative examples, then
    parses one label out of the response. On any failure (LLM error,
    unparseable output) the function falls back to the regex
    ``classify`` so we never return a blank ``query_type``.

    Parameters
    ----------
    query : str
        Raw query string (Arabic / Darja / French).
    llm_pool : LLMPool-like
        Object with a ``.call(prompt, model=..., max_tokens=...,
        temperature=...)`` method that returns a string.
    model : str
        Model name routed by the pool. Defaults to the Gemma slot.
    max_tokens : int
        Cap on the completion. The classifier only needs a single label.
    fallback_to_regex : bool
        If True (default), parse failures fall back to ``classify()``.
        If False, parse failures return ``rule_application`` directly.

    Returns
    -------
    IntentResult
        Mirrors ``classify()``: ``query_type`` is one of the 8 valid
        labels; ``confidence`` is 0.95 for a clean LLM match, 0.6 for
        the regex fallback, 0.0 for the hard default.
    """
    if not isinstance(query, str) or not query.strip():
        # Empty queries can't be classified — return the safe default
        # with confidence 0 so callers can see this branch was hit.
        normalized, lang = normalize_query(query or "")
        return IntentResult(
            query_type="rule_application",
            language=lang,
            answerability_hint="probably_answerable",
            normalized_query=normalized,
            confidence=0.0,
        )

    normalized, lang = normalize_query(query)

    # Run answerability hint via the regex jurisdiction gate; this stays
    # useful as a side-signal even when the LLM picks the query_type.
    from akn_rlm.gates.jurisdiction import is_infected
    answerability = (
        "probably_unanswerable" if is_infected(query)
        else "probably_answerable"
    )

    prompt = _LLM_CLASSIFIER_PROMPT.format(query=query.strip())
    raw: str = ""
    try:
        raw = llm_pool.call(
            prompt, model=model, max_tokens=max_tokens, temperature=0.0,
        )
    except Exception as exc:
        log.warning(
            "llm_classify: LLM call failed (%s) — falling back to regex", exc,
        )

    label = _parse_llm_label(raw) if raw else None
    if label is not None:
        return IntentResult(
            query_type=label,
            language=lang,
            answerability_hint=answerability,
            normalized_query=normalized,
            confidence=0.95,
        )

    # LLM produced no valid label
    if fallback_to_regex:
        log.info(
            "llm_classify: unparseable LLM output (%r) — using regex fallback",
            (raw or "")[:80],
        )
        return classify(query)

    return IntentResult(
        query_type="rule_application",
        language=lang,
        answerability_hint=answerability,
        normalized_query=normalized,
        confidence=0.0,
    )


def make_llm_classifier_fn(
    llm_pool,
    *,
    model: str = DEFAULT_LLM_CLASSIFIER_MODEL,
    max_tokens: int = DEFAULT_LLM_CLASSIFIER_MAX_TOKENS,
):
    """Return a ``query -> query_type`` closure for the dispatcher.

    The dispatcher's ``_classifier_fn`` expects a ``Callable[[str], str]``
    (i.e. returns just the label string, not an IntentResult). This
    helper wraps ``llm_classify`` to match that contract.
    """

    def _fn(query: str) -> str:
        return llm_classify(
            query, llm_pool, model=model, max_tokens=max_tokens,
        ).query_type

    return _fn
