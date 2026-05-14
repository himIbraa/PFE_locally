"""Single-query inference endpoint for the AKN-RLM UI.

Exposes :func:`answer_query` — the function the UI calls with a raw
Arabic / French / Darja query and gets back a structured answer:

  * ``answer_text``: Arabic synthesised answer.
  * ``citations``: list of :class:`Citation` dataclasses, each with
    canonical doc_id / article_ref, the Toulmin argumentation
    (claim / ground / warrant / rebuttal / backing when extracted by
    Phase C pervasive ADU), the in-force version date for temporal
    queries, and the confidence the verifier emitted.
  * ``trajectory``: list of :class:`TrajectoryStep` dataclasses — one
    per pipeline step (route → decompose / KG chain → recursion →
    summarise → faithfulness gate). Each step carries a
    human-readable summary plus the raw detail dict for the UI to
    expand on demand.
  * ``references``: pre-formatted markdown list ready to render under
    the answer ("[1] قانون الأسرة، المادة 54 (نسخة 2005-02-27)").
  * ``abstained``, ``abstention_reason``: when the pipeline could not
    answer (typically the ``unanswerable`` handler).
  * ``latency_s``, ``sub_call_count``: bookkeeping.

The dispatcher is **lazy-initialised** on first call: parsing the
corpus, loading the BM25 / dense indices, connecting to the LLM pool,
building the doc-router, and lazy-loading the KG on first KG-using
dispatch. Subsequent calls reuse the same singleton — fast.

The deployable Phase E pipeline used here is:

  * classifier-typed dispatch via Gemma-4-31B ``llm_classify``,
  * selective E4 HyDE (Qwen3 drafts a hypothetical answer, dense
    embeds query + answer; OFF for TF / CD which the KG handles),
  * pervasive Toulmin ADU on every cited article,
  * gap-driven recursion in RA / MH / TF / CD with
    ``recursion_coverage_min=4`` for MH / RA (Phase E.1 tuning),
  * corrective retry on faithfulness-gate failure (regenerates the
    answer once with "use only cited articles" feedback).

The KG-CONTAINS channels E.2 (``--e5``) / E.3 (``--e6``) / E.4
(``--e7``) added by Phase E are NOT enabled here — they ship behind
feature flags for thesis ablations but were measured to add no
Cite F1 lift and (E.4) explode latency 440×.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A single citation in the answer.

    Attributes mirror the dispatched-citation dict shape so the UI can
    deep-link any field. ``argumentation`` is the Toulmin block when
    Phase C pervasive ADU extracted one for this citation, ``None``
    otherwise (e.g. unanswerable handler emits no citations and
    therefore no argumentation).
    """
    doc_id: str
    article_ref: str
    doc_title: str
    supporting_span: str
    text: str
    confidence: float
    version_date: Optional[str] = None
    kg_source: Optional[str] = None  # "kg" / "fallback" for TF citations
    argumentation: Optional[dict[str, Any]] = None
    verifier_relevant: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    """One human-readable step in the pipeline trajectory.

    ``summary`` is a short Arabic/English sentence the UI can show
    directly; ``detail`` carries the raw step payload for "show more"
    expansions. ``depth`` mirrors the underlying recursion depth
    marker (0 = handler-internal step, 1 = depth-1 retrieve+verify,
    2/3 = recursion passes).
    """
    step: str
    depth: int
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResponse:
    """The full response shape the UI consumes."""
    query: str
    query_type_predicted: str
    handler_used: str
    answer_text: str
    citations: list[Citation]
    references: list[str]
    trajectory: list[TrajectoryStep]
    abstained: bool
    abstention_reason: Optional[str]
    latency_s: float
    sub_call_count: int
    am_faithfulness_score: Optional[float] = None
    recursion_depth_max: int = 1
    corrective_retry_fired: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() if hasattr(c, "to_dict") else c
                          for c in self.citations]
        d["trajectory"] = [s.to_dict() if hasattr(s, "to_dict") else s
                           for s in self.trajectory]
        return d


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------

# Thread-safe lazy init. The UI process imports this module once;
# the first ``answer_query`` call pays the corpus-parse + index-load
# + LLM-pool-connect cost (~5-10 s on a warm laptop), subsequent
# calls just dispatch.
_DISPATCHER: Any = None
_LLM_POOL: Any = None
_INIT_LOCK = threading.Lock()


def reset_dispatcher() -> None:
    """Drop the cached dispatcher + LLM pool. Tests use this to inject
    mocks; production code rarely needs it (a re-import of the module
    has the same effect)."""
    global _DISPATCHER, _LLM_POOL
    with _INIT_LOCK:
        _DISPATCHER = None
        _LLM_POOL = None


def get_dispatcher() -> Any:
    """Return the lazily-built deployable dispatcher singleton."""
    global _DISPATCHER, _LLM_POOL
    if _DISPATCHER is not None:
        return _DISPATCHER
    with _INIT_LOCK:
        if _DISPATCHER is not None:
            return _DISPATCHER
        _DISPATCHER = _build_dispatcher()
        return _DISPATCHER


def _build_dispatcher() -> Any:
    """One-shot construction of the deployable Phase E pipeline.

    Mirrors the flag set of ``rlm_dispatched_full_phase_e_final`` —
    the run that produced Cite F1 = 0.3045 on the full 244-question
    benchmark (HANDOFF §1.4f Task #6).
    """
    # Local imports keep the API module cheap to merely *import* —
    # the heavy corpus/index/LLM modules only load on the first
    # ``answer_query`` call, not at ``from akn_rlm.api import ...``.
    from akn_rlm.config import (
        BM25_INDEX_PATH,
        DENSE_FAISS_PATH,
        DENSE_META_PATH,
    )
    from akn_rlm.corpus.akn_parser import parse_all
    from akn_rlm.corpus.article_registry import ArticleRegistry
    from akn_rlm.indexers.bm25 import BM25Index
    from akn_rlm.indexers.dense import DenseIndex
    from akn_rlm.llm.client import LLMPool
    from akn_rlm.rlm.classifier import (
        DEFAULT_LLM_CLASSIFIER_MODEL,
        make_llm_classifier_fn,
    )
    from akn_rlm.rlm.dispatcher import build_dispatcher
    from akn_rlm.rlm.routing import build_doc_router

    global _LLM_POOL

    log.info("[api] Building deployable AKN-RLM dispatcher (Phase E SOTA) …")
    t0 = time.time()

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {BM25_INDEX_PATH} — run "
            "`scripts/build_indices.py` first."
        )
    if not DENSE_FAISS_PATH.exists() or not DENSE_META_PATH.exists():
        raise FileNotFoundError(
            "Dense index missing — run `scripts/build_indices.py` first."
        )

    registry = ArticleRegistry()
    registry.build(parse_all())

    bm25 = BM25Index.load(BM25_INDEX_PATH)
    dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)

    _LLM_POOL = LLMPool.default()

    classifier_fn = make_llm_classifier_fn(
        _LLM_POOL, model=DEFAULT_LLM_CLASSIFIER_MODEL,
    )

    router = build_doc_router(registry=registry, bm25=bm25)

    # Phase E.4 HyDE wraps the dense index inside the dispatcher when
    # AKN_E4_HYDE=1. Set the env var so the dispatcher's enhancers
    # module picks it up. Mirrors the `--e4` CLI flag.
    import os as _os
    _os.environ.setdefault("AKN_E4_HYDE", "1")

    def _load_kg() -> Any:
        from akn_rlm.corpus.kg_loader import load_kg
        log.info("[api] Lazy-loading KG (~26 s on first KG dispatch) …")
        return load_kg()

    dispatcher = build_dispatcher(
        bm25=bm25, dense=dense, registry=registry,
        llm_pool=_LLM_POOL, router=router,
        kg=None, kg_loader=_load_kg,
        classifier_fn=classifier_fn,
        # Phase C — pervasive Toulmin ADU on every cited article.
        enable_pervasive_adu=True,
        adu_extract_top_n=5,
        # Phase D — gap-driven recursion + corrective retry.
        enable_recursion=True,
        recursion_max_depth=3,
        enable_corrective_retry=True,
        # Phase E.1 — per-handler coverage_min override (MH/RA only).
        # TF/CD keep DEFAULT_COVERAGE_MIN=2 so the gap-probe still
        # fires on their thin pools.
        recursion_coverage_min_overrides={
            "multi_hop":        4,
            "rule_application": 4,
        },
        # E5 / E6 / E7 are NOT enabled — they ship behind feature
        # flags but were measured to add no Cite F1 lift and (E.7)
        # explode router latency 440×. See HANDOFF §1.4f.
    )

    log.info("[api] Dispatcher built in %.2f s", time.time() - t0)
    return dispatcher


# ---------------------------------------------------------------------------
# Trajectory summarisation
# ---------------------------------------------------------------------------

_STEP_LABELS_AR = {
    "route":             "توجيه الوثائق",
    "extract_date":      "استخراج التاريخ المرجعي",
    "decompose":         "تجزئة السؤال",
    "decompose_sweep":   "تجزئة السؤال والاسترجاع لكل جزء",
    "kg_chain":          "استرجاع سلسلة التعديلات من الرسم البياني المعرفي",
    "recursion":         "الاسترجاع التكراري",
    "adu_extract":       "استخراج بنية الحجج (Toulmin)",
    "summarize":         "تركيب الإجابة",
    "faithfulness_gate": "بوابة المطابقة الدلالية",
}

_HANDLER_LABELS_AR = {
    "rule_application":         "تطبيق قاعدة",
    "exact_article":            "مادة محددة",
    "multi_hop":                "متعدد القفزات",
    "temporal_factual":         "زمني-وقائعي",
    "conceptual_definitional":  "تعريف مفهوم",
    "unanswerable":             "غير قابل للإجابة",
    "layman":                   "صياغة عامية",
    "long_context":             "سياق طويل",
}

_GAP_DECISION_LABELS_AR = {
    "depth_1":                "الاسترجاع الأولي",
    "strong_skip":            "ثقة عالية - عدم الحاجة للتكرار",
    "thin_force_yes":         "حوض مرشحين رقيق - تكرار قسري",
    "thin_force_no":          "حوض رقيق لكن لا توجد فجوة قابلة للاسترجاع",
    "weak_probe_yes":         "ثقة متوسطة - الفحص يطلب التكرار",
    "weak_probe_no":          "ثقة متوسطة - الفحص يؤكد الكفاية",
    "covered":                "التغطية كاملة - توقف",
    "max_depth":              "الحد الأقصى للعمق",
}


def _summarise_step(raw: dict[str, Any]) -> TrajectoryStep:
    """Translate a raw trajectory step dict into a human-readable
    :class:`TrajectoryStep`. ``summary`` is a single Arabic sentence;
    ``detail`` keeps the raw payload for UI expansion."""
    step = str(raw.get("step", "?"))
    depth = int(raw.get("depth", 0))
    label_ar = _STEP_LABELS_AR.get(step, step)

    summary = label_ar
    if step == "route":
        ids = raw.get("routed_doc_ids") or []
        summary = f"{label_ar}: {', '.join(ids) if ids else '—'}"
    elif step == "extract_date":
        dates = raw.get("dates") or []
        target = raw.get("target") or "—"
        summary = (f"{label_ar}: تواريخ مستخرجة = {dates}، "
                   f"التاريخ المستهدف = {target}")
    elif step == "decompose_sweep":
        n_sq = raw.get("sub_questions", 0)
        verified = raw.get("verified", 0)
        summary = (f"{label_ar}: {n_sq} أسئلة فرعية، "
                   f"{verified} مرشحاً تم التحقق منه")
    elif step == "kg_chain":
        n = raw.get("candidates", 0)
        v = raw.get("verified", 0)
        summary = f"{label_ar}: {n} مرشحاً، {v} نتيجة من سلسلة التعديلات"
    elif step == "recursion":
        decision = raw.get("gap_decision") or "—"
        decision_ar = _GAP_DECISION_LABELS_AR.get(decision, decision)
        new_cits = raw.get("new_citations", 0)
        sub_q = (raw.get("sub_question") or "")
        if sub_q and len(sub_q) > 80:
            sub_q = sub_q[:77] + "…"
        summary = (f"{label_ar} (عمق {raw.get('depth', depth)}): "
                   f"{decision_ar}، +{new_cits} استشهاد")
        if sub_q:
            summary += f' — "{sub_q}"'
    elif step == "adu_extract":
        n = raw.get("extracts", 0)
        summary = f"{label_ar}: تم تطبيقه على {n} استشهاد"
    elif step == "summarize":
        summary = label_ar
    elif step == "faithfulness_gate":
        fired = raw.get("fired", False)
        pre = raw.get("pre_passed", None)
        post = raw.get("post_passed", None)
        if not fired:
            summary = f"{label_ar}: نجح من البداية"
        else:
            pre_s = "نجح" if pre else "فشل"
            post_s = "نجح" if post else "فشل"
            summary = (f"{label_ar}: قبل = {pre_s}، "
                       f"بعد إعادة التركيب = {post_s}")

    detail = {k: v for k, v in raw.items() if k not in {"step", "depth"}}
    return TrajectoryStep(step=step, depth=depth,
                          summary=summary, detail=detail)


# ---------------------------------------------------------------------------
# Citation + reference formatting
# ---------------------------------------------------------------------------


def _to_citation(raw: dict[str, Any]) -> Citation:
    """Convert a dispatched citation dict into a :class:`Citation`."""
    arg = raw.get("argumentation")
    if isinstance(arg, dict) and not any(arg.values()):
        # An empty / all-null Toulmin block; treat as missing so the
        # UI doesn't show empty rows.
        arg = None
    return Citation(
        doc_id=str(raw.get("doc_id", "")),
        article_ref=str(raw.get("article_ref", "")),
        doc_title=str(raw.get("doc_title", "") or raw.get("doc_id", "")),
        supporting_span=str(raw.get("supporting_span", "") or ""),
        text=str(raw.get("text", "") or ""),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        version_date=(str(raw["version_date"])
                      if raw.get("version_date") else None),
        kg_source=(str(raw["kg_source"]) if raw.get("kg_source") else None),
        argumentation=arg,
        verifier_relevant=(bool(raw["verifier_relevant"])
                           if "verifier_relevant" in raw else None),
    )


def _format_references(citations: list[Citation]) -> list[str]:
    """Format a markdown-ready reference list. One entry per citation,
    keyed by 1-based index. Suitable to drop under the answer."""
    out: list[str] = []
    for i, c in enumerate(citations, start=1):
        doc = c.doc_title or c.doc_id
        ref = f"[{i}] {doc}، المادة {c.article_ref}"
        if c.version_date:
            ref += f" (نسخة {c.version_date})"
        out.append(ref)
    return out


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def answer_query(
    query: str,
    *,
    dispatcher: Optional[Any] = None,
) -> AnswerResponse:
    """Answer a single Arabic / French / Darja legal query.

    The function:

      1. Lazy-builds the deployable Phase E pipeline on first call
         (subsequent calls reuse the singleton).
      2. Runs the classifier-typed dispatcher, which routes the query
         to one of the 8 typed handlers, executes retrieval +
         verification + recursion + summarisation + corrective retry,
         and returns a baseline-shaped answer dict.
      3. Wraps the answer in :class:`AnswerResponse` with
         human-readable trajectory steps, structured citations,
         and a pre-formatted reference list.

    Args:
      query: The user's natural-language query. Non-empty string.
      dispatcher: Optional pre-built dispatcher (mainly for tests).
        When ``None``, the lazy singleton is used.

    Returns:
      :class:`AnswerResponse` ready for JSON serialisation.

    Raises:
      ValueError: empty / whitespace-only query.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("answer_query: 'query' must be a non-empty string")

    disp = dispatcher if dispatcher is not None else get_dispatcher()
    t0 = time.time()

    raw = disp.run(query.strip())
    latency = time.time() - t0

    telemetry = raw.get("_telemetry", {}) or {}
    citations_raw = raw.get("citations", []) or []
    trajectory_raw = raw.get("trajectory", []) or []

    citations = [_to_citation(c) for c in citations_raw]
    trajectory = [_summarise_step(s) for s in trajectory_raw]
    # Fallback for handlers that don't emit a structured trajectory
    # (exact_article / unanswerable / layman). Give the UI at least
    # one row so the "explainability" panel is never empty.
    if not trajectory:
        handler = telemetry.get("dispatched_handler", "?")
        handler_ar = _HANDLER_LABELS_AR.get(handler, handler)
        n_cits = len(citations)
        fallback_summary = (
            f"المعالج: {handler_ar} — تم إصدار {n_cits} استشهاد"
        )
        trajectory.append(TrajectoryStep(
            step="handler_summary",
            depth=0,
            summary=fallback_summary,
            detail={
                "handler":        handler,
                "n_citations":    n_cits,
                "sub_call_count": telemetry.get("sub_call_count", 0),
            },
        ))
    references = _format_references(citations)

    return AnswerResponse(
        query=query.strip(),
        query_type_predicted=str(telemetry.get("dispatched_query_type", "?")),
        handler_used=str(telemetry.get("dispatched_handler", "?")),
        answer_text=str(raw.get("answer_text", "") or ""),
        citations=citations,
        references=references,
        trajectory=trajectory,
        abstained=bool(raw.get("abstention", False)),
        abstention_reason=(str(raw["abstention_reason"])
                           if raw.get("abstention_reason") else None),
        latency_s=latency,
        sub_call_count=int(telemetry.get("sub_call_count", 0) or 0),
        am_faithfulness_score=None,
        recursion_depth_max=int(telemetry.get(
            "recursion_depth_max", raw.get("depth_max_reached", 1)
        ) or 1),
        corrective_retry_fired=bool(
            (telemetry.get("corrective_retry") or {}).get("fired", False)
        ),
    )
