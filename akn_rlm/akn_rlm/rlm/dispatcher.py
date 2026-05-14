"""RLM dispatcher — Phase 2 / R7.

Routes a query to the right Phase-2 typed handler based on its
``query_type``. The Phase-2 handlers (R2-R6) are self-contained
baseline-shaped pipelines that intentionally bypass the freeform-Python
``RootController``; this dispatcher is the production seam that wires
them together so the benchmark runner can dispatch every question to
its native handler in a single sweep.

Pipeline:

  1. If ``query_type`` is supplied (the benchmark always tags every
     question), use it directly. Otherwise fall back to
     ``akn_rlm.rlm.classifier.classify`` to pick.

  2. Map the type to a handler key (the eight ALB v3.0 types are
     1-to-1; the legacy ``temporal`` alias coalesces to
     ``temporal_factual``; unknown types fall back to
     ``rule_application`` — the deterministic-template default).

  3. **Lazy-build** the chosen handler. KG-loading handlers
     (``temporal_factual``, ``conceptual_definitional``) only parse the
     ~26-second TTL when actually dispatched. This keeps the gate cheap
     for slices that don't touch the KG.

  4. Forward ``handler.run(query)``. The answer is shaped exactly like
     the deterministic baselines so
     :func:`akn_rlm.eval.runner._answer_to_result` consumes it
     unchanged.

  5. Patch telemetry: keep the inner handler's ``baseline`` tag (so
     ``compare_baselines.py`` can still pick out per-handler runs from
     the predictions if desired) and add ``dispatched_handler`` /
     ``dispatched_query_type`` / ``dispatch_baseline = "rlm_dispatched"``
     so the dispatched run gets its own column in the comparison
     table.

The long_context handler gets a **per-summariser timeout** wrapper
(default 60 s) before being constructed: HANDOFF §R6.4 documents one
``com_lc_q01`` query that hung Qwen3-30B-A3B-Thinking for ~5 hours on a
10-article prompt, which would let one query hostage an entire
benchmark sweep. On timeout the wrapper raises ``TimeoutError`` so the
existing ``LongContextHandler`` fallback (deterministic Arabic
template) takes over.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Index
from akn_rlm.indexers.dense import DenseIndex
from akn_rlm.rlm.ceiling_breakers import (
    is_enabled as _ceiling_enabled,
    make_concept_amendment_search,
    make_llm_doc_router_call,
    make_nli_verifier_fn,
)
from akn_rlm.rlm.enhancers import (
    HyDEDenseIndex,
    MultiQueryRetrieverWrapper,
    active_enhancers,
    is_e1_enabled,
    is_e2_enabled,
    is_e3_enabled,
    is_e4_enabled,
    is_e5_enabled,
    is_e6_enabled,
    is_e7_enabled,
    make_concept_amendment_fn,
    make_concept_kg_channel,
    make_hyde_query_enhancer,
    make_kg_doc_router_call,
    make_kg_topology_disambiguator,
    make_nli_v2_verifier_fn,
    make_query_paraphrase_fn,
)
from akn_rlm.rlm.classifier import classify as _classify_intent
from akn_rlm.rlm.handlers import (
    LAYMAN_DEFAULT_REWRITE_MODEL,
    build_conceptual_definitional_handler,
    build_exact_article_handler,
    build_layman_handler,
    build_long_context_handler,
    build_multi_hop_handler,
    build_rule_application_handler,
    build_temporal_factual_handler,
    build_unanswerable_handler,
)
from akn_rlm.rlm.routing import DocRouter, build_doc_router
from akn_rlm.rlm.sub_worker import call_summarizer
from akn_rlm.rlm.supervisor import (
    DEFAULT_MODEL as SUPERVISOR_DEFAULT_MODEL,
    DEFAULT_THRESHOLD as SUPERVISOR_DEFAULT_THRESHOLD,
    PlanSupervisorFn,
    SupervisorFn,
    supervise_citations,
    supervise_plan,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Wall-clock seconds before the long_context summariser is forcibly
#: aborted and the handler falls back to the deterministic Arabic
#: template. HANDOFF §R6.4 records a 5-hour hang on ``com_lc_q01``;
#: 60 s is the budget we picked to bound the full-244-q run.
DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S: float = 60.0

#: Telemetry tag stamped onto every dispatched answer. Lets
#: ``compare_baselines.py`` pull dispatched runs out as a single column
#: regardless of which inner handler ran.
DISPATCH_BASELINE: str = "rlm_dispatched"

#: Handler key reached when ``query_type`` is missing / unknown.
DEFAULT_FALLBACK_HANDLER: str = "rule_application"

#: Map every benchmark ``query_type`` (and the legacy ``temporal``
#: alias from ``root_controller.classify_query_type``) to its handler
#: key. Keys MUST match the eight ``query_type`` strings used by the
#: ALB v3.0 records and by ``classifier.classify``.
TYPE_TO_HANDLER: dict[str, str] = {
    "rule_application":         "rule_application",
    "exact_article":            "exact_article",
    "multi_hop":                "multi_hop",
    "unanswerable":             "unanswerable",
    "layman":                   "layman",
    "long_context":             "long_context",
    "conceptual_definitional":  "conceptual_definitional",
    "temporal_factual":         "temporal_factual",
    # Legacy alias emitted by ``root_controller.classify_query_type``;
    # the benchmark itself only ever uses ``temporal_factual``.
    "temporal":                 "temporal_factual",
}

#: Handler keys that need the KG. KG load is ~26 s; we keep the
#: rest of the dispatcher cheap by deferring this until truly needed.
_KG_HANDLER_KEYS: frozenset[str] = frozenset({
    "temporal_factual",
    "conceptual_definitional",
})


ClassifierFn = Callable[[str], str]
KGLoaderFn = Callable[[], Any]


# ---------------------------------------------------------------------------
# Timeout-wrapped summariser for long_context
# ---------------------------------------------------------------------------

def _make_timeout_summarizer(
    timeout_s: float,
    inner: Callable = call_summarizer,
) -> Callable:
    """Wrap a summariser with a thread-based wall-clock timeout.

    Why threads: the underlying LLM call is a blocking native HTTP
    request that doesn't honour ``signal.alarm`` (and ``signal.alarm``
    is unix-only — this project runs on Windows). We submit the call
    to a single-worker executor and ``.result(timeout=...)``. On
    timeout we raise ``TimeoutError`` so the long_context handler's
    existing ``except Exception`` path falls back to the template.

    The thread that's stuck on the slow LLM call cannot be cancelled
    (Python doesn't support thread cancellation for arbitrary native
    calls), so we ``shutdown(wait=False)`` and let the OS reclaim it
    when the process exits. This is acceptable for a CLI benchmark
    runner.
    """

    def wrapped(pool, query, articles, model):
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lc_summarizer"
        )
        fut = executor.submit(inner, pool, query, articles, model)
        try:
            result = fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            executor.shutdown(wait=False)
            log.warning(
                "long_context summariser exceeded %.1fs — falling back "
                "to deterministic template", timeout_s,
            )
            raise TimeoutError(
                f"long_context summariser timeout after {timeout_s:.1f}s"
            )
        executor.shutdown(wait=False)
        return result

    return wrapped


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class RLMDispatcher:
    """Wires the eight Phase-2 typed handlers behind a single
    ``run(query, query_type=None)`` entrypoint."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        kg: Any = None,
        kg_loader: Optional[KGLoaderFn] = None,
        sub_model: str = SUB_LLM_MODEL,
        rewrite_model: str = LAYMAN_DEFAULT_REWRITE_MODEL,
        classifier_fn: Optional[ClassifierFn] = None,
        long_context_timeout_s: float = DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S,
        handler_overrides: Optional[dict[str, Any]] = None,
        supervisor_fn: Optional[SupervisorFn] = supervise_citations,
        supervisor_model: str = SUPERVISOR_DEFAULT_MODEL,
        supervisor_threshold: float = SUPERVISOR_DEFAULT_THRESHOLD,
        # R9.6: plan_supervisor stays *opt-in* — empirical evidence on
        # the n=10 multi_hop slice shows it regresses Cite F1 from
        # 0.167 (R9.3 alone) to 0.070 when wired through the
        # dispatcher. The seam + tests are kept so a future operator
        # can flip it on with a stronger plan-prompt or a different
        # model.
        plan_supervisor_fn: Optional[PlanSupervisorFn] = None,
        # Ceiling-breakers: when enable_ceiling_breakers is True, the
        # dispatcher (a) builds the doc-router with an LLM tie-breaker
        # callable, (b) injects an NLI verifier_fn into RA / MH / EA,
        # and (c) hands the conceptual_definitional handler a
        # concept->amendment SPARQL helper. Defaults to the
        # AKN_CEILING_BREAKERS env flag so HPC runs flip everything
        # on with one variable.
        enable_ceiling_breakers: Optional[bool] = None,
        # Phase C — pervasive Argument Mining. When True, every
        # citation-emitting handler (RA / EA / MH / TF / LC; layman
        # rides RA) attaches a Toulmin ``argumentation`` dict to its
        # top-N citations. CD already runs ADU pre-citation-construction
        # and is left untouched. Per-query cost: +N ADU sub-LM calls
        # per non-CD handler that emits citations (default N=5).
        enable_pervasive_adu: bool = False,
        adu_extract_top_n: int = 5,
        # Phase D — genuine recursion. Wraps each citation-emitting
        # handler's retrieve+verify step in a RecursiveRetriever. The
        # gap-probe (gpt-oss-120b) decides whether to issue extra
        # depth-2/3 retrieval passes; the new candidates are merged
        # additively (never replace existing citations). Default OFF
        # for back-compat.
        enable_recursion: bool = False,
        recursion_max_depth: int = 3,
        # Phase D — corrective retry on faithfulness failure. After the
        # summariser produces an answer, run gates.faithfulness_nli; if
        # it fails, regenerate ONCE with explicit "use only cited
        # articles" feedback. Default OFF for back-compat.
        enable_corrective_retry: bool = False,
        # Phase E — per-handler `recursion_coverage_min` override.
        # Phase D measured MH/RA regressing (−0.018/−0.017 Cite F1)
        # despite the highest recursion firing rates (80.8% / 66.7%).
        # The diagnosis (HANDOFF §1.4e): the default coverage_min=2 is
        # too aggressive for the wide candidate pools MH and RA already
        # produce — gap-probe surfaces adjacent-but-wrong articles that
        # dilute the confidence-sorted top-K. Lifting coverage_min to 4
        # for MH/RA only fires the gap-probe on *genuinely thin* pools
        # while leaving TF/CD at 2 (TF +0.096 from the lower threshold).
        # Keys MUST match TYPE_TO_HANDLER handler keys; unknown keys are
        # silently ignored. ``None`` means "use each handler's default".
        recursion_coverage_min_overrides: Optional[dict[str, int]] = None,
    ) -> None:
        self._registry = registry
        self._llm_pool = llm_pool
        self._ceiling = (
            _ceiling_enabled() if enable_ceiling_breakers is None
            else bool(enable_ceiling_breakers)
        )

        # E1-E4 enhancer wiring. Read flags up front so the run() path
        # never re-checks the env mid-question. Each enhancer is built
        # at most once, fails open, and degrades to the F5 default.
        self._enh_active = active_enhancers()
        log.info("Enhancers active: %s", self._enh_active)

        # E4 (HyDE) wraps the dense index — but **selectively**.
        # Empirical full-244 result: HyDE helps RA/EA/layman/MH (BM25/Dense-
        # driven handlers) and HURTS temporal_factual / conceptual_
        # definitional (handlers that resolve the gold via the KG; a
        # generic hypothetical answer drifts retrieval away from the
        # specific in-force version / definition). We keep both:
        #   self._dense_raw   — plain dense for TF and CD
        #   self._dense       — HyDE-wrapped dense for the others
        self._dense_raw = dense
        if is_e4_enabled():
            try:
                hyde_fn = make_hyde_query_enhancer(llm_pool)
                dense = HyDEDenseIndex(dense, hyde_fn)
                log.info("E4 HyDE: dense wrapped (selective: off for TF/CD)")
            except Exception as exc:
                log.warning("E4 HyDE wiring failed (%s) — falling back to plain dense", exc)
                self._dense_raw = dense

        # E3 paraphrase wrapper. Applies to BOTH BM25 and Dense so
        # every handler that fuses RRF(BM25, Dense) benefits.
        if is_e3_enabled():
            try:
                pf = make_query_paraphrase_fn(llm_pool)
                bm25 = MultiQueryRetrieverWrapper(bm25, pf)
                dense = MultiQueryRetrieverWrapper(dense, pf)
                log.info("E3 paraphrase: BM25 + Dense wrapped with multi-query fusion")
            except Exception as exc:
                log.warning("E3 paraphrase wiring failed (%s) — falling back", exc)

        self._bm25 = bm25
        self._dense = dense

        # E2 — reverse-NLI verifier. Built lazily; the v1 NLI verifier
        # from ceiling_breakers stays available only when enable_
        # ceiling_breakers is on and E2 is off (back-compat).
        self._e2_verifier_fn = None
        if is_e2_enabled():
            try:
                self._e2_verifier_fn = make_nli_v2_verifier_fn(llm_pool)
                log.info("E2 reverse-NLI verifier active")
            except Exception as exc:
                log.warning("E2 NLI v2 wiring failed (%s) — leaving F5 verifier", exc)

        # E1 — concept_amendment helper. Built lazily once the KG loads.
        self._e1_concept_amendment_fn = None
        # E5 — Phase E.2 KG topology disambiguator. Built lazily on
        # first MH dispatch when AKN_E5_KG_TOPOLOGY=1. None means "not
        # yet built"; an identity callable (``lambda q, c: list(c)``)
        # means "build failed — degrade to F5 behaviour silently".
        self._e5_disambiguator_fn: Optional[Callable] = None
        # E6 — Phase E.3 concept-KG retrieval channel. Built lazily on
        # first MH/RA dispatch when AKN_E6_CONCEPT_KG=1. None means
        # "not yet built"; a no-op callable means "build failed".
        self._e6_concept_kg_channel_fn: Optional[Callable] = None
        # E7 — Phase E.4 KG-derived doc-router channel. Attached to
        # the router lazily on first run when AKN_E7_KG_DOC_ROUTER=1.
        # ``_e7_attached`` flips to True once attached (or once a
        # failed-attach has been logged) so we don't re-try every run.
        self._e7_attached: bool = False
        # Build (or reuse) the doc-router; turn on the LLM tie-breaker
        # channel when ceiling-breakers are enabled.
        if router is None:
            router_kwargs: dict[str, Any] = {}
            if self._ceiling:
                router_kwargs["llm_call"] = make_llm_doc_router_call(llm_pool)
            self._router = build_doc_router(
                registry=registry, bm25=bm25, **router_kwargs,
            )
        else:
            self._router = router
        # Pre-built closures used in handler construction.
        self._nli_verifier_fn = make_nli_verifier_fn() if self._ceiling else None
        # Concept-amendment SPARQL is built lazily once the KG loads.
        self._concept_amendment_search = None
        self._kg = kg
        self._kg_loader = kg_loader
        self._sub_model = sub_model
        self._rewrite_model = rewrite_model
        self._classifier_fn = classifier_fn
        self._long_context_timeout_s = float(long_context_timeout_s)
        # R9.5 + R9.6: supervisor seams. ``supervisor_fn`` is the
        # per-citation re-ranker injected into the LLM-using handlers
        # (RA, MH, EA, layman, LC). ``plan_supervisor_fn`` is the
        # multi_hop sub-question planner from R9.6.
        if supervisor_fn is not None:
            def _bound_supervisor(pool, q, cits):
                return supervisor_fn(
                    pool, q, cits,
                    threshold=supervisor_threshold,
                    model=supervisor_model,
                )
            self._supervisor_fn: Optional[SupervisorFn] = _bound_supervisor
        else:
            self._supervisor_fn = None
        self._plan_supervisor_fn = plan_supervisor_fn
        # Phase C — pervasive ADU toggles. Stored eagerly so _build()
        # forwards them to each non-CD handler when constructing.
        self._enable_pervasive_adu = bool(enable_pervasive_adu)
        self._adu_extract_top_n = int(adu_extract_top_n)
        # Phase D — recursion + corrective retry toggles.
        self._enable_recursion = bool(enable_recursion)
        self._recursion_max_depth = int(recursion_max_depth)
        self._enable_corrective_retry = bool(enable_corrective_retry)
        # Phase E — per-handler coverage_min override map. Stored as a
        # plain dict; _build() forwards the per-key value into the
        # handler's recursion_coverage_min kwarg when present.
        self._recursion_coverage_min_overrides: dict[str, int] = (
            dict(recursion_coverage_min_overrides)
            if recursion_coverage_min_overrides
            else {}
        )
        # Pre-populated overrides are honoured as-is — the dispatcher
        # never replaces an injected handler. Tests use this to inject
        # mocks; production code can use it to swap in a custom
        # implementation.
        self._handlers: dict[str, Any] = dict(handler_overrides or {})

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        query_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Dispatch ``query`` to the handler chosen by ``query_type``."""
        if not isinstance(query, str) or not query.strip():
            return self._abstain_envelope(
                reason="empty_query",
                dispatched_handler=None,
                dispatched_query_type=query_type,
            )

        # Phase E.4 — lazy-attach the KG doc-router channel on first
        # run when E7 is enabled. We don't fire KG load on construction
        # so MH/RA-only runs without E7 keep their startup cost low.
        if is_e7_enabled() and not self._e7_attached:
            try:
                kg = self._get_kg()
                sparql_fn = self._make_sparql_fn(kg)
                self._router.set_kg_call(make_kg_doc_router_call(sparql_fn))
                log.info("E7 KG doc-router channel attached to router")
            except Exception as exc:
                log.warning("E7 KG doc-router attach failed (%s)", exc)
            self._e7_attached = True

        resolved_type = (query_type or "").strip() or self._classify(query)
        handler_key = TYPE_TO_HANDLER.get(resolved_type, DEFAULT_FALLBACK_HANDLER)

        try:
            handler = self._get_handler(handler_key)
        except Exception as exc:
            log.error(
                "Dispatcher could not build handler %s: %s",
                handler_key, exc,
            )
            return self._abstain_envelope(
                reason="dispatch_build_error",
                dispatched_handler=handler_key,
                dispatched_query_type=resolved_type,
                error=str(exc),
            )

        # R9.7: per-handler-run call recording. The pool exposes
        # start_recording / stop_recording; we snapshot on entry and
        # exit so even an exception path leaves the pool in a clean
        # state. Mocks that don't implement the methods are tolerated.
        recording_active = False
        if hasattr(self._llm_pool, "start_recording"):
            try:
                self._llm_pool.start_recording()
                recording_active = True
            except Exception:
                recording_active = False

        try:
            answer = handler.run(query)
        except Exception as exc:
            if recording_active:
                try:
                    self._llm_pool.stop_recording()
                except Exception:
                    pass
            log.error(
                "Dispatcher handler %s raised: %s", handler_key, exc,
            )
            return self._abstain_envelope(
                reason="dispatch_pipeline_error",
                dispatched_handler=handler_key,
                dispatched_query_type=resolved_type,
                error=str(exc),
            )

        calls_by_model: dict[str, int] = {}
        if recording_active:
            try:
                calls_by_model = self._llm_pool.stop_recording()
            except Exception:
                calls_by_model = {}

        if not isinstance(answer, dict):
            log.error(
                "Dispatcher handler %s returned non-dict: %r",
                handler_key, type(answer).__name__,
            )
            return self._abstain_envelope(
                reason="dispatch_bad_answer_shape",
                dispatched_handler=handler_key,
                dispatched_query_type=resolved_type,
            )

        # Patch telemetry: keep the inner handler's tag (so per-handler
        # eval runs still classify correctly) and add the dispatcher's
        # bookkeeping.
        tel = answer.setdefault("_telemetry", {})
        if not isinstance(tel, dict):
            tel = {}
            answer["_telemetry"] = tel
        tel["dispatched_handler"] = handler_key
        tel["dispatched_query_type"] = resolved_type
        tel["dispatch_baseline"] = DISPATCH_BASELINE
        # Preserve the inner handler's "baseline" if present; otherwise
        # stamp the dispatcher tag so downstream tooling never sees a
        # blank baseline field.
        tel.setdefault("baseline", DISPATCH_BASELINE)
        # R9.7: per-handler-run model counts.
        tel["calls_by_model"] = calls_by_model
        # E1-E4 ablation telemetry — every dispatched answer carries the
        # active enhancer set so we can audit which run produced what.
        tel["enhancers_active"] = dict(self._enh_active)
        return answer

    # ------------------------------------------------------------------
    # Type resolution
    # ------------------------------------------------------------------

    def _classify(self, query: str) -> str:
        """Resolve ``query_type`` when the caller didn't supply one."""
        if self._classifier_fn is not None:
            try:
                return self._classifier_fn(query)
            except Exception as exc:
                log.warning(
                    "Custom classifier raised %s — falling back to "
                    "akn_rlm.rlm.classifier", exc,
                )
        try:
            return _classify_intent(query).query_type
        except Exception as exc:
            log.warning(
                "Default classifier raised %s — falling back to %s",
                exc, DEFAULT_FALLBACK_HANDLER,
            )
            return DEFAULT_FALLBACK_HANDLER

    # ------------------------------------------------------------------
    # Lazy handler construction
    # ------------------------------------------------------------------

    def _get_handler(self, key: str) -> Any:
        if key not in self._handlers:
            self._handlers[key] = self._build(key)
        return self._handlers[key]

    def _build(self, key: str) -> Any:
        common: dict[str, Any] = dict(
            bm25=self._bm25,
            dense=self._dense,
            registry=self._registry,
            llm_pool=self._llm_pool,
            router=self._router,
            sub_model=self._sub_model,
        )
        # Step 3 — per-citation NLI verifier replaces the LLM verifier in
        # RA / MH / EA when ceiling-breakers are on. The NLI verifier
        # falls back to ``call_verifier`` if the NLI model can't load,
        # so HPC nodes without sentence-transformers / mDeBERTa still
        # work.
        ceiling_kwargs: dict[str, Any] = {}
        # E2 takes precedence over the legacy v1 (wrong NLI direction).
        if self._e2_verifier_fn is not None:
            ceiling_kwargs["verifier_fn"] = self._e2_verifier_fn
        elif self._ceiling and self._nli_verifier_fn is not None:
            ceiling_kwargs["verifier_fn"] = self._nli_verifier_fn

        # Phase C — pervasive ADU kwargs forwarded into every
        # citation-emitting handler. CD owns its own ADU budget.
        adu_kwargs: dict[str, Any] = {}
        if self._enable_pervasive_adu:
            adu_kwargs["enable_adu_extraction"] = True
            adu_kwargs["adu_extract_top_n"] = self._adu_extract_top_n

        # Phase D — recursion + corrective-retry kwargs. Only forwarded
        # to the four handlers that have wired the toggles (RA, MH, TF,
        # CD); EA/LC/layman/UA are unchanged.
        phase_d_kwargs: dict[str, Any] = {}
        if self._enable_recursion:
            phase_d_kwargs["enable_recursion"] = True
            phase_d_kwargs["recursion_max_depth"] = self._recursion_max_depth
            # Phase E: per-handler coverage_min override. Only injected
            # when recursion is enabled AND this handler key is listed
            # in the override map — otherwise the handler keeps its
            # constructor default (DEFAULT_COVERAGE_MIN=2).
            override = self._recursion_coverage_min_overrides.get(key)
            if override is not None:
                phase_d_kwargs["recursion_coverage_min"] = int(override)
        if self._enable_corrective_retry:
            phase_d_kwargs["enable_corrective_retry"] = True

        if key == "rule_application":
            ra_extra: dict[str, Any] = {}
            # Phase E.3 — concept-KG channel for RA. The channel adds
            # candidates (Fix-TF pattern), so it doesn't matter that
            # RA has no consensus boost / disambiguator — it just
            # widens the pool before verification.
            if is_e6_enabled():
                ch = self._get_e6_concept_kg_channel()
                if ch is not None:
                    ra_extra["concept_kg_channel_fn"] = ch
            return build_rule_application_handler(
                **common,
                supervisor_fn=self._supervisor_fn,
                **ceiling_kwargs,
                **adu_kwargs,
                **phase_d_kwargs,
                **ra_extra,
            )
        if key == "exact_article":
            # exact_article never uses dense — see HANDOFF §R6.2.
            return build_exact_article_handler(
                bm25=self._bm25,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
                supervisor_fn=self._supervisor_fn,
                **ceiling_kwargs,
                **adu_kwargs,
            )
        if key == "multi_hop":
            mh_kwargs: dict[str, Any] = dict(common)
            mh_kwargs["supervisor_fn"] = self._supervisor_fn
            if self._plan_supervisor_fn is not None:
                mh_kwargs["plan_supervisor_fn"] = self._plan_supervisor_fn
            mh_kwargs.update(ceiling_kwargs)
            mh_kwargs.update(adu_kwargs)
            mh_kwargs.update(phase_d_kwargs)
            # Phase E.2 — inject the KG topology disambiguator when E5
            # is active. The lazy-build triggers KG loading on first MH
            # dispatch only; an MH-only run with E5 off pays no KG cost.
            if is_e5_enabled():
                disamb = self._get_e5_disambiguator()
                if disamb is not None:
                    mh_kwargs["kg_topology_disambiguator_fn"] = disamb
            # Phase E.3 — inject the concept-KG channel when E6 active.
            if is_e6_enabled():
                ch = self._get_e6_concept_kg_channel()
                if ch is not None:
                    mh_kwargs["concept_kg_channel_fn"] = ch
            return build_multi_hop_handler(**mh_kwargs)
        if key == "long_context":
            timeout_summarizer = _make_timeout_summarizer(
                self._long_context_timeout_s
            )
            # Fix-LC: turn on AKN chapter/section neighbor expansion for
            # the dispatched-production path. Long-context gold sets are
            # whole sections; broad RRF alone catches 2-3 of 4-6 gold
            # articles, neighbors close the gap.
            return build_long_context_handler(
                bm25=self._bm25,
                dense=self._dense,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
                summarizer_fn=timeout_summarizer,
                supervisor_fn=self._supervisor_fn,
                enable_chapter_expansion=True,
                **adu_kwargs,
            )
        if key == "layman":
            # Pervasive ADU lives inside the rule_application handler
            # that layman wraps; forward the toggle via
            # ``rule_handler_kwargs`` (build_layman_handler passes
            # **kwargs to its child RA handler).
            return build_layman_handler(
                bm25=self._bm25,
                dense=self._dense,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
                rewrite_model=self._rewrite_model,
                supervisor_fn=self._supervisor_fn,
                **adu_kwargs,
            )
        if key == "unanswerable":
            return build_unanswerable_handler(
                bm25=self._bm25,
                dense=self._dense,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
            )
        if key in _KG_HANDLER_KEYS:
            kg = self._get_kg()
            if key == "temporal_factual":
                # E4 selective: use the RAW dense (no HyDE) — empirically
                # HyDE hurts TF by drifting retrieval away from the
                # specific in-force version.
                return build_temporal_factual_handler(
                    kg=kg,
                    bm25=self._bm25,
                    dense=self._dense_raw,
                    registry=self._registry,
                    llm_pool=self._llm_pool,
                    router=self._router,
                    sub_model=self._sub_model,
                    **adu_kwargs,
                    **phase_d_kwargs,
                )
            if key == "conceptual_definitional":
                # E4 selective: use the RAW dense (no HyDE) — empirically
                # HyDE hurts CD because the handler already does its own
                # paraphrase widening; HyDE's hypothetical answer
                # conflicts with that.
                cd_kwargs: dict[str, Any] = dict(
                    kg=kg,
                    bm25=self._bm25,
                    dense=self._dense_raw,
                    registry=self._registry,
                    llm_pool=self._llm_pool,
                    router=self._router,
                    sub_model=self._sub_model,
                )
                # E1 — build the concept-amendment helper now that we
                # have the KG. Cached on self so a second CD dispatch
                # reuses it without re-building.
                if is_e1_enabled():
                    if self._e1_concept_amendment_fn is None:
                        try:
                            self._e1_concept_amendment_fn = make_concept_amendment_fn(kg)
                            log.info("E1 concept->amendment helper built")
                        except Exception as exc:
                            log.warning("E1 helper build failed (%s)", exc)
                    if self._e1_concept_amendment_fn is not None:
                        cd_kwargs["concept_amendment_fn"] = self._e1_concept_amendment_fn
                cd_kwargs.update(phase_d_kwargs)
                return build_conceptual_definitional_handler(**cd_kwargs)
        # Unreachable when TYPE_TO_HANDLER is exhaustive — guard anyway
        # so a typo in a future key is loud, not silent.
        raise ValueError(f"Unknown dispatcher handler key: {key!r}")

    def _get_kg(self) -> Any:
        if self._kg is None:
            if self._kg_loader is None:
                raise RuntimeError(
                    "Handler requires the KG but neither `kg` nor "
                    "`kg_loader` was supplied to RLMDispatcher."
                )
            log.info("Lazy-loading KG via supplied loader …")
            self._kg = self._kg_loader()
        return self._kg

    # ------------------------------------------------------------------
    # E5 — KG topology disambiguator (Phase E.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sparql_fn(kg: Any) -> Optional[Callable[[str], Any]]:
        """Mirror of ``temporal_factual._default_sparql_fn`` so MH can
        access the same KG without depending on a TF instance."""
        if kg is None:
            return None

        def _fn(query: str) -> Any:
            try:
                results = kg.query(query)
            except Exception as exc:
                log.debug("E5 SPARQL query failed: %s", exc)
                return []
            stripped = query.strip().lower()
            if stripped.startswith("ask"):
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

    def _get_e6_concept_kg_channel(self) -> Optional[Callable]:
        """Build (or return cached) Phase E.3 concept-KG channel.

        Triggers KG lazy-load on first call, so MH/RA-only runs without
        E6 still avoid the ~26 s rdflib parse. On any build failure
        installs a no-op callable so subsequent dispatches degrade
        cleanly to F5 hybrid-only retrieval.
        """
        if self._e6_concept_kg_channel_fn is not None:
            return self._e6_concept_kg_channel_fn
        try:
            kg = self._get_kg()
            sparql_fn = self._make_sparql_fn(kg)
            self._e6_concept_kg_channel_fn = make_concept_kg_channel(sparql_fn)
            log.info("E6 concept-KG channel built")
        except Exception as exc:
            log.warning("E6 concept-KG channel build failed (%s)", exc)
            self._e6_concept_kg_channel_fn = lambda _q, _r=None: []
        return self._e6_concept_kg_channel_fn

    def _get_e5_disambiguator(self) -> Optional[Callable]:
        """Build (or return cached) Phase E.2 KG topology disambiguator.

        Triggers KG lazy-load on first call, so MH-only runs without
        E5 still avoid the ~26 s rdflib parse. On any build failure
        installs an identity callable so subsequent dispatches degrade
        cleanly to the F5 ranking.
        """
        if self._e5_disambiguator_fn is not None:
            return self._e5_disambiguator_fn
        try:
            kg = self._get_kg()
            sparql_fn = self._make_sparql_fn(kg)
            # _resolve_article_uri lives in temporal_factual but is
            # generic over any sparql_fn — import lazily to avoid the
            # heavy TF handler module load when E5 is off.
            from akn_rlm.rlm.handlers.temporal_factual import (
                _resolve_article_uri,
            )

            def _bound_resolve(doc_id: str, ref: str) -> Optional[str]:
                return _resolve_article_uri(sparql_fn, doc_id, ref)

            self._e5_disambiguator_fn = make_kg_topology_disambiguator(
                sparql_fn, resolve_uri=_bound_resolve,
            )
            log.info("E5 KG topology disambiguator built")
        except Exception as exc:
            log.warning("E5 disambiguator build failed (%s)", exc)
            self._e5_disambiguator_fn = lambda _q, c: list(c)
        return self._e5_disambiguator_fn

    # ------------------------------------------------------------------
    # Abstention envelope shared by every error path
    # ------------------------------------------------------------------

    @staticmethod
    def _abstain_envelope(
        *,
        reason: str,
        dispatched_handler: Optional[str],
        dispatched_query_type: Optional[str],
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        telemetry: dict[str, Any] = {
            "retry_count":            0,
            "gate_results":           {},
            "baseline":               DISPATCH_BASELINE,
            "dispatch_baseline":      DISPATCH_BASELINE,
            "dispatched_handler":     dispatched_handler,
            "dispatched_query_type":  dispatched_query_type,
            "sub_call_count":         0,
        }
        if error is not None:
            telemetry["error"] = error
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [
                f"dispatched_query_type={dispatched_query_type}",
                f"dispatched_handler={dispatched_handler}",
                f"reason={reason}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry":        telemetry,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dispatcher(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    kg: Any = None,
    kg_loader: Optional[KGLoaderFn] = None,
    **kwargs: Any,
) -> RLMDispatcher:
    """Factory mirroring the per-handler ``build_*_handler`` helpers."""
    return RLMDispatcher(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        kg=kg,
        kg_loader=kg_loader,
        **kwargs,
    )
