"""Tests for the Phase-2 / R7 dispatcher.

The dispatcher must route ``query_type`` → typed handler, lazy-build
KG-loading handlers (temporal_factual / conceptual_definitional) only
when actually needed, wrap the long_context summariser with a wall-
clock timeout, and stamp dispatched telemetry without clobbering the
inner handler's own ``baseline`` tag.

Every handler is mocked via ``handler_overrides``; this suite never
touches a real index, LLM, or KG, and runs in milliseconds.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from akn_rlm.eval.runner import _answer_to_result
from akn_rlm.rlm.dispatcher import (
    DEFAULT_FALLBACK_HANDLER,
    DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S,
    DISPATCH_BASELINE,
    TYPE_TO_HANDLER,
    RLMDispatcher,
    _make_timeout_summarizer,
    build_dispatcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_handler(
    *,
    answer_text: str = "نص الإجابة",
    citations: list | None = None,
    abstention: bool = False,
    abstention_reason: str | None = None,
    baseline: str = "rlm_stub",
    sub_call_count: int = 1,
    extra_telemetry: dict | None = None,
):
    """Return a MagicMock with a `.run(query) -> dict` baseline-shaped reply."""
    mock = MagicMock()
    payload = {
        "answer_text":       answer_text,
        "abstention":        abstention,
        "abstention_reason": abstention_reason,
        "citations":         citations if citations is not None else [
            {
                "doc_id":          "84-11_1984-06-09",
                "article_ref":     "5",
                "doc_title":       "قانون الأسرة",
                "supporting_span": "نص الدعم",
                "text":            "نص المادة الكامل",
                "confidence":      0.9,
            }
        ],
        "reasoning_chain":   ["step 1"],
        "trajectory":        [],
        "tokens_used":       1024,
        "depth_max_reached": 1,
        "_telemetry": {
            "retry_count":     0,
            "gate_results":    {},
            "baseline":        baseline,
            "sub_call_count":  sub_call_count,
            **(extra_telemetry or {}),
        },
    }
    mock.run.return_value = payload
    return mock


def _make_dispatcher(
    *,
    overrides: dict | None = None,
    classifier_fn=None,
    kg=None,
    kg_loader=None,
    long_context_timeout_s: float = DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S,
) -> RLMDispatcher:
    return RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=kg,
        kg_loader=kg_loader,
        classifier_fn=classifier_fn,
        long_context_timeout_s=long_context_timeout_s,
        handler_overrides=overrides or {},
    )


# ---------------------------------------------------------------------------
# Defaults & contract
# ---------------------------------------------------------------------------


def test_dispatch_baseline_is_rlm_dispatched():
    assert DISPATCH_BASELINE == "rlm_dispatched"


def test_default_long_context_timeout_is_60s():
    assert DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S == 60.0


def test_default_fallback_handler_is_rule_application():
    assert DEFAULT_FALLBACK_HANDLER == "rule_application"


def test_type_map_covers_all_eight_benchmark_types():
    """Every ALB v3.0 query_type must dispatch to a real handler."""
    expected = {
        "rule_application", "exact_article", "multi_hop", "unanswerable",
        "layman", "long_context", "conceptual_definitional", "temporal_factual",
    }
    assert expected.issubset(set(TYPE_TO_HANDLER.keys()))


def test_type_map_aliases_temporal_to_temporal_factual():
    """Legacy ``temporal`` from root_controller.classify_query_type
    must coalesce to the canonical ``temporal_factual`` handler."""
    assert TYPE_TO_HANDLER["temporal"] == "temporal_factual"


def test_factory_builds():
    dispatcher = build_dispatcher(
        bm25=MagicMock(), dense=MagicMock(), registry=MagicMock(),
        llm_pool=MagicMock(), router=MagicMock(),
    )
    assert isinstance(dispatcher, RLMDispatcher)


def test_run_returns_baseline_shaped_dict():
    handler = _stub_handler()
    disp = _make_dispatcher(overrides={"rule_application": handler})
    answer = disp.run("ما هي شروط الزواج؟", query_type="rule_application")
    for key in (
        "answer_text", "abstention", "citations", "reasoning_chain",
        "trajectory", "tokens_used", "_telemetry",
    ):
        assert key in answer


# ---------------------------------------------------------------------------
# Routing — explicit query_type wins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qt",
    [
        "rule_application", "exact_article", "multi_hop", "unanswerable",
        "layman", "long_context", "conceptual_definitional", "temporal_factual",
    ],
)
def test_run_routes_each_query_type_to_its_handler(qt):
    handlers = {qt: _stub_handler()}
    # Inject stubs for every other type too so the dispatcher cannot
    # accidentally fall through to a real factory.
    for other in TYPE_TO_HANDLER.values():
        handlers.setdefault(other, _stub_handler())
    disp = _make_dispatcher(overrides=handlers)

    disp.run("سؤال ما", query_type=qt)

    assert handlers[qt].run.call_count == 1
    for other_key, other_h in handlers.items():
        if other_key == qt:
            continue
        assert other_h.run.call_count == 0, (
            f"handler {other_key} should not have been called for {qt}"
        )


def test_legacy_temporal_routes_to_temporal_factual_handler():
    tf = _stub_handler()
    disp = _make_dispatcher(overrides={"temporal_factual": tf})
    disp.run("متى صدر القانون؟", query_type="temporal")
    assert tf.run.call_count == 1


def test_unknown_type_falls_back_to_rule_application():
    fallback = _stub_handler()
    disp = _make_dispatcher(overrides={"rule_application": fallback})
    disp.run("سؤال ما", query_type="totally_made_up_type")
    assert fallback.run.call_count == 1


def test_handler_receives_original_query():
    handler = _stub_handler()
    disp = _make_dispatcher(overrides={"rule_application": handler})
    disp.run("سؤال محدد", query_type="rule_application")
    handler.run.assert_called_once_with("سؤال محدد")


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_dispatched_handler_recorded_in_telemetry():
    handler = _stub_handler(baseline="rlm_multi_hop")
    disp = _make_dispatcher(overrides={"multi_hop": handler})
    answer = disp.run("سؤال متعدد", query_type="multi_hop")
    tel = answer["_telemetry"]
    assert tel["dispatched_handler"] == "multi_hop"
    assert tel["dispatched_query_type"] == "multi_hop"
    assert tel["dispatch_baseline"] == DISPATCH_BASELINE


def test_inner_baseline_tag_is_preserved():
    """compare_baselines.py keys per-handler runs by the ``baseline``
    field — the dispatcher must NOT clobber it."""
    handler = _stub_handler(baseline="rlm_multi_hop")
    disp = _make_dispatcher(overrides={"multi_hop": handler})
    answer = disp.run("سؤال", query_type="multi_hop")
    assert answer["_telemetry"]["baseline"] == "rlm_multi_hop"


def test_inner_handler_telemetry_keys_preserved():
    handler = _stub_handler(
        baseline="rlm_multi_hop",
        extra_telemetry={"routed_doc_ids": ["84-11_1984-06-09"], "sub_call_count": 7},
    )
    disp = _make_dispatcher(overrides={"multi_hop": handler})
    answer = disp.run("س", query_type="multi_hop")
    tel = answer["_telemetry"]
    assert tel["routed_doc_ids"] == ["84-11_1984-06-09"]
    assert tel["sub_call_count"] == 7


def test_legacy_temporal_query_type_recorded_as_resolved():
    tf = _stub_handler(baseline="rlm_temporal_factual")
    disp = _make_dispatcher(overrides={"temporal_factual": tf})
    answer = disp.run("متى؟", query_type="temporal")
    tel = answer["_telemetry"]
    assert tel["dispatched_handler"] == "temporal_factual"
    assert tel["dispatched_query_type"] == "temporal"


# ---------------------------------------------------------------------------
# Classifier fallback when query_type is missing
# ---------------------------------------------------------------------------


def test_no_query_type_uses_default_classifier():
    """Without query_type the dispatcher falls back to
    ``akn_rlm.rlm.classifier.classify`` — exact_article queries should
    land on the exact_article handler via the regex."""
    ea = _stub_handler()
    rule = _stub_handler()
    disp = _make_dispatcher(overrides={
        "exact_article": ea, "rule_application": rule,
    })
    disp.run("ما هو نص المادة 7 من القانون المدني؟")
    assert ea.run.call_count == 1
    assert rule.run.call_count == 0


def test_custom_classifier_fn_is_called():
    seen = {}

    def fake_classify(query: str) -> str:
        seen["query"] = query
        return "multi_hop"

    mh = _stub_handler()
    disp = _make_dispatcher(
        overrides={"multi_hop": mh},
        classifier_fn=fake_classify,
    )
    disp.run("سؤال غامض")
    assert seen["query"] == "سؤال غامض"
    assert mh.run.call_count == 1


def test_classifier_exception_falls_back_to_rule_application():
    def boom(_q):
        raise RuntimeError("bad classifier")

    rule = _stub_handler()
    # Inject every handler as a stub so the failing classifier path
    # cannot accidentally land on a real factory.
    overrides = {key: _stub_handler() for key in set(TYPE_TO_HANDLER.values())}
    overrides["rule_application"] = rule
    disp = _make_dispatcher(overrides=overrides, classifier_fn=boom)
    disp.run("سؤال")
    assert rule.run.call_count == 1


def test_explicit_query_type_skips_classifier():
    fake_classify = MagicMock(return_value="multi_hop")
    rule = _stub_handler()
    mh = _stub_handler()
    disp = _make_dispatcher(
        overrides={"rule_application": rule, "multi_hop": mh},
        classifier_fn=fake_classify,
    )
    disp.run("سؤال", query_type="rule_application")
    fake_classify.assert_not_called()
    assert rule.run.call_count == 1


def test_blank_query_type_falls_through_to_classifier():
    fake_classify = MagicMock(return_value="multi_hop")
    mh = _stub_handler()
    disp = _make_dispatcher(
        overrides={"multi_hop": mh},
        classifier_fn=fake_classify,
    )
    disp.run("سؤال", query_type="   ")
    fake_classify.assert_called_once()
    assert mh.run.call_count == 1


# ---------------------------------------------------------------------------
# Lazy KG loading
# ---------------------------------------------------------------------------


def test_kg_handler_calls_loader_lazily():
    sentinel_kg = object()
    loader = MagicMock(return_value=sentinel_kg)
    tf = _stub_handler()
    # Inject only temporal_factual override so the dispatcher path
    # exercises the KG branch but doesn't actually call the real
    # build_temporal_factual_handler.
    disp = _make_dispatcher(
        overrides={"temporal_factual": tf},
        kg_loader=loader,
    )
    # First, dispatch a non-KG type — loader must NOT fire.
    disp._handlers["rule_application"] = _stub_handler()  # type: ignore[attr-defined]
    disp.run("سؤال", query_type="rule_application")
    assert loader.call_count == 0

    # Now dispatch a KG type — but only the override path is used,
    # so the loader still doesn't fire (the override short-circuits
    # _build). The lazy contract is: loader fires only inside _build,
    # which is only called for keys NOT pre-populated.
    disp.run("متى؟", query_type="temporal_factual")
    assert loader.call_count == 0
    assert tf.run.call_count == 1


def test_kg_loader_required_for_kg_handler_when_no_override(monkeypatch):
    """If neither `kg` nor `kg_loader` is supplied AND no override exists
    for a KG-using handler key, the dispatcher must surface a clear
    build error."""
    # Patch the real KG factories to MagicMocks so the dispatcher's
    # _build branch reaches _get_kg() before constructing anything.
    fake_factory = MagicMock(return_value=_stub_handler())
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_temporal_factual_handler",
        fake_factory,
    )
    disp = _make_dispatcher(overrides={}, kg=None, kg_loader=None)
    answer = disp.run("متى؟", query_type="temporal_factual")
    assert answer["abstention"] is True
    assert answer["abstention_reason"] == "dispatch_build_error"
    fake_factory.assert_not_called()  # KG fetch failed before factory ran


def test_kg_loader_called_once_across_multiple_kg_dispatches(monkeypatch):
    sentinel = object()
    loader = MagicMock(return_value=sentinel)

    captured_kgs: list = []

    def fake_tf_factory(**kwargs):
        captured_kgs.append(kwargs.get("kg"))
        return _stub_handler()

    def fake_cd_factory(**kwargs):
        captured_kgs.append(kwargs.get("kg"))
        return _stub_handler()

    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_temporal_factual_handler",
        fake_tf_factory,
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_conceptual_definitional_handler",
        fake_cd_factory,
    )

    disp = _make_dispatcher(kg=None, kg_loader=loader)
    disp.run("متى؟", query_type="temporal_factual")
    disp.run("ما تعريف؟", query_type="conceptual_definitional")
    # Loader fires exactly once even though two KG handlers were built.
    assert loader.call_count == 1
    # Both handlers received the same KG instance.
    assert captured_kgs == [sentinel, sentinel]


def test_pre_supplied_kg_skips_loader(monkeypatch):
    sentinel = object()
    loader = MagicMock()
    captured = {}

    def fake_factory(**kwargs):
        captured["kg"] = kwargs.get("kg")
        return _stub_handler()

    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_temporal_factual_handler",
        fake_factory,
    )
    disp = _make_dispatcher(kg=sentinel, kg_loader=loader)
    disp.run("متى؟", query_type="temporal_factual")
    assert captured["kg"] is sentinel
    loader.assert_not_called()


# ---------------------------------------------------------------------------
# Lazy handler caching
# ---------------------------------------------------------------------------


def test_handler_constructed_once_across_calls(monkeypatch):
    factory = MagicMock(return_value=_stub_handler())
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_rule_application_handler",
        factory,
    )
    disp = _make_dispatcher()
    disp.run("س1", query_type="rule_application")
    disp.run("س2", query_type="rule_application")
    disp.run("س3", query_type="rule_application")
    assert factory.call_count == 1


def test_unrelated_handlers_not_built(monkeypatch):
    """Dispatching `rule_application` must NOT build the multi_hop or
    layman handlers — saves startup cost on per-type slices."""
    ra_factory = MagicMock(return_value=_stub_handler())
    mh_factory = MagicMock(return_value=_stub_handler())
    lm_factory = MagicMock(return_value=_stub_handler())
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_rule_application_handler", ra_factory,
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler", mh_factory,
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_layman_handler", lm_factory,
    )
    disp = _make_dispatcher()
    disp.run("س", query_type="rule_application")
    assert ra_factory.call_count == 1
    mh_factory.assert_not_called()
    lm_factory.assert_not_called()


def test_handler_overrides_take_precedence_over_factories(monkeypatch):
    factory = MagicMock(return_value=_stub_handler())
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_rule_application_handler", factory,
    )
    override = _stub_handler()
    disp = _make_dispatcher(overrides={"rule_application": override})
    disp.run("س", query_type="rule_application")
    factory.assert_not_called()
    assert override.run.call_count == 1


# ---------------------------------------------------------------------------
# Long-context timeout wrapping
# ---------------------------------------------------------------------------


def test_long_context_factory_receives_timeout_summarizer(monkeypatch):
    captured: dict = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return _stub_handler()

    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_long_context_handler", fake_factory,
    )
    disp = _make_dispatcher()
    disp.run("س", query_type="long_context")
    assert "summarizer_fn" in captured
    assert callable(captured["summarizer_fn"])


def test_timeout_summarizer_returns_inner_result_under_budget():
    inner = MagicMock(return_value={"summary": "ok", "key_articles": []})
    wrapped = _make_timeout_summarizer(timeout_s=2.0, inner=inner)
    result = wrapped(MagicMock(), "س", [{"text": "نص"}], "model")
    assert result["summary"] == "ok"
    inner.assert_called_once()


def test_timeout_summarizer_raises_timeout_error_on_overrun():
    def slow(*_args, **_kwargs):
        time.sleep(1.0)
        return {"summary": "should not return"}

    wrapped = _make_timeout_summarizer(timeout_s=0.2, inner=slow)
    with pytest.raises(TimeoutError):
        wrapped(MagicMock(), "س", [], "model")


def test_timeout_summarizer_propagates_inner_exception():
    def boom(*_args, **_kwargs):
        raise RuntimeError("LLM down")

    wrapped = _make_timeout_summarizer(timeout_s=2.0, inner=boom)
    with pytest.raises(RuntimeError, match="LLM down"):
        wrapped(MagicMock(), "س", [], "model")


def test_long_context_timeout_is_configurable(monkeypatch):
    captured: dict = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return _stub_handler()

    def fake_make_timeout_summarizer(timeout_s, inner=None):
        captured["timeout_s"] = timeout_s
        return MagicMock()

    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_long_context_handler", fake_factory,
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher._make_timeout_summarizer",
        fake_make_timeout_summarizer,
    )
    disp = _make_dispatcher(long_context_timeout_s=15.0)
    disp.run("س", query_type="long_context")
    assert captured["timeout_s"] == 15.0


# ---------------------------------------------------------------------------
# Error & abstention paths
# ---------------------------------------------------------------------------


def test_empty_query_returns_dispatcher_abstention():
    disp = _make_dispatcher()
    answer = disp.run("", query_type="rule_application")
    assert answer["abstention"] is True
    assert answer["abstention_reason"] == "empty_query"
    assert answer["citations"] == []


def test_whitespace_only_query_returns_abstention():
    disp = _make_dispatcher()
    answer = disp.run("   \n  \t ", query_type="multi_hop")
    assert answer["abstention"] is True
    assert answer["abstention_reason"] == "empty_query"


def test_handler_exception_returns_dispatch_pipeline_error():
    bad = MagicMock()
    bad.run.side_effect = RuntimeError("inner crash")
    disp = _make_dispatcher(overrides={"rule_application": bad})
    answer = disp.run("س", query_type="rule_application")
    assert answer["abstention"] is True
    assert answer["abstention_reason"] == "dispatch_pipeline_error"
    assert answer["_telemetry"]["dispatched_handler"] == "rule_application"
    assert "inner crash" in answer["_telemetry"]["error"]


def test_handler_returning_non_dict_is_caught():
    bad = MagicMock()
    bad.run.return_value = "not a dict"
    disp = _make_dispatcher(overrides={"rule_application": bad})
    answer = disp.run("س", query_type="rule_application")
    assert answer["abstention"] is True
    assert answer["abstention_reason"] == "dispatch_bad_answer_shape"


def test_inner_abstention_propagates():
    handler = _stub_handler(
        abstention=True,
        abstention_reason="no_verified_articles",
        citations=[],
    )
    disp = _make_dispatcher(overrides={"unanswerable": handler})
    answer = disp.run("س", query_type="unanswerable")
    assert answer["abstention"] is True
    assert answer["abstention_reason"] == "no_verified_articles"
    assert answer["_telemetry"]["dispatched_handler"] == "unanswerable"


# ---------------------------------------------------------------------------
# End-to-end: dispatcher answers feed _answer_to_result unchanged
# ---------------------------------------------------------------------------


def test_dispatched_answer_consumed_by_answer_to_result():
    handler = _stub_handler()
    disp = _make_dispatcher(overrides={"rule_application": handler})
    answer = disp.run("ما هي شروط الزواج؟", query_type="rule_application")

    question = {
        "id":               "q_test",
        "query":            "ما هي شروط الزواج؟",
        "query_type":       "rule_application",
        "legal_category":   "family_law",
        "difficulty":       "medium",
        "language":         "ar",
        "split":            "test",
        "temporal_note":    None,
        "gold_doc_ids":     ["84-11_1984-06-09"],
        "gold_article_ids": ["84-11_1984-06-09#art_5"],
        "gold_citations":   [{"doc_id": "84-11_1984-06-09", "article_ref": "5"}],
        "gold_abstain":     False,
        "gold_answer":      "...",
    }
    result = _answer_to_result(question, answer)
    assert result["pred_doc_ids"] == ["84-11_1984-06-09"]
    assert result["pred_article_ids"] == ["84-11_1984-06-09#art_5"]
    assert result["predicted_abstain"] is False
    assert result["query_type"] == "rule_application"


def test_abstention_envelope_consumed_by_answer_to_result():
    bad = MagicMock()
    bad.run.side_effect = RuntimeError("crash")
    disp = _make_dispatcher(overrides={"rule_application": bad})
    answer = disp.run("سؤال", query_type="rule_application")

    question = {
        "id":               "q_test",
        "query":            "سؤال",
        "query_type":       "rule_application",
        "gold_doc_ids":     [],
        "gold_article_ids": [],
        "gold_citations":   [],
        "gold_abstain":     False,
    }
    result = _answer_to_result(question, answer)
    assert result["predicted_abstain"] is True
    assert result["pred_doc_ids"] == []
    assert result["pred_article_ids"] == []


# ---------------------------------------------------------------------------
# Phase E — per-handler recursion_coverage_min override
# ---------------------------------------------------------------------------


def _capture_kwargs(captured: dict, key: str):
    """Build a fake handler factory that records its kwargs under ``key``."""
    def _factory(**kwargs):
        captured[key] = kwargs
        return _stub_handler()
    return _factory


def test_recursion_coverage_min_override_forwarded_to_mh_ra_only(monkeypatch):
    """Phase E: when recursion is on AND an override map is supplied,
    MH/RA receive the overridden ``recursion_coverage_min`` while TF/CD
    keep their constructor default (we don't pass the kwarg)."""
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_rule_application_handler",
        _capture_kwargs(captured, "rule_application"),
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler",
        _capture_kwargs(captured, "multi_hop"),
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_temporal_factual_handler",
        _capture_kwargs(captured, "temporal_factual"),
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_conceptual_definitional_handler",
        _capture_kwargs(captured, "conceptual_definitional"),
    )

    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=object(),  # avoid kg_loader path
        enable_recursion=True,
        recursion_max_depth=3,
        recursion_coverage_min_overrides={"multi_hop": 4, "rule_application": 4},
    )
    disp.run("س1", query_type="rule_application")
    disp.run("س2", query_type="multi_hop")
    disp.run("س3", query_type="temporal_factual")
    disp.run("س4", query_type="conceptual_definitional")

    assert captured["rule_application"]["recursion_coverage_min"] == 4
    assert captured["multi_hop"]["recursion_coverage_min"] == 4
    assert "recursion_coverage_min" not in captured["temporal_factual"]
    assert "recursion_coverage_min" not in captured["conceptual_definitional"]
    # enable_recursion is still forwarded to all four handlers.
    for key in ("rule_application", "multi_hop", "temporal_factual",
                "conceptual_definitional"):
        assert captured[key]["enable_recursion"] is True


def test_recursion_coverage_min_override_ignored_when_recursion_off(monkeypatch):
    """If enable_recursion=False the override map is silently dropped —
    no recursion kwargs reach the handlers."""
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler",
        _capture_kwargs(captured, "multi_hop"),
    )
    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        enable_recursion=False,
        recursion_coverage_min_overrides={"multi_hop": 4},
    )
    disp.run("س", query_type="multi_hop")
    # When recursion is off, NO phase-D kwargs flow through — handler
    # keeps its own DEFAULT_COVERAGE_MIN.
    assert "recursion_coverage_min" not in captured["multi_hop"]
    assert "enable_recursion" not in captured["multi_hop"]


def test_e5_disambiguator_forwarded_to_mh_when_enabled(monkeypatch):
    """Phase E.2: when AKN_E5_KG_TOPOLOGY=1 the dispatcher builds the
    disambiguator on first MH dispatch (lazy KG load) and forwards it
    via ``kg_topology_disambiguator_fn``."""
    monkeypatch.setenv("AKN_E5_KG_TOPOLOGY", "1")
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler",
        _capture_kwargs(captured, "multi_hop"),
    )
    # Stub make_kg_topology_disambiguator so we don't traverse the real
    # rdflib path; assert it was called with the dispatcher's KG.
    fake_disamb = MagicMock(name="disamb")
    captured_kg = {}
    def _fake_make_disamb(sparql_fn, *, resolve_uri):
        captured_kg["sparql_fn"] = sparql_fn
        captured_kg["resolve_uri"] = resolve_uri
        return fake_disamb
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.make_kg_topology_disambiguator",
        _fake_make_disamb,
    )

    kg_sentinel = MagicMock(name="kg_graph")
    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=kg_sentinel,
    )
    disp.run("س", query_type="multi_hop")
    # Disambiguator was built once and forwarded.
    assert captured["multi_hop"]["kg_topology_disambiguator_fn"] is fake_disamb
    assert captured_kg["sparql_fn"] is not None  # built from kg
    assert callable(captured_kg["resolve_uri"])


def test_e5_disambiguator_not_built_when_disabled(monkeypatch):
    """Without AKN_E5_KG_TOPOLOGY, MH does NOT receive a disambiguator
    and the KG isn't lazily loaded just for MH."""
    monkeypatch.delenv("AKN_E5_KG_TOPOLOGY", raising=False)
    monkeypatch.delenv("AKN_ENHANCERS", raising=False)
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler",
        _capture_kwargs(captured, "multi_hop"),
    )
    loader = MagicMock()  # would error if called
    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=None,
        kg_loader=loader,
    )
    disp.run("س", query_type="multi_hop")
    assert "kg_topology_disambiguator_fn" not in captured["multi_hop"]
    loader.assert_not_called()


def test_e5_disambiguator_cached_across_calls(monkeypatch):
    """Disambiguator is built once even when MH is dispatched multiple
    times — the KG load is amortised."""
    monkeypatch.setenv("AKN_E5_KG_TOPOLOGY", "1")
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler",
        _capture_kwargs(captured, "multi_hop"),
    )
    build_call_count = [0]
    def _fake_make_disamb(_sparql_fn, *, resolve_uri):
        build_call_count[0] += 1
        return MagicMock()
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.make_kg_topology_disambiguator",
        _fake_make_disamb,
    )
    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=MagicMock(),
    )
    # Multiple MH dispatches — handler is built once (existing lazy
    # behaviour), and the disambiguator is also built only once.
    disp.run("س1", query_type="multi_hop")
    disp.run("س2", query_type="multi_hop")
    disp.run("س3", query_type="multi_hop")
    assert build_call_count[0] == 1


def test_e6_concept_kg_channel_forwarded_to_mh_and_ra(monkeypatch):
    """Phase E.3: when AKN_E6_CONCEPT_KG=1 the dispatcher builds the
    channel on first MH/RA dispatch and forwards it via
    ``concept_kg_channel_fn``. EA/LC/UA/CD/TF/layman do NOT receive
    the channel (Fix-TF already owns CD/TF's KG-first path)."""
    monkeypatch.setenv("AKN_E6_CONCEPT_KG", "1")
    captured: dict[str, dict] = {}
    for key in ("multi_hop", "rule_application", "exact_article",
                "long_context", "layman", "unanswerable",
                "temporal_factual", "conceptual_definitional"):
        monkeypatch.setattr(
            f"akn_rlm.rlm.dispatcher.build_{key}_handler",
            _capture_kwargs(captured, key),
        )
    fake_channel = MagicMock(name="concept_kg_channel")

    def _fake_make_channel(sparql_fn):
        return fake_channel
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.make_concept_kg_channel",
        _fake_make_channel,
    )

    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=MagicMock(),
    )
    disp.run("س1", query_type="multi_hop")
    disp.run("س2", query_type="rule_application")
    disp.run("س3", query_type="exact_article")
    disp.run("س4", query_type="temporal_factual")
    disp.run("س5", query_type="conceptual_definitional")

    assert captured["multi_hop"]["concept_kg_channel_fn"] is fake_channel
    assert captured["rule_application"]["concept_kg_channel_fn"] is fake_channel
    # EA/TF/CD/etc do NOT receive the channel kwarg.
    for k in ("exact_article", "temporal_factual", "conceptual_definitional"):
        assert "concept_kg_channel_fn" not in captured[k]


def test_e6_concept_kg_channel_not_built_when_disabled(monkeypatch):
    """Without AKN_E6_CONCEPT_KG, MH/RA do NOT receive a channel and
    the KG isn't loaded just for them."""
    monkeypatch.delenv("AKN_E6_CONCEPT_KG", raising=False)
    monkeypatch.delenv("AKN_ENHANCERS", raising=False)
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_multi_hop_handler",
        _capture_kwargs(captured, "multi_hop"),
    )
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_rule_application_handler",
        _capture_kwargs(captured, "rule_application"),
    )
    loader = MagicMock()  # would error if called
    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        kg=None,
        kg_loader=loader,
    )
    disp.run("س1", query_type="multi_hop")
    disp.run("س2", query_type="rule_application")
    assert "concept_kg_channel_fn" not in captured["multi_hop"]
    assert "concept_kg_channel_fn" not in captured["rule_application"]
    loader.assert_not_called()


def test_recursion_coverage_min_override_default_none_passes_no_kwarg(monkeypatch):
    """When no override map is set (or set to None), recursion is still
    forwarded but ``recursion_coverage_min`` is NOT injected — handler
    defaults apply."""
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        "akn_rlm.rlm.dispatcher.build_rule_application_handler",
        _capture_kwargs(captured, "rule_application"),
    )
    disp = RLMDispatcher(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        enable_recursion=True,
        recursion_max_depth=3,
        recursion_coverage_min_overrides=None,
    )
    disp.run("س", query_type="rule_application")
    assert captured["rule_application"]["enable_recursion"] is True
    assert "recursion_coverage_min" not in captured["rule_application"]


# ---------------------------------------------------------------------------
# Cross-stratum sweep: every query_type dispatches end-to-end
# ---------------------------------------------------------------------------


def test_stratified_dispatch_runs_every_type_end_to_end():
    """Mirrors the R7 gate ('all --stratified 2 runs end-to-end') with
    fully mocked handlers so the test runs in milliseconds. Each
    dispatched answer must be both a valid baseline-shaped dict AND
    consumable by _answer_to_result."""
    handlers = {key: _stub_handler(baseline=f"rlm_{key}")
                for key in set(TYPE_TO_HANDLER.values())}
    disp = _make_dispatcher(overrides=handlers)

    types = [
        "rule_application", "exact_article", "multi_hop", "unanswerable",
        "layman", "long_context", "conceptual_definitional", "temporal_factual",
    ]
    for qt in types:
        question = {
            "id":               f"q_{qt}",
            "query":            f"سؤال {qt}",
            "query_type":       qt,
            "gold_doc_ids":     ["84-11_1984-06-09"],
            "gold_article_ids": ["84-11_1984-06-09#art_5"],
            "gold_citations":   [{"doc_id": "84-11_1984-06-09", "article_ref": "5"}],
            "gold_abstain":     False,
        }
        answer = disp.run(question["query"], query_type=qt)
        assert answer["_telemetry"]["dispatched_handler"] == TYPE_TO_HANDLER[qt]
        result = _answer_to_result(question, answer)
        assert result["query_type"] == qt
        assert result["predicted_abstain"] is False
