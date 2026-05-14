"""Unit tests for the Phase D corrective retry helper."""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.gates.base import GateResult
from akn_rlm.rlm.corrective_retry import (
    CorrectiveRetryTrace,
    build_corrective_question,
    maybe_corrective_retry,
)


def _cit(doc: str, ref: str, text: str = "نص المادة") -> dict:
    return {
        "doc_id": doc, "article_ref": ref,
        "text": text, "supporting_span": text,
        "confidence": 0.7,
    }


def _stub_summarizer(summary: str | None):
    """Returns a callable mimicking call_summarizer's signature."""
    fn = MagicMock(return_value={"summary": summary, "key_articles": [], "caveats": None})
    return fn


def _gate(passed: bool, score: float = 0.5, claims: list[str] | None = None) -> GateResult:
    details = []
    if not passed and claims:
        details = [{"claim": c, "best_cit": "x", "best_score": 0.2} for c in claims]
    return GateResult(passed=passed, score=score, details=details)


# ---------------------------------------------------------------------------
# build_corrective_question
# ---------------------------------------------------------------------------


def test_build_corrective_question_includes_claims_and_original():
    out = build_corrective_question("ما الحكم؟", ["claim1", "claim2"])
    assert "ما الحكم؟" in out
    assert "claim1" in out
    assert "claim2" in out


def test_build_corrective_question_caps_long_lists():
    out = build_corrective_question("Q", [f"c{i}" for i in range(20)], max_claims=3)
    # First three present, c10 not present.
    assert "c0" in out and "c1" in out and "c2" in out
    assert "c10" not in out


def test_build_corrective_question_empty_claims_uses_fallback_text():
    out = build_corrective_question("Q", [])
    # The fallback line is in Arabic; check the structural marker remains.
    assert "Q" in out


# ---------------------------------------------------------------------------
# maybe_corrective_retry
# ---------------------------------------------------------------------------


def test_disabled_returns_unchanged():
    summarizer = _stub_summarizer("new")
    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[_cit("d", "1")], original_question="Q",
        summarizer_fn=summarizer, llm_pool=None, sub_model="m",
        enabled=False, gate_fn=lambda *a, **kw: _gate(False, claims=["c"]),
    )
    assert answer == "orig"
    assert trace.fired is False
    assert summarizer.call_count == 0


def test_no_citations_returns_unchanged():
    summarizer = _stub_summarizer("new")
    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[], original_question="Q",
        summarizer_fn=summarizer, llm_pool=None, sub_model="m",
        enabled=True, gate_fn=lambda *a, **kw: _gate(False, claims=["c"]),
    )
    assert answer == "orig"
    assert trace.fired is False
    assert summarizer.call_count == 0


def test_passing_gate_returns_unchanged():
    summarizer = _stub_summarizer("new")
    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[_cit("d", "1")], original_question="Q",
        summarizer_fn=summarizer, llm_pool=None, sub_model="m",
        enabled=True, gate_fn=lambda *a, **kw: _gate(True, score=0.9),
    )
    assert answer == "orig"
    assert trace.fired is False
    assert trace.pre_passed is True
    assert summarizer.call_count == 0


def test_failing_gate_triggers_one_retry_with_feedback():
    summarizer = _stub_summarizer("regenerated answer")
    gate_calls: list[tuple[str, ...]] = []

    def _gate_fn(text, citations, **kw):
        if not gate_calls:
            gate_calls.append(("pre",))
            return _gate(False, score=0.3, claims=["bad claim"])
        gate_calls.append(("post",))
        return _gate(True, score=0.9)

    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[_cit("d", "1")], original_question="Q",
        summarizer_fn=summarizer, llm_pool=None, sub_model="m",
        enabled=True, gate_fn=_gate_fn,
    )
    assert answer == "regenerated answer"
    assert trace.fired is True
    assert trace.retry_succeeded is True
    assert trace.pre_passed is False
    assert trace.post_passed is True
    assert summarizer.call_count == 1
    # Probe the feedback question that the summariser was called with.
    feedback_q = summarizer.call_args[0][1]
    assert "bad claim" in feedback_q
    assert "Q" in feedback_q


def test_empty_retry_summary_falls_back_to_template():
    summarizer = _stub_summarizer(None)   # null summary
    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[_cit("d", "1")], original_question="Q",
        summarizer_fn=summarizer, llm_pool=None, sub_model="m",
        enabled=True, template_fallback="TEMPLATE",
        gate_fn=lambda *a, **kw: _gate(False, claims=["c"]),
    )
    assert answer == "TEMPLATE"
    assert trace.fired is True
    assert trace.retry_succeeded is False


def test_summariser_exception_preserves_original():
    bad_summarizer = MagicMock(side_effect=RuntimeError("boom"))
    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[_cit("d", "1")], original_question="Q",
        summarizer_fn=bad_summarizer, llm_pool=None, sub_model="m",
        enabled=True, gate_fn=lambda *a, **kw: _gate(False, claims=["c"]),
    )
    assert answer == "orig"
    assert trace.error.startswith("summarizer_error")
    assert trace.fired is False


def test_pre_gate_exception_preserves_original():
    summarizer = _stub_summarizer("new")

    def _bad_gate(*_a, **_kw):
        raise RuntimeError("nli down")

    answer, trace = maybe_corrective_retry(
        answer_text="orig", citations=[_cit("d", "1")], original_question="Q",
        summarizer_fn=summarizer, llm_pool=None, sub_model="m",
        enabled=True, gate_fn=_bad_gate,
    )
    assert answer == "orig"
    assert trace.error.startswith("pre_gate_error")
    assert trace.fired is False
    assert summarizer.call_count == 0


def test_trace_to_dict_has_canonical_keys():
    t = CorrectiveRetryTrace(fired=True, pre_passed=False, post_passed=True)
    d = t.to_dict()
    assert {"fired", "pre_passed", "post_passed", "pre_gate_score", "post_gate_score",
            "unsupported_claims", "retry_succeeded", "sub_call_count", "error"} <= set(d.keys())
