"""Tests for the UI-facing endpoint :func:`akn_rlm.api.answer.answer_query`.

The dispatcher is mocked so these tests don't load corpus / indices /
LLM — they validate the response shape, trajectory humanisation, and
citation formatting that the front-end will rely on.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.api.answer import (
    AnswerResponse,
    Citation,
    TrajectoryStep,
    _format_references,
    _summarise_step,
    _to_citation,
    answer_query,
    reset_dispatcher,
)


# ---------------------------------------------------------------------------
# Fixtures: a mock dispatcher that returns a baseline-shaped answer
# ---------------------------------------------------------------------------


def _stub_dispatcher(payload: dict) -> MagicMock:
    m = MagicMock()
    m.run.return_value = payload
    return m


def _payload(
    *,
    abstention: bool = False,
    citations: list[dict] | None = None,
    trajectory: list[dict] | None = None,
    handler: str = "rule_application",
    query_type: str = "rule_application",
    sub_calls: int = 7,
    answer: str = "إجابة تجريبية",
    depth_max: int = 2,
    corrective_retry_fired: bool = False,
) -> dict:
    cits = citations if citations is not None else [
        {
            "doc_id":          "84-11_1984-06-09",
            "article_ref":     "54",
            "doc_title":       "قانون الأسرة",
            "supporting_span": "نص داعم",
            "text":            "نص المادة الكامل",
            "confidence":      0.85,
            "version_date":    "2005-02-27",
            "kg_source":       "kg",
            "argumentation": {
                "claim":   "ادعاء توضيحي",
                "ground":  "أساس داعم",
                "warrant": "تبرير",
            },
            "verifier_relevant": True,
        },
    ]
    trj = trajectory if trajectory is not None else [
        {"step": "route", "depth": 0,
         "routed_doc_ids": ["84-11_1984-06-09"]},
        {"step": "summarize", "depth": 0},
    ]
    telemetry = {
        "dispatched_handler":     handler,
        "dispatched_query_type":  query_type,
        "sub_call_count":         sub_calls,
        "recursion_depth_max":    depth_max,
    }
    if corrective_retry_fired:
        telemetry["corrective_retry"] = {"fired": True,
                                         "pre_passed": False,
                                         "post_passed": True}
    return {
        "answer_text":       answer,
        "abstention":        abstention,
        "abstention_reason": "no_verified_articles" if abstention else None,
        "citations":         [] if abstention else cits,
        "trajectory":        trj,
        "depth_max_reached": depth_max,
        "_telemetry":        telemetry,
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_answer_query_rejects_empty_query():
    with pytest.raises(ValueError):
        answer_query("")


def test_answer_query_rejects_whitespace_query():
    with pytest.raises(ValueError):
        answer_query("   \n  ")


def test_answer_query_uses_injected_dispatcher(monkeypatch):
    """Test contract: an explicit ``dispatcher`` kwarg short-circuits
    the lazy singleton."""
    reset_dispatcher()
    fake = _stub_dispatcher(_payload())
    response = answer_query("سؤال", dispatcher=fake)
    assert isinstance(response, AnswerResponse)
    # Singleton stayed empty (no real dispatcher was built).
    fake.run.assert_called_once_with("سؤال")


def test_answer_query_response_shape():
    fake = _stub_dispatcher(_payload())
    response = answer_query("ما هي شروط الزواج؟", dispatcher=fake)
    assert response.query == "ما هي شروط الزواج؟"
    assert response.handler_used == "rule_application"
    assert response.query_type_predicted == "rule_application"
    assert response.answer_text == "إجابة تجريبية"
    assert response.abstained is False
    assert response.abstention_reason is None
    assert response.sub_call_count == 7
    assert response.recursion_depth_max == 2
    assert response.corrective_retry_fired is False
    assert response.latency_s >= 0.0


def test_response_to_dict_is_json_serialisable():
    import json
    fake = _stub_dispatcher(_payload())
    response = answer_query("سؤال", dispatcher=fake)
    d = response.to_dict()
    # round-trip through json without TypeError
    json.dumps(d, ensure_ascii=False)


def test_citations_have_toulmin_argumentation_when_present():
    fake = _stub_dispatcher(_payload())
    response = answer_query("سؤال", dispatcher=fake)
    assert len(response.citations) == 1
    c = response.citations[0]
    assert c.doc_id == "84-11_1984-06-09"
    assert c.article_ref == "54"
    assert c.version_date == "2005-02-27"
    assert c.argumentation is not None
    assert c.argumentation["claim"] == "ادعاء توضيحي"


def test_empty_argumentation_dict_is_normalised_to_none():
    payload = _payload(citations=[{
        "doc_id": "d", "article_ref": "1",
        "doc_title": "t", "supporting_span": "s",
        "text": "x", "confidence": 0.5,
        "argumentation": {"claim": "", "ground": None, "warrant": None,
                          "rebuttal": None, "backing": None},
    }])
    fake = _stub_dispatcher(payload)
    response = answer_query("سؤال", dispatcher=fake)
    assert response.citations[0].argumentation is None


# ---------------------------------------------------------------------------
# References formatting
# ---------------------------------------------------------------------------


def test_references_formatted_with_version_date():
    cits = [
        Citation(doc_id="d1", article_ref="5", doc_title="قانون الأسرة",
                 supporting_span="", text="", confidence=0.9,
                 version_date="2005-02-27"),
    ]
    refs = _format_references(cits)
    assert refs == ["[1] قانون الأسرة، المادة 5 (نسخة 2005-02-27)"]


def test_references_omit_version_date_when_missing():
    cits = [
        Citation(doc_id="d1", article_ref="12", doc_title="القانون التجاري",
                 supporting_span="", text="", confidence=0.9),
    ]
    assert _format_references(cits) == ["[1] القانون التجاري، المادة 12"]


def test_references_numbered_starting_at_one():
    cits = [
        Citation(doc_id="d1", article_ref="1", doc_title="A",
                 supporting_span="", text="", confidence=0.9),
        Citation(doc_id="d2", article_ref="2", doc_title="B",
                 supporting_span="", text="", confidence=0.9),
        Citation(doc_id="d3", article_ref="3", doc_title="C",
                 supporting_span="", text="", confidence=0.9),
    ]
    refs = _format_references(cits)
    assert refs[0].startswith("[1] ")
    assert refs[1].startswith("[2] ")
    assert refs[2].startswith("[3] ")


def test_references_fallback_to_doc_id_when_no_title():
    cits = [
        Citation(doc_id="84-11_1984-06-09", article_ref="1", doc_title="",
                 supporting_span="", text="", confidence=0.9),
    ]
    # _format_references uses doc_title or doc_id.
    refs = _format_references(cits)
    assert "84-11_1984-06-09" in refs[0]


# ---------------------------------------------------------------------------
# Trajectory summarisation (Arabic explainability)
# ---------------------------------------------------------------------------


def test_trajectory_route_step_lists_doc_ids():
    step = _summarise_step({"step": "route", "depth": 0,
                            "routed_doc_ids": ["84-11_1984-06-09",
                                               "75-58_1975-09-26"]})
    assert isinstance(step, TrajectoryStep)
    assert step.step == "route"
    assert "84-11_1984-06-09" in step.summary
    assert "75-58_1975-09-26" in step.summary


def test_trajectory_recursion_step_includes_decision_and_sub_question():
    step = _summarise_step({
        "step": "recursion", "depth": 2,
        "sub_question": "ما هي شروط الخلع الجديدة في تعديل 2005؟",
        "gap_decision": "weak_probe_yes",
        "new_citations": 2,
    })
    # Mentions the Arabic decision label and the truncated sub-question.
    assert "weak" in step.summary.lower() or "متوسطة" in step.summary
    assert "شروط الخلع" in step.summary


def test_trajectory_extract_date_step_carries_target():
    step = _summarise_step({
        "step": "extract_date", "depth": 0,
        "dates": ["2005-12-31"], "target": "2005-12-31",
    })
    assert "2005-12-31" in step.summary


def test_trajectory_faithfulness_gate_pre_and_post():
    step = _summarise_step({
        "step": "faithfulness_gate", "depth": 0,
        "fired": True, "pre_passed": False, "post_passed": True,
    })
    assert "نجح" in step.summary
    assert "فشل" in step.summary


def test_trajectory_unknown_step_falls_back_to_raw_label():
    step = _summarise_step({"step": "unknown_step", "depth": 0})
    assert step.step == "unknown_step"
    # No crash — summary defaults to the raw step name.
    assert step.summary


def test_trajectory_detail_preserves_raw_payload():
    raw = {"step": "kg_chain", "depth": 1, "candidates": 5, "verified": 3}
    step = _summarise_step(raw)
    assert step.detail == {"candidates": 5, "verified": 3}


# ---------------------------------------------------------------------------
# Abstention path
# ---------------------------------------------------------------------------


def test_abstained_response_has_no_citations():
    fake = _stub_dispatcher(_payload(abstention=True))
    response = answer_query("سؤال غامض", dispatcher=fake)
    assert response.abstained is True
    assert response.abstention_reason == "no_verified_articles"
    assert response.citations == []
    assert response.references == []


def test_corrective_retry_flag_propagates():
    fake = _stub_dispatcher(_payload(corrective_retry_fired=True))
    response = answer_query("سؤال", dispatcher=fake)
    assert response.corrective_retry_fired is True


# ---------------------------------------------------------------------------
# _to_citation low-level
# ---------------------------------------------------------------------------


def test_to_citation_coerces_missing_fields_to_safe_defaults():
    raw = {"doc_id": "d", "article_ref": "1"}
    c = _to_citation(raw)
    assert c.doc_id == "d"
    assert c.article_ref == "1"
    assert c.doc_title == "d"  # falls back to doc_id
    assert c.confidence == 0.0
    assert c.version_date is None
    assert c.argumentation is None


# ---------------------------------------------------------------------------
# Fallback trajectory for handlers without an emitted trajectory
# ---------------------------------------------------------------------------


def test_empty_trajectory_falls_back_to_handler_summary_row():
    """exact_article / unanswerable / layman currently don't emit a
    trajectory list. The API must inject a single ``handler_summary``
    row so the UI's explainability panel is never empty."""
    payload = _payload(trajectory=[], handler="exact_article",
                       query_type="exact_article")
    fake = _stub_dispatcher(payload)
    response = answer_query("سؤال", dispatcher=fake)
    assert len(response.trajectory) == 1
    only = response.trajectory[0]
    assert only.step == "handler_summary"
    # The Arabic handler label is in the summary line.
    assert "مادة محددة" in only.summary


def test_multi_step_trajectory_is_preserved_in_order():
    """Handlers that DO emit a trajectory (RA / MH / TF / CD) should
    pass through every step in order, with each translated to a
    human-readable summary."""
    payload = _payload(
        trajectory=[
            {"step": "route", "depth": 0, "routed_doc_ids": ["d1"]},
            {"step": "decompose_sweep", "depth": 1,
             "sub_questions": 3, "verified": 5},
            {"step": "recursion", "depth": 2,
             "sub_question": "?", "gap_decision": "weak_probe_yes",
             "new_citations": 1},
            {"step": "adu_extract", "depth": 0, "extracts": 3},
            {"step": "summarize", "depth": 0},
            {"step": "faithfulness_gate", "depth": 0,
             "fired": False, "pre_passed": True, "post_passed": True},
        ],
        handler="multi_hop", query_type="multi_hop",
    )
    fake = _stub_dispatcher(payload)
    response = answer_query("سؤال", dispatcher=fake)
    steps = [s.step for s in response.trajectory]
    assert steps == ["route", "decompose_sweep", "recursion",
                     "adu_extract", "summarize", "faithfulness_gate"]
