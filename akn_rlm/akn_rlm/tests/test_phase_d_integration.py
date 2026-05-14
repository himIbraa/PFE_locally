"""Phase D end-to-end integration tests on the RA handler.

These verify the wired-in toggles produce the expected telemetry shape
and that recursion is *additive* (never reduces the citation set).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.gates.base import GateResult
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.rule_application import RuleApplicationHandler
from akn_rlm.rlm.routing import RouteResult


def _bm25(doc: str, ref: str, score: float = 5.0) -> BM25Hit:
    return BM25Hit(chunk_id=f"{doc}#art_{ref}", doc_id=doc, article_ref=ref,
                   score=score, text="نص")


def _dense(doc: str, ref: str, score: float = 0.5) -> DenseHit:
    return DenseHit(chunk_id=f"{doc}#art_{ref}", doc_id=doc, article_ref=ref,
                    score=score, text="نص")


def _stub_router(doc_ids: list[str]) -> MagicMock:
    router = MagicMock()
    router.route.return_value = RouteResult(
        doc_ids=list(doc_ids),
        scores={d: 1.0 for d in doc_ids},
        sources={d: ["alias"] for d in doc_ids},
        confidence=1.0,
    )
    return router


def _stub_bm25(hits: list[BM25Hit]) -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = hits
    return idx


def _stub_dense(hits: list[DenseHit]) -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = hits
    return idx


def _stub_registry() -> MagicMock:
    reg = MagicMock()
    reg.get_doc.return_value = MagicMock(doc_title="قانون")
    return reg


def _verifier(default_conf: float = 0.7):
    def _fn(_pool, _q, art, _model):
        return {
            "relevant": True,
            "supporting_span": None,
            "contradicting_span": None,
            "confidence": default_conf,
        }
    return MagicMock(side_effect=_fn)


def _summarizer(text: str = "ملخص جيد"):
    return MagicMock(return_value={"summary": text, "key_articles": [], "caveats": None})


# ---------------------------------------------------------------------------
# Recursion ON
# ---------------------------------------------------------------------------


def test_ra_recursion_off_emits_no_recursion_trace_field_filled():
    handler = RuleApplicationHandler(
        bm25=_stub_bm25([_bm25("d1", "1")]),
        dense=_stub_dense([_dense("d1", "1")]),
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1"]),
        verifier_fn=_verifier(0.8),
        summarizer_fn=_summarizer(),
        enable_recursion=False,
    )
    answer = handler.run("Q")
    assert answer["abstention"] is False
    tel = answer["_telemetry"]
    # When recursion is off the trace is empty but the key still exists
    # so the predictions.jsonl shape stays uniform.
    assert tel["recursion_trace"] == []
    assert tel["recursion_depth_max"] == 1


def test_ra_recursion_on_strong_does_not_recurse():
    """High verifier confidence → strong_skip (no probe call)."""
    bad_probe = MagicMock(return_value=(False, ""))
    handler = RuleApplicationHandler(
        bm25=_stub_bm25([_bm25("d1", "1"), _bm25("d1", "2")]),
        dense=_stub_dense([_dense("d1", "1"), _dense("d1", "2")]),
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1"]),
        verifier_fn=_verifier(0.95),   # > confidence_strong=0.9
        summarizer_fn=_summarizer(),
        enable_recursion=True,
        recursion_max_depth=3,
    )
    answer = handler.run("Q")
    tel = answer["_telemetry"]
    # Trace recorded depth-1 + the strong_skip marker.
    assert any(s["gap_decision"] == "depth_1" for s in tel["recursion_trace"])
    assert any(s["gap_decision"] == "strong_skip" for s in tel["recursion_trace"])


def test_ra_recursion_on_thin_calls_probe_and_merges_additively():
    """Thin retrieval (1 verified) → probe fires → gap query merges new candidates."""
    # Depth-1 surfaces d1 only; depth-2 (gap question) surfaces d2 only.
    # Both BM25 and Dense are stubbed independently so the depth-1 vs
    # depth-2 toggle is purely query-driven.
    bm25 = MagicMock()
    dense = MagicMock()

    def _bm_search(q, k):
        if "الآخر" in q:
            return [_bm25("d2", "1")]
        return [_bm25("d1", "1")]

    def _dn_search(q, k):
        if "الآخر" in q:
            return [_dense("d2", "1")]
        return [_dense("d1", "1")]

    bm25.search.side_effect = _bm_search
    dense.search.side_effect = _dn_search

    probe_recurse = MagicMock(return_value=(True, "ما هو القانون الآخر؟"))
    handler = RuleApplicationHandler(
        bm25=bm25,
        dense=dense,
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1", "d2"]),
        verifier_fn=_verifier(0.6),    # mid-band, not strong
        summarizer_fn=_summarizer(),
        enable_recursion=True,
        recursion_max_depth=2,
        recursion_coverage_min=2,      # 1 verified < 2 → thin
        recursion_probe_fn=probe_recurse,
    )
    answer = handler.run("Q")
    tel = answer["_telemetry"]
    # Probe must have been called.
    assert probe_recurse.called
    # Recursion added at least one new citation at depth-2.
    new_step = next(s for s in tel["recursion_trace"] if s["depth"] == 2)
    assert new_step["new_citations"] >= 1
    assert tel["recursion_depth_max"] == 2
    # Final citations include both d1 and d2 (additive merge).
    cit_keys = {(c["doc_id"], c["article_ref"]) for c in answer["citations"]}
    assert ("d1", "1") in cit_keys
    assert ("d2", "1") in cit_keys


def test_ra_trajectory_lists_route_rank_summarize_steps():
    handler = RuleApplicationHandler(
        bm25=_stub_bm25([_bm25("d1", "1")]),
        dense=_stub_dense([_dense("d1", "1")]),
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1"]),
        verifier_fn=_verifier(0.8),
        summarizer_fn=_summarizer(),
    )
    answer = handler.run("Q")
    steps = [s.get("step") for s in answer["trajectory"]]
    assert "route" in steps
    assert "rank" in steps
    assert "summarize" in steps


# ---------------------------------------------------------------------------
# Corrective retry ON
# ---------------------------------------------------------------------------


def test_ra_corrective_retry_off_no_corrective_field():
    handler = RuleApplicationHandler(
        bm25=_stub_bm25([_bm25("d1", "1")]),
        dense=_stub_dense([_dense("d1", "1")]),
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1"]),
        verifier_fn=_verifier(0.8),
        summarizer_fn=_summarizer(),
        enable_corrective_retry=False,
    )
    answer = handler.run("Q")
    assert "corrective_retry" not in answer["_telemetry"]


def test_ra_corrective_retry_on_passing_gate_no_retry(monkeypatch):
    """Faithfulness passes → retry never fires; field exists with fired=False."""
    from akn_rlm.gates.base import GateResult as _GR
    from akn_rlm.rlm import corrective_retry as cr_mod

    monkeypatch.setattr(
        cr_mod, "run_gate",
        lambda *a, **kw: _GR(passed=True, score=0.9, details=[]),
    )

    summarizer = _summarizer()
    handler = RuleApplicationHandler(
        bm25=_stub_bm25([_bm25("d1", "1")]),
        dense=_stub_dense([_dense("d1", "1")]),
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1"]),
        verifier_fn=_verifier(0.8),
        summarizer_fn=summarizer,
        enable_corrective_retry=True,
    )
    answer = handler.run("Q")
    tel = answer["_telemetry"]
    assert "corrective_retry" in tel
    assert tel["corrective_retry"]["fired"] is False
    assert tel["retry_count"] == 0
    # Summariser called exactly once (no retry).
    assert summarizer.call_count == 1


def test_ra_corrective_retry_on_failing_gate_retries_once(monkeypatch):
    from akn_rlm.gates.base import GateResult as _GR
    from akn_rlm.rlm import corrective_retry as cr_mod

    state = {"calls": 0}

    def _gate(text, citations, **kw):
        state["calls"] += 1
        # Fail first, pass second.
        if state["calls"] == 1:
            return _GR(passed=False, score=0.3,
                       details=[{"claim": "غير مدعوم", "best_cit": "x", "best_score": 0.2}])
        return _GR(passed=True, score=0.9, details=[])

    monkeypatch.setattr(cr_mod, "run_gate", _gate)

    summarizer = MagicMock(side_effect=[
        {"summary": "first answer", "key_articles": [], "caveats": None},
        {"summary": "regenerated answer", "key_articles": [], "caveats": None},
    ])

    handler = RuleApplicationHandler(
        bm25=_stub_bm25([_bm25("d1", "1")]),
        dense=_stub_dense([_dense("d1", "1")]),
        registry=_stub_registry(),
        llm_pool=MagicMock(),
        router=_stub_router(["d1"]),
        verifier_fn=_verifier(0.8),
        summarizer_fn=summarizer,
        enable_corrective_retry=True,
    )
    answer = handler.run("Q")
    tel = answer["_telemetry"]
    assert tel["corrective_retry"]["fired"] is True
    assert tel["corrective_retry"]["retry_succeeded"] is True
    assert tel["retry_count"] == 1
    assert answer["answer_text"] == "regenerated answer"
    assert summarizer.call_count == 2


# ---------------------------------------------------------------------------
# Trajectory persistence through the runner
# ---------------------------------------------------------------------------


def test_answer_to_result_persists_trajectory():
    from akn_rlm.eval.runner import _answer_to_result
    answer = {
        "answer_text": "x",
        "abstention": False,
        "abstention_reason": None,
        "citations": [],
        "reasoning_chain": [],
        "trajectory": [{"step": "route", "depth": 0}, {"step": "summarize", "depth": 0}],
        "tokens_used": 0,
        "depth_max_reached": 1,
        "_telemetry": {"baseline": "rlm_dispatched"},
    }
    q = {"id": "q1", "query": "Q"}
    result = _answer_to_result(q, answer)
    assert result["trajectory"] == [
        {"step": "route", "depth": 0},
        {"step": "summarize", "depth": 0},
    ]
