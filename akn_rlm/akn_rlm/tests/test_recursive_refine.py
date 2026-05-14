"""Unit tests for the Phase D RecursiveRetriever helper."""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.rlm.recursive_refine import (
    RecursionStep,
    RecursiveRetriever,
    _merge_into,
    _normalise_question,
    _should_recurse,
    _top_confidence,
    call_gap_probe,
)


def _cit(doc: str, ref: str, conf: float = 0.7, text: str = "نص") -> dict:
    return {
        "doc_id": doc,
        "article_ref": ref,
        "supporting_span": text,
        "text": text,
        "confidence": conf,
    }


def _stub_retrieve(*hits: tuple[str, str, float]):
    """Return a retrieve_verify_fn that emits citations for the given (doc, ref, conf) triples."""
    def _fn(_q):
        return {(d, r): _cit(d, r, c) for d, r, c in hits}
    return MagicMock(side_effect=_fn)


def _no_recurse_probe(*_a, **_kw):
    return False, ""


def _force_recurse_probe(gap: str = "ما هو السؤال البديل؟"):
    def _fn(_pool, _q, _cits, **_kw):
        return True, gap
    return _fn


# ---------------------------------------------------------------------------
# _merge_into
# ---------------------------------------------------------------------------


def test_merge_into_adds_new_keys():
    base = {("d1", "1"): _cit("d1", "1", 0.7)}
    incoming = {("d1", "2"): _cit("d1", "2", 0.6)}
    merged, n_new = _merge_into(base, incoming)
    assert n_new == 1
    assert set(merged.keys()) == {("d1", "1"), ("d1", "2")}


def test_merge_into_keeps_higher_confidence():
    base = {("d1", "1"): _cit("d1", "1", 0.5)}
    incoming = {("d1", "1"): _cit("d1", "1", 0.9)}
    merged, n_new = _merge_into(base, incoming)
    assert n_new == 0
    assert merged[("d1", "1")]["confidence"] == 0.9


def test_merge_into_preserves_lower_when_incoming_weaker():
    base = {("d1", "1"): _cit("d1", "1", 0.9)}
    incoming = {("d1", "1"): _cit("d1", "1", 0.2)}
    merged, n_new = _merge_into(base, incoming)
    assert n_new == 0
    assert merged[("d1", "1")]["confidence"] == 0.9


def test_merge_into_preserves_argumentation_when_incoming_lacks_it():
    base = {("d1", "1"): {**_cit("d1", "1", 0.5), "argumentation": {"claim": "x"}}}
    incoming = {("d1", "1"): _cit("d1", "1", 0.9)}
    merged, _ = _merge_into(base, incoming)
    assert merged[("d1", "1")]["argumentation"] == {"claim": "x"}


# ---------------------------------------------------------------------------
# _top_confidence + _normalise_question
# ---------------------------------------------------------------------------


def test_top_confidence_empty():
    assert _top_confidence([]) == 0.0


def test_top_confidence_picks_max():
    assert _top_confidence([_cit("d1", "1", 0.3), _cit("d1", "2", 0.8)]) == 0.8


def test_normalise_question_strips_and_lowercases():
    a = _normalise_question("  Hello World  ")
    b = _normalise_question("hello world")
    assert a == b


# ---------------------------------------------------------------------------
# _should_recurse decision matrix
# ---------------------------------------------------------------------------


def test_should_recurse_strong_skip():
    cits = [_cit("d1", "1", 0.95), _cit("d1", "2", 0.92)]
    recurse, gap, label, calls = _should_recurse(
        "q", cits, llm_pool=None,
        coverage_min=2, confidence_weak=0.55, confidence_strong=0.9,
        probe_fn=_no_recurse_probe, probe_model="x",
    )
    assert recurse is False
    assert label == "strong_skip"
    assert calls == 0


def test_should_recurse_thin_force_yes():
    cits = [_cit("d1", "1", 0.6)]   # only 1 < coverage_min=2
    recurse, gap, label, calls = _should_recurse(
        "q", cits, llm_pool=None,
        coverage_min=2, confidence_weak=0.55, confidence_strong=0.9,
        probe_fn=_force_recurse_probe("gap?"), probe_model="x",
    )
    assert recurse is True
    assert gap == "gap?"
    assert label == "thin_force_yes"
    assert calls == 1


def test_should_recurse_weak_probe_no():
    cits = [_cit("d1", "1", 0.6), _cit("d1", "2", 0.6)]
    recurse, gap, label, calls = _should_recurse(
        "q", cits, llm_pool=None,
        coverage_min=2, confidence_weak=0.55, confidence_strong=0.9,
        probe_fn=_no_recurse_probe, probe_model="x",
    )
    assert recurse is False
    assert label == "weak_probe_no"
    assert calls == 1


def test_should_recurse_probe_exception_skips():
    def _bad_probe(*_a, **_kw):
        raise RuntimeError("boom")
    cits = [_cit("d1", "1", 0.6)]
    recurse, gap, label, calls = _should_recurse(
        "q", cits, llm_pool=None,
        coverage_min=2, confidence_weak=0.55, confidence_strong=0.9,
        probe_fn=_bad_probe, probe_model="x",
    )
    assert recurse is False
    assert label.endswith("_error")
    assert calls == 1


# ---------------------------------------------------------------------------
# RecursiveRetriever
# ---------------------------------------------------------------------------


def test_recursive_retriever_no_recurse_when_strong():
    fn = _stub_retrieve(("d1", "1", 0.95), ("d1", "2", 0.95))
    r = RecursiveRetriever(
        llm_pool=None, retrieve_verify_fn=fn,
        max_depth=3,
        coverage_min=2,
        confidence_weak=0.55,
        confidence_strong=0.9,
        probe_fn=_no_recurse_probe,
    )
    acc, steps, probe_calls = r.run("Q")
    assert len(acc) == 2
    # Two steps: depth-1 + skip marker recording strong_skip.
    assert len(steps) == 2
    assert steps[0].depth == 1
    assert steps[0].gap_decision == "depth_1"
    assert steps[1].gap_decision == "strong_skip"
    assert probe_calls == 0
    assert fn.call_count == 1


def test_recursive_retriever_recurses_and_merges_additively():
    # Depth-1 returns one weak citation; gap pass adds a second.
    call_log: list[str] = []

    def _retrieve(q: str):
        call_log.append(q)
        if q == "Q":
            return {("d1", "1"): _cit("d1", "1", 0.4)}
        return {("d1", "2"): _cit("d1", "2", 0.7)}

    r = RecursiveRetriever(
        llm_pool=None, retrieve_verify_fn=_retrieve,
        max_depth=3,
        coverage_min=2,
        confidence_weak=0.55,
        confidence_strong=0.9,
        probe_fn=_force_recurse_probe("gap-1"),
    )
    acc, steps, probe_calls = r.run("Q")
    assert set(acc.keys()) == {("d1", "1"), ("d1", "2")}, "Both keys merged additively"
    assert probe_calls >= 1
    assert call_log[0] == "Q"
    assert call_log[1] == "gap-1"
    # Step 1 is depth_1; at least one later step added new citations.
    assert steps[0].depth == 1
    productive = [s for s in steps if s.new_citations > 0]
    assert any(s.depth > 1 for s in productive), \
        f"Expected at least one productive recursion step; got {steps}"


def test_recursive_retriever_respects_max_depth():
    """With max_depth=2 we run depth-1 and at most ONE extra retrieval."""
    call_log: list[str] = []

    def _retrieve(q: str):
        call_log.append(q)
        return {("d1", str(len(call_log))): _cit("d1", str(len(call_log)), 0.4)}

    r = RecursiveRetriever(
        llm_pool=None, retrieve_verify_fn=_retrieve,
        max_depth=2,
        coverage_min=99,  # always thin → always probes
        probe_fn=_force_recurse_probe("gap-x"),
    )
    acc, steps, _ = r.run("Q")
    assert len(call_log) == 2  # depth-1 and exactly ONE gap pass
    # Two recursion steps: depth_1 + the depth_2 pass.
    assert max(s.depth for s in steps) == 2


def test_recursive_retriever_dedups_repeated_gap_question():
    """If the probe keeps returning the same gap question, we stop."""
    call_log: list[str] = []

    def _retrieve(q: str):
        call_log.append(q)
        return {}

    r = RecursiveRetriever(
        llm_pool=None, retrieve_verify_fn=_retrieve,
        max_depth=4,
        coverage_min=99,
        probe_fn=_force_recurse_probe("the-same-gap"),
    )
    acc, steps, _ = r.run("Q")
    # depth-1 (Q) + ONE recurse with "the-same-gap" then dedup-stop.
    assert call_log == ["Q", "the-same-gap"]
    assert any(s.gap_decision.endswith("_duplicate") for s in steps)


def test_recursive_retriever_seed_accumulator_skips_depth_1_call():
    """Caller can supply a pre-built accumulator (multi_hop pattern)."""
    seed = {("d1", "1"): _cit("d1", "1", 0.4)}
    fn = _stub_retrieve(("d1", "2", 0.7))

    r = RecursiveRetriever(
        llm_pool=None, retrieve_verify_fn=fn,
        max_depth=3,
        coverage_min=99,    # force probe
        probe_fn=_force_recurse_probe("gap"),
        seed_accumulator=seed,
    )
    acc, steps, _ = r.run("Q")
    # retrieve_verify_fn must NOT have been called for depth-1.
    assert fn.call_count == 1, "Depth-1 used seed; depth-2 fired the gap retrieval"
    assert ("d1", "1") in acc and ("d1", "2") in acc


def test_recursive_retriever_handles_retrieve_exception():
    def _bad(_q):
        raise RuntimeError("nope")
    r = RecursiveRetriever(
        llm_pool=None, retrieve_verify_fn=_bad,
        max_depth=2, probe_fn=_no_recurse_probe,
    )
    acc, steps, _ = r.run("Q")
    assert acc == {}
    assert steps[0].new_citations == 0


def test_recursion_step_to_dict_has_all_keys():
    s = RecursionStep(depth=2, sub_question="g", new_citations=1)
    d = s.to_dict()
    assert {"depth", "sub_question", "new_citations", "gap_decision"} <= set(d.keys())


# ---------------------------------------------------------------------------
# call_gap_probe — JSON parse paths
# ---------------------------------------------------------------------------


class _StubPool:
    def __init__(self, response: str):
        self._response = response

    def call(self, prompt, **kwargs):
        return self._response


def test_call_gap_probe_sufficient_returns_no_recurse():
    pool = _StubPool('{"sufficient": true}')
    recurse, gap = call_gap_probe(pool, "Q", [_cit("d1", "1")])
    assert recurse is False
    assert gap == ""


def test_call_gap_probe_with_gap_returns_recurse():
    pool = _StubPool('{"sufficient": false, "gap_question": "ما الحكم؟"}')
    recurse, gap = call_gap_probe(pool, "Q", [_cit("d1", "1")])
    assert recurse is True
    assert gap == "ما الحكم؟"


def test_call_gap_probe_missing_gap_returns_no_recurse():
    pool = _StubPool('{"sufficient": false}')
    recurse, gap = call_gap_probe(pool, "Q", [_cit("d1", "1")])
    assert recurse is False
    assert gap == ""


def test_call_gap_probe_garbage_returns_no_recurse():
    pool = _StubPool("not json at all")
    recurse, gap = call_gap_probe(pool, "Q", [_cit("d1", "1")])
    assert recurse is False
    assert gap == ""


def test_call_gap_probe_pool_exception_returns_no_recurse():
    class _BadPool:
        def call(self, *_a, **_kw):
            raise RuntimeError("boom")
    recurse, gap = call_gap_probe(_BadPool(), "Q", [_cit("d1", "1")])
    assert recurse is False
    assert gap == ""


def test_call_gap_probe_empty_query_returns_no_recurse():
    recurse, gap = call_gap_probe(_StubPool('{"sufficient": false, "gap_question": "x"}'), "", [])
    assert recurse is False
    assert gap == ""
