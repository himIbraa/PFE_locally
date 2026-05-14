"""Unit tests for :mod:`akn_rlm.rlm.routing.doc_router`.

The router is exercised with both a real :class:`ArticleRegistry` (built
from a tiny in-memory article list, no XML parsing) and a mocked
:class:`BM25Index`, so these tests don't depend on any on-disk index.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.rlm.routing.doc_router import (
    DEFAULT_TOP_N,
    DocRouter,
    RouteResult,
    build_doc_router,
)


# ---------------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------------


@dataclass
class _StubArticle:
    """Minimal Article shape consumed by ArticleRegistry.build()."""
    doc_id: str
    article_ref: str
    eid: str
    filename_stem: str
    doc_title: str
    doc_date: str
    doc_type: str


def _registry_with_docs(*doc_ids: str) -> ArticleRegistry:
    """Build a registry with one stub article per requested doc_id."""
    articles = [
        _StubArticle(
            doc_id=did,
            article_ref="1",
            eid="art_1",
            filename_stem=did.split("_")[0],
            doc_title=f"doc {did}",
            doc_date="1900-01-01",
            doc_type="act",
        )
        for did in doc_ids
    ]
    reg = ArticleRegistry()
    reg.build(articles)
    return reg


def _bm25_returning(hits: list[BM25Hit]) -> MagicMock:
    bm25 = MagicMock()
    bm25.search.return_value = hits
    return bm25


def _hit(doc_id: str, ref: str, score: float, text: str = "") -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


# ---------------------------------------------------------------------------
# Contract / defaults
# ---------------------------------------------------------------------------


def test_defaults_match_handoff():
    """Top-N defaults to 3 — the gate target in HANDOFF.md §5 (R1)."""
    assert DEFAULT_TOP_N == 3


def test_route_returns_routeresult_with_required_fields():
    reg = _registry_with_docs("84-11_1984-06-09")
    bm25 = _bm25_returning([])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("ما هي شروط الزواج في قانون الأسرة؟")
    assert isinstance(out, RouteResult)
    assert isinstance(out.doc_ids, list)
    assert isinstance(out.scores, dict)
    assert isinstance(out.sources, dict)
    assert isinstance(out.confidence, float)


def test_top_n_passthrough_caps_returned_doc_ids():
    """Even with many BM25 docs the result must respect top_n."""
    reg = _registry_with_docs("aaa", "bbb", "ccc", "ddd", "eee")
    bm25 = _bm25_returning([
        _hit("aaa", "1", 5.0),
        _hit("bbb", "1", 4.0),
        _hit("ccc", "1", 3.0),
        _hit("ddd", "1", 2.0),
        _hit("eee", "1", 1.0),
    ])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("any query", top_n=2)
    assert len(out.doc_ids) == 2
    assert out.doc_ids == ["aaa", "bbb"]


# ---------------------------------------------------------------------------
# Channel 1 — alias scan
# ---------------------------------------------------------------------------


def test_arabic_alias_resolves_to_canonical_doc_id():
    """قانون الأسرة (Family Code) -> 84-11_1984-06-09 via static alias map."""
    reg = _registry_with_docs("84-11_1984-06-09")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("ما هي شروط الزواج في قانون الأسرة؟")
    assert "84-11_1984-06-09" in out.doc_ids
    assert "alias" in out.sources["84-11_1984-06-09"]


def test_english_multiword_alias_matches_substring():
    """'civil procedure' must beat 'civil' — longest-first ordering."""
    reg = _registry_with_docs("75-58_1975-09-26", "08-09_2008-02-25")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("Explain the civil procedure in court hearings.")
    assert out.doc_ids[0] == "08-09_2008-02-25"
    assert "alias" in out.sources["08-09_2008-02-25"]


def test_short_latin_abbreviation_requires_word_boundary():
    """'cpp' must NOT match inside 'applicants' or other words."""
    reg = _registry_with_docs("25-14_2025-08-03", "75-58_1975-09-26")
    router = build_doc_router(registry=reg, bm25=None)
    # 'cpp' is in the alias map for 25-14_2025-08-03; 'applicants' contains the
    # substring 'pp' but not 'cpp' as a whole word.
    out = router.route("All applicants must follow the rules.")
    assert "25-14_2025-08-03" not in out.doc_ids


def test_short_latin_abbreviation_matches_when_word_bounded():
    """A whitespace-separated 'cpp' token must trigger the alias channel."""
    reg = _registry_with_docs("25-14_2025-08-03")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("Reference: cpp art.5.")
    assert "25-14_2025-08-03" in out.doc_ids
    assert "alias" in out.sources["25-14_2025-08-03"]


def test_alias_only_routing_returns_high_confidence():
    """Alias-only hit -> confidence == 1.0."""
    reg = _registry_with_docs("84-11_1984-06-09")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("ما هي شروط الزواج في قانون الأسرة؟")
    assert out.confidence == 1.0


# ---------------------------------------------------------------------------
# Channel 2 — numeric law-id scan
# ---------------------------------------------------------------------------


def test_numeric_law_id_resolves_to_canonical_doc_id():
    """'84-11' inside a sentence must resolve via the registry alias map."""
    reg = _registry_with_docs("84-11_1984-06-09")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("Refer to law 84-11 for the answer.")
    assert "84-11_1984-06-09" in out.doc_ids


def test_numeric_law_id_does_not_match_inside_a_date():
    """A date like 1984-06-09 must NOT be parsed as law id 84-06."""
    # If 84-06 is not a registered alias, the safety net is registry resolution
    # returning None.  The stronger test: the doc id should not surface from
    # an in-text date when nothing else mentions it.
    reg = _registry_with_docs("90-11_1990-04-21")  # something unrelated
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("Decree dated 1984-06-09 was issued.")
    assert out.doc_ids == []


def test_numeric_law_id_handles_hyphenated_clusters():
    """'84-11_1984-06-09' as a literal string must still resolve to the canonical id."""
    reg = _registry_with_docs("84-11_1984-06-09")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("see 84-11")
    assert "84-11_1984-06-09" in out.doc_ids


# ---------------------------------------------------------------------------
# Channel 3 — BM25 aggregation
# ---------------------------------------------------------------------------


def test_bm25_only_routing_picks_top_doc_by_aggregate():
    """Two docs in BM25 hits — the one with higher cumulative score wins."""
    reg = _registry_with_docs("aaa", "bbb")
    bm25 = _bm25_returning([
        _hit("aaa", "1", 1.0),
        _hit("aaa", "2", 1.0),
        _hit("bbb", "1", 1.5),
        _hit("aaa", "3", 1.0),
    ])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("free-form query")
    # aaa cumulative = 3.0, bbb cumulative = 1.5 (after per-doc cap=5).
    assert out.doc_ids[0] == "aaa"
    assert "bm25" in out.sources["aaa"]


def test_bm25_per_doc_cap_prevents_one_law_from_crowding_out_another():
    """Many low-scoring chunks of doc A should not beat one strong chunk of B."""
    reg = _registry_with_docs("flood", "strong")
    flood_hits = [_hit("flood", str(i), 0.5) for i in range(10)]
    bm25 = _bm25_returning(flood_hits + [_hit("strong", "1", 5.0)])
    router = build_doc_router(registry=reg, bm25=bm25, bm25_per_doc_cap=2)
    out = router.route("query")
    # flood cumulative (cap=2) = 1.0; strong = 5.0.  Strong must come first.
    assert out.doc_ids[0] == "strong"


def test_bm25_only_routing_returns_low_confidence():
    """No alias hit -> confidence drops to 0.6."""
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([_hit("aaa", "1", 1.0)])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("free-form query without aliases")
    assert out.confidence == 0.6


def test_bm25_search_called_with_configured_pool_size():
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([])
    router = build_doc_router(registry=reg, bm25=bm25, bm25_pool=42)
    router.route("q")
    bm25.search.assert_called_once_with("q", k=42)


def test_bm25_failure_does_not_raise():
    """A flaky BM25 returning an exception -> router degrades gracefully."""
    reg = _registry_with_docs("aaa")
    bm25 = MagicMock()
    bm25.search.side_effect = RuntimeError("boom")
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("q")
    assert isinstance(out, RouteResult)


# ---------------------------------------------------------------------------
# Channel 4 — optional LLM tie-breaker
# ---------------------------------------------------------------------------


def test_llm_channel_off_by_default():
    """Without an llm_call hook the router does not call any LLM."""
    reg = _registry_with_docs("aaa", "bbb")
    bm25 = _bm25_returning([_hit("aaa", "1", 1.0), _hit("bbb", "1", 0.5)])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("q")
    assert "llm" not in out.sources.get(out.doc_ids[0], [])


def test_llm_channel_can_promote_lower_ranked_candidate():
    """When the LLM picks a doc_id, its fused score gets a bonus."""
    reg = _registry_with_docs("aaa", "bbb")
    # Without LLM, BM25 ranks aaa > bbb; LLM promotes bbb to the top.
    bm25 = _bm25_returning([
        _hit("aaa", "1", 1.0),
        _hit("bbb", "1", 0.5),
    ])

    def llm_pick(_q: str, candidates: list[str]) -> list[str]:
        return ["bbb"]  # explicit promotion

    router = build_doc_router(
        registry=reg, bm25=bm25, llm_call=llm_pick, llm_bonus=10.0,
    )
    out = router.route("q")
    assert out.doc_ids[0] == "bbb"
    assert "llm" in out.sources["bbb"]


def test_llm_channel_is_robust_to_unknown_doc_ids():
    """LLM returning a doc_id that wasn't in the candidate set is ignored."""
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([_hit("aaa", "1", 1.0)])

    def llm_pick(_q: str, _candidates: list[str]) -> list[str]:
        return ["does-not-exist"]

    router = build_doc_router(registry=reg, bm25=bm25, llm_call=llm_pick)
    out = router.route("q")
    assert out.doc_ids == ["aaa"]


def test_llm_channel_failure_does_not_break_routing():
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([_hit("aaa", "1", 1.0)])

    def llm_pick(_q: str, _c: list[str]) -> list[str]:
        raise RuntimeError("LLM unavailable")

    router = build_doc_router(registry=reg, bm25=bm25, llm_call=llm_pick)
    out = router.route("q")
    assert out.doc_ids == ["aaa"]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_alias_hit_outranks_bm25_only_hit():
    """A doc named explicitly in the query must come first even when BM25
    favours another doc."""
    reg = _registry_with_docs("84-11_1984-06-09", "aaa")
    bm25 = _bm25_returning([
        _hit("aaa", "1", 1.0),  # high BM25 but no alias match
        _hit("aaa", "2", 1.0),
        _hit("84-11_1984-06-09", "1", 0.1),
    ])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("ما هي شروط الزواج في قانون الأسرة؟")
    assert out.doc_ids[0] == "84-11_1984-06-09"


def test_ranking_is_deterministic_on_ties():
    """Equal fused scores must break ties by lexicographic doc_id order."""
    reg = _registry_with_docs("zzz", "aaa", "mmm")
    bm25 = _bm25_returning([
        _hit("zzz", "1", 1.0),
        _hit("aaa", "1", 1.0),
        _hit("mmm", "1", 1.0),
    ])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("q")
    assert out.doc_ids == ["aaa", "mmm", "zzz"]


def test_scores_breakdown_contains_every_signalling_doc():
    """`scores` exposes every doc that received any signal, not just top-N."""
    reg = _registry_with_docs("aaa", "bbb", "ccc")
    bm25 = _bm25_returning([
        _hit("aaa", "1", 1.0),
        _hit("bbb", "1", 0.5),
        _hit("ccc", "1", 0.1),
    ])
    router = build_doc_router(registry=reg, bm25=bm25, top_n=1)
    out = router.route("q")
    assert set(out.scores.keys()) == {"aaa", "bbb", "ccc"}
    assert len(out.doc_ids) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_result():
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([_hit("aaa", "1", 1.0)])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("   ")
    assert out.doc_ids == []
    assert out.scores == {}
    assert out.sources == {}
    assert out.confidence == 0.0


def test_no_signals_returns_empty_result():
    """Query with no aliases / no BM25 hits -> empty result."""
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("garbage qqqzzz")
    assert out.doc_ids == []
    assert out.confidence == 0.0


def test_router_works_without_bm25_index():
    """When no BM25 index is supplied, alias-only routing still works."""
    reg = _registry_with_docs("84-11_1984-06-09")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("ما هي شروط الزواج في قانون الأسرة؟")
    assert out.doc_ids == ["84-11_1984-06-09"]


def test_top_n_zero_returns_empty_result():
    reg = _registry_with_docs("aaa")
    bm25 = _bm25_returning([_hit("aaa", "1", 1.0)])
    router = build_doc_router(registry=reg, bm25=bm25)
    out = router.route("q", top_n=0)
    assert out.doc_ids == []


def test_canonical_id_alias_resolves_to_itself():
    """A literal canonical id appearing in the query routes to that id."""
    reg = _registry_with_docs("90-11_1990-04-21")
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route("see 90-11_1990-04-21 for details")
    assert "90-11_1990-04-21" in out.doc_ids


# ---------------------------------------------------------------------------
# Phase E.4 — KG-derived doc-router channel
# ---------------------------------------------------------------------------


def test_kg_channel_adds_bonus_to_returned_doc_ids():
    """The KG channel returns doc_ids the KG associates with the query
    concept; each gets ``kg_bonus`` (default 0.5)."""
    reg = _registry_with_docs("90-11_1990-04-21", "84-11_1984-06-09")
    bm25 = _bm25_returning([])  # no BM25 signal

    def _kg_call(_q: str) -> list[str]:
        return ["84-11_1984-06-09"]

    router = DocRouter(registry=reg, bm25=bm25, kg_call=_kg_call, kg_bonus=0.5)
    out = router.route("سؤال بدون أي إشارة لفظية", top_n=3)
    assert "84-11_1984-06-09" in out.doc_ids
    assert "kg" in out.sources["84-11_1984-06-09"]
    assert out.scores["84-11_1984-06-09"] == 0.5


def test_kg_channel_stacks_with_alias_and_bm25():
    """When alias + BM25 + KG all surface the same doc, all three
    bonuses stack — the doc ranks well above any single-channel
    competitor."""
    reg = _registry_with_docs("84-11_1984-06-09", "75-58_1975-09-26")
    bm25 = _bm25_returning([_hit("84-11_1984-06-09", "1", 5.0)])

    def _kg_call(_q: str) -> list[str]:
        return ["84-11_1984-06-09"]

    router = DocRouter(
        registry=reg, bm25=bm25, kg_call=_kg_call,
        alias_bonus=1.0, bm25_weight=1.0, kg_bonus=0.5,
    )
    out = router.route("ما هي شروط الزواج في قانون الأسرة؟", top_n=3)
    assert out.doc_ids[0] == "84-11_1984-06-09"
    assert {"alias", "bm25", "kg"} <= set(out.sources["84-11_1984-06-09"])


def test_kg_channel_no_call_is_noop():
    """When ``kg_call`` is None, the router behaves identically to F5."""
    reg = _registry_with_docs("84-11_1984-06-09")
    bm25 = _bm25_returning([_hit("84-11_1984-06-09", "1", 5.0)])
    router = DocRouter(registry=reg, bm25=bm25, kg_call=None)
    out = router.route("قانون الأسرة", top_n=3)
    assert "kg" not in out.sources.get("84-11_1984-06-09", [])


def test_kg_channel_silent_on_exception():
    """A raising kg_call must not break the router; it falls back to
    the alias/BM25 result."""
    reg = _registry_with_docs("84-11_1984-06-09", "75-58_1975-09-26")
    bm25 = _bm25_returning([_hit("75-58_1975-09-26", "1", 3.0)])

    def _broken_kg(_q: str) -> list[str]:
        raise RuntimeError("graph crash")

    router = DocRouter(registry=reg, bm25=bm25, kg_call=_broken_kg)
    out = router.route("سؤال", top_n=3)
    assert "75-58_1975-09-26" in out.doc_ids
    assert "kg" not in out.sources.get("75-58_1975-09-26", [])


def test_kg_channel_set_after_init():
    """``set_kg_call`` lets the dispatcher attach the KG channel after
    the router was built (lazy KG load pattern)."""
    reg = _registry_with_docs("84-11_1984-06-09")
    bm25 = _bm25_returning([])
    router = DocRouter(registry=reg, bm25=bm25)
    out1 = router.route("سؤال غير محدد", top_n=3)
    assert "kg" not in out1.sources.get("84-11_1984-06-09", [])
    router.set_kg_call(lambda _q: ["84-11_1984-06-09"])
    out2 = router.route("سؤال غير محدد", top_n=3)
    assert "84-11_1984-06-09" in out2.doc_ids
    assert "kg" in out2.sources["84-11_1984-06-09"]


# ---------------------------------------------------------------------------
# Integration with benchmark-style queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("ما هي شروط الزواج في قانون الأسرة؟", "84-11_1984-06-09"),
    ("ما هي أحكام البيع في القانون المدني؟", "75-58_1975-09-26"),
    ("ما هي عقوبة السرقة في قانون العقوبات؟", "66-156_1966-06-08"),
    ("ما هو قانون الاستثمار الجزائري الجديد؟", "22-18_2022-07-24"),
    ("ما هي حقوق العامل في قانون العمل؟", "90-11_1990-04-21"),
])
def test_realistic_arabic_queries_route_to_correct_doc(query, expected):
    """Spot check: each major Algerian code has an Arabic alias that should
    route the question correctly.  This locks the alias-channel contract
    that the smoke-eval gate (≥80% top-3 recall) depends on."""
    reg = _registry_with_docs(
        "84-11_1984-06-09",
        "75-58_1975-09-26",
        "66-156_1966-06-08",
        "22-18_2022-07-24",
        "90-11_1990-04-21",
    )
    router = build_doc_router(registry=reg, bm25=None)
    out = router.route(query, top_n=3)
    assert expected in out.doc_ids
