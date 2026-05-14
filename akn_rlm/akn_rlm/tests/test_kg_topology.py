"""Tests for the Phase E.2 KG topology helpers in
``akn_rlm.rlm.enhancers``: structural-distance, chapter-title
match, articles-in-container, and the multi_hop disambiguator
factory.

SPARQL is fully mocked — these tests never touch rdflib and run in
milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.rlm.enhancers import (
    _kg_article_ancestors,
    is_e5_enabled,
    kg_articles_in_container,
    kg_containers_matching_phrase,
    kg_structural_distance,
    make_kg_topology_disambiguator,
)


# ---------------------------------------------------------------------------
# Mock SPARQL helpers
# ---------------------------------------------------------------------------


def _stub_sparql(routing: dict[str, list[dict]] | None = None) -> MagicMock:
    """Build a mock SPARQL function that returns canned rows depending on
    the substring keys present in the query (matched verbatim).
    """
    routing = dict(routing or {})

    def _fn(query: str):
        # Return the rows for the FIRST key found inside the query.
        for marker, rows in routing.items():
            if marker in query:
                return list(rows)
        return []

    return MagicMock(side_effect=_fn)


# ---------------------------------------------------------------------------
# is_e5_enabled
# ---------------------------------------------------------------------------


def test_is_e5_enabled_off_by_default(monkeypatch):
    monkeypatch.delenv("AKN_E5_KG_TOPOLOGY", raising=False)
    monkeypatch.delenv("AKN_ENHANCERS", raising=False)
    assert is_e5_enabled() is False


def test_is_e5_enabled_on(monkeypatch):
    monkeypatch.setenv("AKN_E5_KG_TOPOLOGY", "1")
    assert is_e5_enabled() is True


def test_is_e5_enabled_master_flag(monkeypatch):
    monkeypatch.delenv("AKN_E5_KG_TOPOLOGY", raising=False)
    monkeypatch.setenv("AKN_ENHANCERS", "all")
    assert is_e5_enabled() is True


# ---------------------------------------------------------------------------
# kg_article_ancestors (private — used by structural distance)
# ---------------------------------------------------------------------------


def test_ancestors_returns_chapters_and_sections():
    uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    sparql = _stub_sparql({
        uri: [
            {"ancestor": "https://legal.dz/resource/law/1984-06-09/84-11#chp_2", "type": "Chapter"},
            {"ancestor": "https://legal.dz/resource/law/1984-06-09/84-11#chp_2_sec_3", "type": "Section"},
        ],
    })
    result = _kg_article_ancestors(sparql, uri)
    assert result == {
        "https://legal.dz/resource/law/1984-06-09/84-11#chp_2",
        "https://legal.dz/resource/law/1984-06-09/84-11#chp_2_sec_3",
    }


def test_ancestors_empty_sparql_fn():
    assert _kg_article_ancestors(None, "u") == set()


def test_ancestors_empty_uri():
    sparql = _stub_sparql({})
    assert _kg_article_ancestors(sparql, "") == set()


def test_ancestors_sparql_exception_returns_empty():
    raising = MagicMock(side_effect=RuntimeError("boom"))
    assert _kg_article_ancestors(raising, "uri") == set()


# ---------------------------------------------------------------------------
# kg_structural_distance
# ---------------------------------------------------------------------------


def test_structural_distance_same_section_returns_0():
    uri_a = "https://legal.dz/resource/law/X/foo#art_1"
    uri_b = "https://legal.dz/resource/law/X/foo#art_2"
    sparql = _stub_sparql({
        uri_a: [
            {"ancestor": "ch1", "type": "Chapter"},
            {"ancestor": "ch1_s1", "type": "Section"},
        ],
        uri_b: [
            {"ancestor": "ch1", "type": "Chapter"},
            {"ancestor": "ch1_s1", "type": "Section"},
        ],
    })
    assert kg_structural_distance(sparql, uri_a, uri_b) == 0.0


def test_structural_distance_same_chapter_different_section_returns_1():
    uri_a = "https://legal.dz/resource/law/X/foo#art_1"
    uri_b = "https://legal.dz/resource/law/X/foo#art_2"
    sparql = _stub_sparql({
        uri_a: [
            {"ancestor": "ch1", "type": "Chapter"},
            {"ancestor": "ch1_s1", "type": "Section"},
        ],
        uri_b: [
            {"ancestor": "ch1", "type": "Chapter"},
            {"ancestor": "ch1_s2", "type": "Section"},
        ],
    })
    assert kg_structural_distance(sparql, uri_a, uri_b) == 1.0


def test_structural_distance_same_doc_different_chapter_returns_2():
    uri_a = "https://legal.dz/resource/law/X/foo#art_1"
    uri_b = "https://legal.dz/resource/law/X/foo#art_42"
    sparql = _stub_sparql({
        uri_a: [{"ancestor": "ch1", "type": "Chapter"}],
        uri_b: [{"ancestor": "ch2", "type": "Chapter"}],
    })
    # Doc URI prefix matches → 2.0
    assert kg_structural_distance(sparql, uri_a, uri_b) == 2.0


def test_structural_distance_different_docs_returns_infinity():
    uri_a = "https://legal.dz/resource/law/X/foo#art_1"
    uri_b = "https://legal.dz/resource/law/Y/bar#art_1"
    sparql = _stub_sparql({
        uri_a: [{"ancestor": "fooch1", "type": "Chapter"}],
        uri_b: [{"ancestor": "barch1", "type": "Chapter"}],
    })
    assert kg_structural_distance(sparql, uri_a, uri_b) == float("inf")


def test_structural_distance_no_ancestors_falls_back_to_doc_uri():
    uri_a = "https://legal.dz/resource/law/X/foo#art_1"
    uri_b = "https://legal.dz/resource/law/X/foo#art_2"
    sparql = _stub_sparql({})  # no ancestors for either
    # No ancestors → same doc URI prefix → distance 2.0
    assert kg_structural_distance(sparql, uri_a, uri_b) == 2.0


# ---------------------------------------------------------------------------
# kg_containers_matching_phrase
# ---------------------------------------------------------------------------


def test_containers_matching_phrase_returns_uri_with_concept_in_fragment():
    doc_prefix = "https://legal.dz/resource/law/X/foo"
    sparql = _stub_sparql({
        doc_prefix: [
            {"container": f"{doc_prefix}#chp_الثاني_السلطة_و_تنظيمها",
             "type": "Chapter"},
            {"container": f"{doc_prefix}#chp_1",  # bare numeric — no match
             "type": "Chapter"},
        ],
    })
    out = kg_containers_matching_phrase(sparql, doc_prefix, ["السلطة"])
    assert len(out) == 1
    assert out[0][0] == f"{doc_prefix}#chp_الثاني_السلطة_و_تنظيمها"
    assert out[0][1] == "السلطة"


def test_containers_matching_phrase_skips_short_phrases():
    doc_prefix = "https://legal.dz/resource/law/X/foo"
    sparql = _stub_sparql({
        doc_prefix: [{"container": f"{doc_prefix}#chp_AB", "type": "Chapter"}],
    })
    # < 3 chars → silently skipped
    assert kg_containers_matching_phrase(sparql, doc_prefix, ["AB"]) == []


def test_containers_matching_phrase_no_kg_returns_empty():
    assert kg_containers_matching_phrase(None, "x", ["y"]) == []


def test_containers_matching_phrase_empty_phrases_returns_empty():
    sparql = _stub_sparql({})
    assert kg_containers_matching_phrase(sparql, "x", []) == []


# ---------------------------------------------------------------------------
# kg_articles_in_container
# ---------------------------------------------------------------------------


def test_articles_in_container_returns_uri_list_ordered():
    container = "https://legal.dz/resource/law/X/foo#chp_1"
    sparql = _stub_sparql({
        container: [
            {"article": f"{container.split('#')[0]}#art_3"},
            {"article": f"{container.split('#')[0]}#art_1"},
            {"article": f"{container.split('#')[0]}#art_2"},
        ],
    })
    out = kg_articles_in_container(sparql, container)
    # The SPARQL itself contains the ORDER BY clause; the helper
    # passes rows through in whatever order the stub returns, so the
    # ordering check here just verifies the list is collected
    # without dedup.
    assert len(out) == 3
    assert "art_1" in out[0] or "art_1" in out[1] or "art_1" in out[2]


def test_articles_in_container_respects_limit():
    container = "https://legal.dz/resource/law/X/foo#chp_1"
    sparql = _stub_sparql({
        container: [{"article": f"a/{i}"} for i in range(10)],
    })
    out = kg_articles_in_container(sparql, container, limit=3)
    assert len(out) == 3


def test_articles_in_container_no_sparql_returns_empty():
    assert kg_articles_in_container(None, "x") == []


# ---------------------------------------------------------------------------
# make_kg_topology_disambiguator — integration
# ---------------------------------------------------------------------------


def _resolve_uri_stub(mapping: dict[tuple[str, str], str]):
    def _fn(doc_id: str, ref: str):
        return mapping.get((doc_id, ref))
    return _fn


def test_disambiguator_no_sparql_is_identity():
    fn = make_kg_topology_disambiguator(None, resolve_uri=lambda d, r: None)
    cits = [{"doc_id": "X_2020-01-01", "article_ref": "1", "confidence": 0.5}]
    assert fn("any", cits) == cits


def test_disambiguator_returns_identity_when_fewer_than_two_citations():
    sparql = _stub_sparql({})
    fn = make_kg_topology_disambiguator(sparql, resolve_uri=lambda d, r: None)
    assert fn("q", []) == []
    one = [{"doc_id": "d", "article_ref": "1", "confidence": 0.5}]
    assert fn("q", one) == one


def test_disambiguator_promotes_title_matched_candidate_when_scores_tied():
    """The headline E.2 case: two MH candidates in the same doc with
    similar confidence — one lives in a chapter whose title contains
    the query concept; the disambiguator promotes it via +promote_bonus.
    """
    doc_prefix = "https://legal.dz/resource/law/X/foo"
    uri_a = f"{doc_prefix}#art_5"   # in title-matched chapter
    uri_b = f"{doc_prefix}#art_42"  # in non-matched chapter
    matched_ch = f"{doc_prefix}#chp_الثاني_السلطة_و_تنظيمها"
    other_ch = f"{doc_prefix}#chp_1"

    sparql = _stub_sparql({
        uri_a: [{"ancestor": matched_ch, "type": "Chapter"}],
        uri_b: [{"ancestor": other_ch, "type": "Chapter"}],
        doc_prefix: [
            {"container": matched_ch, "type": "Chapter"},
            {"container": other_ch, "type": "Chapter"},
        ],
    })
    resolve = _resolve_uri_stub({("X_2020-01-01", "5"): uri_a,
                                 ("X_2020-01-01", "42"): uri_b})
    fn = make_kg_topology_disambiguator(
        sparql, resolve_uri=resolve, promote_bonus=0.30,
    )
    # b has higher initial confidence; after disambiguation a should win
    cits = [
        {"doc_id": "X_2020-01-01", "article_ref": "42", "confidence": 0.55},
        {"doc_id": "X_2020-01-01", "article_ref": "5",  "confidence": 0.50},
    ]
    # Query contains "السلطة" — match-phrase fragment.
    out = fn("ما هي السلطة التنفيذية للسلطة المركزية؟", cits)
    assert out[0]["article_ref"] == "5"
    assert out[0].get("kg_topology_promoted") is True
    assert out[0]["confidence"] >= 0.80  # 0.50 + 0.30 bonus


def test_disambiguator_skips_when_no_title_match():
    """Both candidates live in chapters whose titles don't contain any
    query phrase → no promotion, original order preserved."""
    doc_prefix = "https://legal.dz/resource/law/X/foo"
    uri_a = f"{doc_prefix}#art_5"
    uri_b = f"{doc_prefix}#art_42"
    sparql = _stub_sparql({
        uri_a: [{"ancestor": f"{doc_prefix}#chp_1", "type": "Chapter"}],
        uri_b: [{"ancestor": f"{doc_prefix}#chp_2", "type": "Chapter"}],
        doc_prefix: [
            {"container": f"{doc_prefix}#chp_1", "type": "Chapter"},
            {"container": f"{doc_prefix}#chp_2", "type": "Chapter"},
        ],
    })
    resolve = _resolve_uri_stub({("X_2020-01-01", "5"): uri_a,
                                 ("X_2020-01-01", "42"): uri_b})
    fn = make_kg_topology_disambiguator(sparql, resolve_uri=resolve)
    cits = [
        {"doc_id": "X_2020-01-01", "article_ref": "42", "confidence": 0.55},
        {"doc_id": "X_2020-01-01", "article_ref": "5",  "confidence": 0.50},
    ]
    # No phrase from this query appears in either chapter URI fragment.
    out = fn("ما هي السلطة التنفيذية؟", cits)
    # Order preserved (b was first, still first).
    assert out[0]["article_ref"] == "42"
    assert "kg_topology_promoted" not in out[0]
    assert "kg_topology_promoted" not in out[1]


def test_disambiguator_skips_when_different_docs():
    """Candidates in different docs are never disambiguated — the
    title-match channel only meaningfully fires within a doc."""
    doc_a = "https://legal.dz/resource/law/X/foo"
    doc_b = "https://legal.dz/resource/law/Y/bar"
    matched_ch_b = f"{doc_b}#chp_السلطة"
    sparql = _stub_sparql({
        f"{doc_a}#art_5": [{"ancestor": f"{doc_a}#chp_1", "type": "Chapter"}],
        f"{doc_b}#art_42": [{"ancestor": matched_ch_b, "type": "Chapter"}],
        doc_b: [{"container": matched_ch_b, "type": "Chapter"}],
    })
    resolve = _resolve_uri_stub({
        ("X_2020-01-01", "5"): f"{doc_a}#art_5",
        ("Y_2020-01-01", "42"): f"{doc_b}#art_42",
    })
    fn = make_kg_topology_disambiguator(sparql, resolve_uri=resolve)
    cits = [
        {"doc_id": "X_2020-01-01", "article_ref": "5",  "confidence": 0.55},
        {"doc_id": "Y_2020-01-01", "article_ref": "42", "confidence": 0.50},
    ]
    out = fn("السلطة", cits)
    # No promotion across docs; original order kept.
    assert out[0]["article_ref"] == "5"
    assert "kg_topology_promoted" not in out[0]


def test_disambiguator_skips_when_scores_outside_window():
    """Candidates with abs(score_diff) > similarity_window are not
    considered tied and the disambiguator doesn't touch them."""
    doc_prefix = "https://legal.dz/resource/law/X/foo"
    uri_a = f"{doc_prefix}#art_5"
    uri_b = f"{doc_prefix}#art_42"
    matched_ch = f"{doc_prefix}#chp_السلطة"
    sparql = _stub_sparql({
        uri_a: [{"ancestor": matched_ch, "type": "Chapter"}],
        uri_b: [{"ancestor": f"{doc_prefix}#chp_1", "type": "Chapter"}],
        doc_prefix: [{"container": matched_ch, "type": "Chapter"}],
    })
    resolve = _resolve_uri_stub({("d", "5"): uri_a, ("d", "42"): uri_b})
    fn = make_kg_topology_disambiguator(
        sparql, resolve_uri=resolve, similarity_window=0.10,
    )
    # 0.90 vs 0.50 — diff 0.40 > 0.10 window → no disambiguation.
    cits = [
        {"doc_id": "d", "article_ref": "42", "confidence": 0.90},
        {"doc_id": "d", "article_ref": "5",  "confidence": 0.50},
    ]
    out = fn("السلطة", cits)
    assert out[0]["article_ref"] == "42"
    assert "kg_topology_promoted" not in out[0]


def test_disambiguator_handles_unresolvable_uri_silently():
    """If resolve_uri returns None for a candidate, the disambiguator
    skips it without crashing."""
    sparql = _stub_sparql({})

    def _resolve(_doc, _ref):
        return None  # always fails

    fn = make_kg_topology_disambiguator(sparql, resolve_uri=_resolve)
    cits = [
        {"doc_id": "d", "article_ref": "5",  "confidence": 0.55},
        {"doc_id": "d", "article_ref": "42", "confidence": 0.50},
    ]
    out = fn("q", cits)
    assert len(out) == 2  # no crash
    assert "kg_topology_promoted" not in out[0]
