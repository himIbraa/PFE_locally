"""Tests for the Phase E.3 concept-KG retrieval channel:
:func:`akn_rlm.rlm.enhancers.make_concept_kg_channel` and
:func:`akn_rlm.rlm.enhancers.merge_hybrid_with_concept_kg`.

SPARQL is fully mocked — these tests never touch rdflib.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.rlm.enhancers import (
    DEFAULT_E6_BASE_SCORE,
    is_e6_enabled,
    is_e7_enabled,
    make_concept_kg_channel,
    make_kg_doc_router_call,
    merge_hybrid_with_concept_kg,
)


# ---------------------------------------------------------------------------
# is_e6_enabled
# ---------------------------------------------------------------------------


def test_is_e6_enabled_off_by_default(monkeypatch):
    monkeypatch.delenv("AKN_E6_CONCEPT_KG", raising=False)
    monkeypatch.delenv("AKN_ENHANCERS", raising=False)
    assert is_e6_enabled() is False


def test_is_e6_enabled_on(monkeypatch):
    monkeypatch.setenv("AKN_E6_CONCEPT_KG", "1")
    assert is_e6_enabled() is True


# ---------------------------------------------------------------------------
# make_concept_kg_channel
# ---------------------------------------------------------------------------


def _routing_sparql(phrase_to_rows: dict[str, list[dict]]) -> MagicMock:
    """Build a mock SPARQL that returns rows keyed on the phrase
    appearing inside the SPARQL string (between the FILTER's quotes)."""
    def _fn(query: str):
        for phrase, rows in phrase_to_rows.items():
            if f'"{phrase}"' in query:
                return list(rows)
        return []
    return MagicMock(side_effect=_fn)


def test_concept_kg_channel_no_sparql_returns_noop():
    channel = make_concept_kg_channel(None)
    assert channel("any query", None) == []


def test_concept_kg_channel_empty_query_returns_empty():
    sparql = _routing_sparql({})
    channel = make_concept_kg_channel(sparql)
    assert channel("", None) == []


def test_concept_kg_channel_emits_candidates_for_matching_phrase():
    """A SPARQL match on any phrase produces a candidate dict shaped
    like the RRF fuse output: chunk_id / doc_id / article_ref / text /
    score / retriever / kg_concept_match."""
    sparql = _routing_sparql({
        # The TF tokenizer produces "علاقات العمل" as a bigram from
        # this query — that's the phrase the SPARQL string will contain.
        "علاقات العمل": [
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_1",
             "text": "نص المادة الأولى"},
        ],
    })
    channel = make_concept_kg_channel(sparql)
    out = channel("ما هي علاقات العمل في القانون؟", None)
    assert len(out) == 1
    cand = out[0]
    assert cand["doc_id"] == "90-11_1990-04-21"
    assert cand["article_ref"] == "1"
    assert cand["text"] == "نص المادة الأولى"
    assert cand["retriever"] == "kg_concept"
    assert cand["kg_concept_match"] is True
    assert cand["score"] == DEFAULT_E6_BASE_SCORE


def test_concept_kg_channel_multi_phrase_hit_bumps_score():
    """When the same article matches multiple phrases, its score
    accumulates by +0.05 per additional hit (capped at 1.0)."""
    art_uri = "https://legal.dz/resource/law/1990-04-21/90-11#art_1"
    sparql = _routing_sparql({
        "علاقات العمل": [{"article": art_uri, "text": "النص"}],
        # The TF tokenizer also emits other bigrams from this query.
        "العمل بين": [{"article": art_uri, "text": "النص"}],
    })
    channel = make_concept_kg_channel(sparql)
    out = channel("علاقات العمل بين العمال", None)
    # Both phrases matched the SAME article → one entry, score bumped.
    assert len(out) == 1
    assert out[0]["score"] >= DEFAULT_E6_BASE_SCORE + 0.05


def test_concept_kg_channel_respects_routed_ids():
    """Candidates whose doc_id is not in the routed set are dropped
    so the channel respects the dispatcher's doc routing."""
    sparql = _routing_sparql({
        "علاقات العمل": [
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_1",
             "text": "match"},
            {"article": "https://legal.dz/resource/law/1990-04-21/04-99#art_1",
             "text": "wrong doc"},
        ],
    })
    channel = make_concept_kg_channel(sparql)
    out = channel("ما هي علاقات العمل؟", routed_ids=["90-11_1990-04-21"])
    assert len(out) == 1
    assert out[0]["doc_id"] == "90-11_1990-04-21"


def test_concept_kg_channel_silently_skips_unparseable_uri():
    """A SPARQL row whose article URI doesn't match the AKN-RLM pattern
    is silently skipped — no crash, no entry."""
    sparql = _routing_sparql({
        "علاقات العمل": [{"article": "not_a_valid_uri", "text": "x"}],
    })
    channel = make_concept_kg_channel(sparql)
    assert channel("علاقات العمل بين العمال", None) == []


def test_concept_kg_channel_sparql_exception_silently_returns_empty():
    """An rdflib exception during the SPARQL call must not propagate;
    the channel returns whatever it could collect (here: nothing)."""
    raising = MagicMock(side_effect=RuntimeError("graph crash"))
    channel = make_concept_kg_channel(raising)
    # Phrase extraction succeeds; SPARQL call raises per-phrase but is
    # caught inside the channel — the function returns [] cleanly.
    assert channel("علاقات العمل بين العمال", None) == []


def test_concept_kg_channel_no_phrases_returns_empty():
    """Queries too short to yield bigrams AND too short for unigrams
    (no token >=6 chars) produce no SPARQL calls."""
    sparql = MagicMock()
    channel = make_concept_kg_channel(sparql)
    out = channel("لا", None)
    assert out == []
    sparql.assert_not_called()


# ---------------------------------------------------------------------------
# Unigram fallback (Phase E.3 — 2026-05-14 extension)
# ---------------------------------------------------------------------------


def test_concept_kg_channel_unigram_fallback_off_by_default():
    """The unigram fallback is OPT-IN only — by default the channel
    is bigram-only, so a query with no bigram hits returns []. This
    documents the 2026-05-14 behaviour change after the unigram-
    fallback regressed MH/RA on the full-244 benchmark."""
    art_uri = "https://legal.dz/resource/law/1990-04-21/90-11#art_1"
    sparql = _routing_sparql({
        "العلاقات": [{"article": art_uri, "text": "النص"}],
    })
    channel = make_concept_kg_channel(sparql)  # default: fallback off
    out = channel("ما هي العلاقات الفردية في العمل؟", None)
    assert out == []


def test_concept_kg_channel_unigram_fallback_fires_when_enabled():
    """With ``enable_unigram_fallback=True``, the fallback surfaces
    long-unigram matches when bigrams produce zero candidates.
    Preserved for the future operator (Phase F corpus-tuned
    embedder may make this safe to re-enable)."""
    art_uri = "https://legal.dz/resource/law/1990-04-21/90-11#art_1"
    sparql = _routing_sparql({
        "العلاقات": [{"article": art_uri, "text": "النص"}],
    })
    channel = make_concept_kg_channel(sparql, enable_unigram_fallback=True)
    out = channel("ما هي العلاقات الفردية في العمل؟", None)
    assert len(out) >= 1
    from akn_rlm.rlm.enhancers import DEFAULT_E6_UNIGRAM_BASE_SCORE
    assert out[0]["score"] == DEFAULT_E6_UNIGRAM_BASE_SCORE
    assert out[0]["doc_id"] == "90-11_1990-04-21"


def test_concept_kg_channel_unigram_fallback_skipped_when_bigrams_hit():
    """Even with the fallback enabled, it does NOT fire when bigrams
    found any candidate — we keep the precise multi-token signal."""
    bigram_match = "https://legal.dz/resource/law/2020-01-01/foo#art_5"
    sparql = _routing_sparql({
        "علاقات العمل": [{"article": bigram_match, "text": "match"}],
        "العلاقات": [{"article": "https://legal.dz/resource/law/2020-01-01/bar#art_1",
                       "text": "should not appear"}],
    })
    channel = make_concept_kg_channel(sparql, enable_unigram_fallback=True)
    out = channel("ما هي علاقات العمل بين العمال؟", None)
    refs = {c["article_ref"] for c in out}
    assert "1" not in refs


def test_concept_kg_channel_unigram_fallback_caps_at_top_n():
    """Cap unigram noise even when the fallback is enabled."""
    base = "https://legal.dz/resource/law/2020-01-01/foo#art_"
    sparql = _routing_sparql({
        "العقوبة": [{"article": f"{base}{i}", "text": "x"} for i in range(1, 11)],
    })
    channel = make_concept_kg_channel(
        sparql, enable_unigram_fallback=True, unigram_top_n=3,
    )
    out = channel("ما هي العقوبة في القانون؟", None)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# merge_hybrid_with_concept_kg
# ---------------------------------------------------------------------------


def test_merge_empty_concept_kg_returns_hybrid_unchanged():
    hybrid = [{"doc_id": "d", "article_ref": "1", "score": 0.5}]
    assert merge_hybrid_with_concept_kg(hybrid, []) == hybrid


def test_merge_adds_new_concept_kg_candidates():
    hybrid = [
        {"doc_id": "d1", "article_ref": "1", "score": 0.5, "text": "h1"},
    ]
    kg = [
        {"doc_id": "d1", "article_ref": "2", "score": 0.35, "text": "k2",
         "kg_concept_match": True},
    ]
    out = merge_hybrid_with_concept_kg(hybrid, kg)
    refs = {c["article_ref"] for c in out}
    assert refs == {"1", "2"}


def test_merge_keeps_higher_score_when_keys_collide():
    hybrid = [
        {"doc_id": "d1", "article_ref": "1", "score": 0.6, "text": "from_hybrid"},
    ]
    kg = [
        {"doc_id": "d1", "article_ref": "1", "score": 0.35, "text": "from_kg",
         "kg_concept_match": True},
    ]
    out = merge_hybrid_with_concept_kg(hybrid, kg)
    assert len(out) == 1
    # Hybrid wins on score; concept-match flag is preserved.
    assert out[0]["score"] == 0.6
    assert out[0]["text"] == "from_hybrid"
    assert out[0].get("kg_concept_match") is True


def test_merge_kg_wins_when_higher_score():
    """When the KG entry has a higher score (multi-phrase bonus), it
    wins and the hybrid text falls back if KG text was missing."""
    hybrid = [
        {"doc_id": "d1", "article_ref": "1", "score": 0.2, "text": "from_hybrid"},
    ]
    kg = [
        {"doc_id": "d1", "article_ref": "1", "score": 0.6, "text": "",
         "kg_concept_match": True},
    ]
    out = merge_hybrid_with_concept_kg(hybrid, kg)
    assert len(out) == 1
    assert out[0]["score"] == 0.6
    # Empty KG text falls back to hybrid's text.
    assert out[0]["text"] == "from_hybrid"
    assert out[0].get("kg_concept_match") is True


def test_merge_returns_sorted_by_score_desc():
    hybrid = [
        {"doc_id": "d1", "article_ref": "1", "score": 0.3},
        {"doc_id": "d1", "article_ref": "2", "score": 0.5},
    ]
    kg = [
        {"doc_id": "d1", "article_ref": "3", "score": 0.7,
         "kg_concept_match": True},
    ]
    out = merge_hybrid_with_concept_kg(hybrid, kg)
    scores = [c["score"] for c in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0]["article_ref"] == "3"


# ---------------------------------------------------------------------------
# Phase E.4 — make_kg_doc_router_call
# ---------------------------------------------------------------------------


def test_is_e7_enabled_off_by_default(monkeypatch):
    monkeypatch.delenv("AKN_E7_KG_DOC_ROUTER", raising=False)
    monkeypatch.delenv("AKN_ENHANCERS", raising=False)
    assert is_e7_enabled() is False


def test_is_e7_enabled_on(monkeypatch):
    monkeypatch.setenv("AKN_E7_KG_DOC_ROUTER", "1")
    assert is_e7_enabled() is True


def test_kg_doc_router_call_no_sparql_returns_noop():
    fn = make_kg_doc_router_call(None)
    assert fn("any query") == []


def test_kg_doc_router_call_emits_distinct_doc_ids():
    """The router call returns one doc_id per matched article (order-
    preserving, deduplicated). Article URIs are parsed to (doc, ref)
    pairs and the doc component is collected."""
    sparql = _routing_sparql({
        # Multiple articles in the same doc match the same phrase.
        "علاقات العمل": [
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_1"},
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_5"},
            {"article": "https://legal.dz/resource/law/1984-06-09/84-11#art_2"},
        ],
    })
    fn = make_kg_doc_router_call(sparql)
    out = fn("ما هي علاقات العمل بين العمال؟")
    # 90-11 surfaces once even though two articles matched; 84-11 once.
    assert out == ["90-11_1990-04-21", "84-11_1984-06-09"]


def test_kg_doc_router_call_respects_max_docs():
    """The cap prevents the channel from flooding the router with
    weak-precision matches."""
    sparql = _routing_sparql({
        "علاقات العمل": [
            {"article": f"https://legal.dz/resource/law/2020-01-01/doc{i}#art_1"}
            for i in range(20)
        ],
    })
    fn = make_kg_doc_router_call(sparql, max_docs=3)
    out = fn("ما هي علاقات العمل بين العمال؟")
    assert len(out) == 3


def test_kg_doc_router_call_empty_query_returns_empty():
    sparql = MagicMock()
    fn = make_kg_doc_router_call(sparql)
    assert fn("") == []
    sparql.assert_not_called()


def test_kg_doc_router_call_silent_on_sparql_exception():
    raising = MagicMock(side_effect=RuntimeError("graph crash"))
    fn = make_kg_doc_router_call(raising)
    assert fn("علاقات العمل بين العمال") == []
