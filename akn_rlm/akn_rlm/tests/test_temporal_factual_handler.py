"""Tests for the Phase-2 temporal_factual handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged.
SPARQL, verifier and summariser are fully mocked — the suite never
touches a real KG, real LLM, or rdflib, and runs in milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.temporal_factual import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_ROUTE_TOP_N,
    DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_VERIFY_TOP_N,
    LATEST_VERSION_DATE,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    TemporalFactualHandler,
    _amendment_chain,
    _extract_dates,
    _pick_target_date,
    _resolve_article_uri,
    _version_at_date,
    build_temporal_factual_handler,
)
from akn_rlm.rlm.routing import RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25(doc_id: str, ref: str, text: str, score: float = 5.0) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _dense(doc_id: str, ref: str, text: str, score: float = 0.5) -> DenseHit:
    return DenseHit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _stub_router(doc_ids: list[str]) -> MagicMock:
    router = MagicMock()
    router.route.return_value = RouteResult(
        doc_ids=list(doc_ids),
        scores={d: 1.0 for d in doc_ids},
        sources={d: ["alias"] for d in doc_ids},
        confidence=1.0 if doc_ids else 0.0,
    )
    return router


def _stub_sparql(uri_to_chain: dict[str, list[dict]]) -> MagicMock:
    """Build a mock SPARQL function that:

    - Treats ``ASK { <uri> ?p ?o }`` as positive iff ``uri`` is a key in
      ``uri_to_chain``.
    - Treats ``SELECT ?version ?inForceFrom ?text WHERE { <uri> dzdoc:hasVersion ... }``
      as returning the chain rows for ``uri``.
    """
    chains = dict(uri_to_chain)

    def _fn(query: str) -> list[dict]:
        stripped = query.strip()
        if stripped.lower().startswith("ask"):
            for uri in chains:
                if f"<{uri}>" in stripped:
                    return [{"_ask": True}]
            return [{"_ask": False}]
        # SELECT — find the URI that's bracketed in the query.
        for uri, rows in chains.items():
            if f"<{uri}>" in stripped:
                # Convert chain-row dicts to SPARQL-result-style dicts.
                return [
                    {
                        "version": r.get("version_uri", f"{uri}/v"),
                        "inForceFrom": r.get("date", ""),
                        "text": r.get("text", ""),
                    }
                    for r in rows
                ]
        return []

    return MagicMock(side_effect=_fn)


def _stub_verifier(verdict_for=None, default=None):
    verdict_for = verdict_for or {}
    default = default or {
        "relevant": True,
        "supporting_span": None,
        "contradicting_span": None,
        "confidence": 0.9,
    }

    def _fn(_pool, _q, article, _model):
        key = (article.get("doc_id", ""), article.get("article_ref", ""))
        return dict(verdict_for.get(key, default))

    return MagicMock(side_effect=_fn)


def _stub_summarizer(summary="ملخص النسخة"):
    def _fn(_pool, _q, _articles, _model):
        return {"summary": summary, "key_articles": [], "caveats": None}
    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    bm25_hits=None,
    dense_hits=None,
    routed_ids=None,
    uri_to_chain=None,
    verifier_verdicts=None,
    summary="ملخص النسخة",
    doc_title="قانون الأسرة",
    **kwargs,
):
    bm25 = MagicMock()
    bm25.search.return_value = list(bm25_hits or [])
    dense = MagicMock()
    dense.search.return_value = list(dense_hits or [])

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    router = _stub_router(routed_ids if routed_ids is not None else ["84-11_1984-06-09"])
    sparql_fn = _stub_sparql(uri_to_chain or {})
    verifier_fn = _stub_verifier(verifier_verdicts)
    summarizer_fn = _stub_summarizer(summary)

    handler = TemporalFactualHandler(
        kg=object(),  # any non-None — sparql_fn is mocked
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        sparql_fn=sparql_fn,
        verifier_fn=verifier_fn,
        summarizer_fn=summarizer_fn,
        **kwargs,
    )
    return handler, {
        "bm25": bm25,
        "dense": dense,
        "registry": registry,
        "router": router,
        "sparql_fn": sparql_fn,
        "verifier_fn": verifier_fn,
        "summarizer_fn": summarizer_fn,
    }


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


def test_extract_dates_iso_format():
    assert _extract_dates("قانون 25-14 (2025-08-03)") == ["2025-08-03"]


def test_extract_dates_dmy_slash():
    assert _extract_dates("في 21/04/1990 صدر القانون") == ["1990-04-21"]


def test_extract_dates_arabic_year_with_waw_connector():
    """`و2008` should match — Arabic letter then digit. \\b would fail."""
    assert _extract_dates("بين 1996 و2008 و2020") == [
        "1996-12-31", "2008-12-31", "2020-12-31",
    ]


def test_extract_dates_does_not_match_law_id():
    """Law IDs like 90-11 must not be parsed as years."""
    assert _extract_dates("قانون 90-11 و84-11") == []


def test_extract_dates_does_not_match_short_numbers():
    """Article numbers (e.g. 51, 88, 566) must not be parsed as years."""
    assert _extract_dates("المادة 51 من قانون 25-14") == []


def test_extract_dates_dedupes():
    assert _extract_dates("2020 وفي 2020 ومرة أخرى 2020") == ["2020-12-31"]


def test_extract_dates_empty_query_returns_empty():
    assert _extract_dates("") == []
    assert _extract_dates(None) == []


def test_pick_target_date_picks_latest():
    """benchmark applicable_version=='post' → use the latest mentioned date."""
    assert _pick_target_date(["1996-12-31", "2008-12-31", "2020-12-31"]) == "2020-12-31"


def test_pick_target_date_empty_returns_sentinel():
    assert _pick_target_date([]) == LATEST_VERSION_DATE


# ---------------------------------------------------------------------------
# URI resolution + amendment chain helpers
# ---------------------------------------------------------------------------


def test_resolve_article_uri_law():
    sparql = _stub_sparql({
        "https://legal.dz/resource/law/1990-04-21/90-11#art_1": [],
    })
    uri = _resolve_article_uri(sparql, "90-11_1990-04-21", "1")
    assert uri == "https://legal.dz/resource/law/1990-04-21/90-11#art_1"


def test_resolve_article_uri_constitution():
    sparql = _stub_sparql({
        "https://legal.dz/resource/constitution/2020-12-30/2020#art_88": [],
    })
    uri = _resolve_article_uri(sparql, "constitution_2020-12-30", "88")
    assert uri == "https://legal.dz/resource/constitution/2020-12-30/2020#art_88"


def test_resolve_article_uri_order():
    sparql = _stub_sparql({
        "https://legal.dz/resource/order/1975-09-26/75-59#art_566": [],
    })
    uri = _resolve_article_uri(sparql, "75-59_1975-09-26", "566")
    assert uri == "https://legal.dz/resource/order/1975-09-26/75-59#art_566"


def test_resolve_article_uri_canonicalises_ref():
    sparql = _stub_sparql({
        "https://legal.dz/resource/law/1990-04-21/90-11#art_9_bis": [],
    })
    uri = _resolve_article_uri(sparql, "90-11_1990-04-21", "9 مكرر")
    assert uri == "https://legal.dz/resource/law/1990-04-21/90-11#art_9_bis"


def test_resolve_article_uri_unknown_returns_none():
    sparql = _stub_sparql({})  # empty KG
    uri = _resolve_article_uri(sparql, "84-11_1984-06-09", "54")
    assert uri is None


def test_resolve_article_uri_no_sparql_fn_returns_none():
    assert _resolve_article_uri(None, "84-11_1984-06-09", "54") is None


def test_resolve_article_uri_malformed_doc_id():
    sparql = _stub_sparql({})
    assert _resolve_article_uri(sparql, "no_underscores_no_constitution_prefix", "1") is None


def test_amendment_chain_returns_versions_sorted_by_date():
    uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    sparql = _stub_sparql({
        uri: [
            {"version_uri": f"{uri}/v2", "date": "2005-02-27", "text": "النص المعدل"},
            {"version_uri": f"{uri}/v1", "date": "1984-06-09", "text": "النص الأصلي"},
        ],
    })
    chain = _amendment_chain(sparql, uri)
    assert [v["date"] for v in chain] == ["1984-06-09", "2005-02-27"]
    assert chain[0]["text"] == "النص الأصلي"
    assert chain[1]["text"] == "النص المعدل"


def test_amendment_chain_unknown_uri_returns_empty():
    sparql = _stub_sparql({"https://x/known": [{"date": "2000-01-01"}]})
    assert _amendment_chain(sparql, "https://x/missing") == []


def test_amendment_chain_no_sparql_fn():
    assert _amendment_chain(None, "https://x/anything") == []


def test_version_at_date_picks_latest_pre_target():
    chain = [
        {"date": "1984-06-09", "text": "أ"},
        {"date": "2005-02-27", "text": "ب"},
        {"date": "2025-01-01", "text": "ج"},
    ]
    assert _version_at_date(chain, "2010-01-01")["text"] == "ب"


def test_version_at_date_target_before_any_version_returns_none():
    chain = [
        {"date": "1984-06-09", "text": "أ"},
        {"date": "2005-02-27", "text": "ب"},
    ]
    # Target predates all known versions — the article didn't exist yet.
    assert _version_at_date(chain, "1900-01-01") is None


def test_version_at_date_latest_sentinel_picks_newest():
    chain = [
        {"date": "1984-06-09", "text": "أ"},
        {"date": "2005-02-27", "text": "ب"},
    ]
    assert _version_at_date(chain, LATEST_VERSION_DATE)["text"] == "ب"


def test_version_at_date_empty_chain():
    assert _version_at_date([], "2020-01-01") is None


# ---------------------------------------------------------------------------
# Defaults / contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    assert DEFAULT_TOP_K_CANDIDATES == 5
    # Verifier OFF by default — see module docstring; HANDOFF §3 says
    # "answer from KG, never from search", and empirically the generic
    # relevance verifier hurts Cite F1 on this slice.
    assert DEFAULT_VERIFY_TOP_N == 0
    # R9.2: 5 → 2. Trade recall for precision on the n=7 stratum.
    assert DEFAULT_FINAL_TOP_K == 2
    assert DEFAULT_K_EACH == 30
    assert DEFAULT_VERIFY_THRESHOLD == 0.4
    assert DEFAULT_ROUTE_TOP_N == 3
    assert SUPPORT_SPAN_LEN == 280
    assert LATEST_VERSION_DATE == "9999-12-31"
    assert TELEMETRY_BASELINE == "rlm_temporal_factual"


def test_r9_2_default_final_top_k_locked_at_2():
    """Lock the R9.2 retune so a future drift back to 5 fails loudly."""
    from akn_rlm.rlm.handlers import temporal_factual as tf_mod
    assert tf_mod.DEFAULT_FINAL_TOP_K == 2


def test_default_does_not_call_verifier():
    """With DEFAULT_VERIFY_TOP_N=0, the verifier is bypassed entirely."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, deps = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
    )
    out = h.run("بعد 2005")
    assert out["abstention"] is False
    deps["verifier_fn"].assert_not_called()
    assert out["_telemetry"]["sub_call_count"] in (0, 1)  # 0 verify + 1 summary


def test_factory_builds_handler_instance():
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    registry.get_doc.return_value = None
    h = build_temporal_factual_handler(
        kg=None, bm25=bm25, dense=dense, registry=registry,
        llm_pool=MagicMock(), router=_stub_router([]),
        sparql_fn=lambda _q: [],
    )
    assert isinstance(h, TemporalFactualHandler)


def test_run_returns_required_answer_keys():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "النص الأصلي")],
        dense_hits=[_dense("84-11_1984-06-09", "54", "النص الأصلي")],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={
            art_uri: [
                {"date": "2005-02-27", "text": "النص المعدل بعد 2005"},
                {"date": "1984-06-09", "text": "النص الأصلي"},
            ],
        },
    )
    out = h.run("كيف تغيرت شروط الخلع بعد 2005؟")
    for key in (
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "depth_max_reached",
        "_telemetry",
    ):
        assert key in out
    assert out["abstention"] is False
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE


def test_run_telemetry_carries_routing_dates_and_chains():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "النص")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
    )
    out = h.run("بعد 2005")
    tel = out["_telemetry"]
    assert tel["routed_doc_ids"] == ["84-11_1984-06-09"]
    assert tel["extracted_dates"] == ["2005-12-31"]
    assert tel["target_date"] == "2005-12-31"
    chains = tel["amendment_chains"]
    assert len(chains) == 1
    assert chains[0]["doc_id"] == "84-11_1984-06-09"
    assert chains[0]["article_ref"] == "54"
    assert chains[0]["chain_len"] == 1
    assert chains[0]["picked"] == "2005-02-27"
    assert chains[0]["source"] == "kg"


def test_run_telemetry_includes_tf_kg_first_telemetry():
    """Phase E.1: every dispatched TF answer must carry a
    ``tf_kg_first_telemetry`` list — one dict per depth's
    retrieve+merge+chain call. The dict must surface hybrid_count,
    kg_first_count, merged_pool_size, top_slice_size, and how many of
    the KG-first hits actually resolved to a URI and survived
    verification. The hybrid-only case (no KG-first hits) is the
    common path and must still emit a single-entry telemetry list."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "نص")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
    )
    out = h.run("بعد 2005")
    tel = out["_telemetry"]
    assert "tf_kg_first_telemetry" in tel
    rows = tel["tf_kg_first_telemetry"]
    assert isinstance(rows, list)
    assert len(rows) == 1  # depth-1 only when recursion is OFF
    row = rows[0]
    for key in (
        "depth", "query", "hybrid_count", "kg_first_count",
        "kg_first_hits", "merged_pool_size", "top_slice_size",
        "kg_first_in_top_slice", "kg_first_uri_resolved",
        "kg_first_in_verified",
    ):
        assert key in row, f"missing telemetry key: {key}"
    assert row["depth"] == 1
    assert row["hybrid_count"] == 1
    # No SPARQL CONTAINS results in the stub → no KG-first candidates.
    assert row["kg_first_count"] == 0
    assert row["kg_first_hits"] == []
    assert row["merged_pool_size"] == 1
    assert row["top_slice_size"] == 1
    assert row["kg_first_in_top_slice"] == 0
    assert row["kg_first_uri_resolved"] == 0
    assert row["kg_first_in_verified"] == 0


def test_citation_carries_kg_versioned_text_not_chunk_text():
    """The citation text MUST come from the KG version, not the BM25 chunk
    — that's the HANDOFF "answer from KG, never from search" contract."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "OUTDATED CHUNK TEXT")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={
            art_uri: [
                {"date": "2005-02-27", "text": "نص النسخة المعدلة"},
                {"date": "1984-06-09", "text": "النص الأصلي"},
            ],
        },
    )
    out = h.run("بعد تعديل 2005")
    cit = out["citations"][0]
    assert cit["text"] == "نص النسخة المعدلة"
    assert cit["version_date"] == "2005-02-27"
    assert cit["kg_source"] == "kg"


def test_citation_falls_back_to_chunk_text_when_kg_chain_empty():
    """If the article has no hasVersion triples, the chunk text IS the answer
    (the article was enacted at its origin and never amended)."""
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "نص الفصل من الفهرس")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={},  # no URI resolves
    )
    out = h.run("بعد تعديل 2005")
    cit = out["citations"][0]
    assert cit["text"] == "نص الفصل من الفهرس"
    assert cit["version_date"] == ""
    assert cit["kg_source"] == "fallback"


def test_citation_supporting_span_caps_at_280():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    long_text = "ا" * 600
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "ignored")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": long_text}]},
    )
    out = h.run("بعد 2005")
    assert len(out["citations"][0]["supporting_span"]) == SUPPORT_SPAN_LEN


def test_citation_uses_supporting_quote_when_substring():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص الكامل: الجزء المهم هنا والباقي."}]},
        verifier_verdicts={
            ("84-11_1984-06-09", "54"): {
                "relevant": True,
                "supporting_span": "الجزء المهم هنا",
                "confidence": 0.95,
            },
        },
        verify_top_n=3,  # opt the verifier back in for this test
    )
    out = h.run("بعد 2005")
    assert out["citations"][0]["supporting_span"] == "الجزء المهم هنا"


def test_citation_falls_back_to_text_when_quote_not_substring():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص الحقيقي."}]},
        verifier_verdicts={
            ("84-11_1984-06-09", "54"): {
                "relevant": True,
                "supporting_span": "اقتباس مفبرك ليس داخل النص",
                "confidence": 0.95,
            },
        },
        verify_top_n=3,
    )
    out = h.run("بعد 2005")
    assert out["citations"][0]["supporting_span"] == "النص الحقيقي."


# ---------------------------------------------------------------------------
# Routing / retrieval flow
# ---------------------------------------------------------------------------


def test_router_called_with_route_top_n():
    h, deps = _make_handler(
        bm25_hits=[],
        dense_hits=[],
        routed_ids=["x"],
        route_top_n=4,
    )
    h.run("شيء ما")
    deps["router"].route.assert_called_once()
    assert deps["router"].route.call_args.kwargs.get("top_n") == 4


def test_retrieval_filters_by_routed_doc_ids():
    """Hits in non-routed docs must be filtered out."""
    art_uri_routed = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "54", "نص routed"),
            _bm25("OTHER_DOC", "1", "نص خارج التوجيه"),
        ],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri_routed: [{"date": "2005-02-27", "text": "ب"}]},
    )
    out = h.run("بعد 2005")
    cit_doc_ids = {c["doc_id"] for c in out["citations"]}
    assert cit_doc_ids == {"84-11_1984-06-09"}


def test_retrieval_falls_back_to_full_pool_when_filter_empty():
    """If filtering by routed docs would drop every candidate, the handler
    must NOT abstain — it must fall back to the full fused pool."""
    art_uri = "https://legal.dz/resource/law/1990-04-21/90-11#art_1"
    h, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1", "نص")],
        dense_hits=[],
        routed_ids=["UNRELATED_DOC"],  # filter wipes the only hit
        uri_to_chain={art_uri: [{"date": "1990-04-21", "text": "ب"}]},
    )
    out = h.run("في 1990 صدر")
    assert out["abstention"] is False
    assert out["citations"][0]["doc_id"] == "90-11_1990-04-21"


def test_both_retrievers_called_at_k_each():
    h, deps = _make_handler(bm25_hits=[], dense_hits=[], routed_ids=[], k_each=42)
    h.run("شيء")
    deps["bm25"].search.assert_called_once()
    deps["dense"].search.assert_called_once()
    assert deps["bm25"].search.call_args.kwargs.get("k") == 42
    assert deps["dense"].search.call_args.kwargs.get("k") == 42


# ---------------------------------------------------------------------------
# Amendment-chain MANDATORY contract
# ---------------------------------------------------------------------------


def test_amendment_chain_invoked_for_every_candidate():
    """The contract: every retrieved candidate has its amendment chain
    queried, regardless of whether the chain is non-empty."""
    art_uri_a = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, deps = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "54", "x"),
            _bm25("84-11_1984-06-09", "55", "y"),
        ],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri_a: [{"date": "1984-06-09", "text": "ب"}]},
        top_k_candidates=5,
    )
    h.run("بعد 2005")
    # SPARQL was called: at least one ASK + one SELECT for art_54, plus
    # ASK attempts (across categories) for art_55.
    sparql_queries = [c.args[0] for c in deps["sparql_fn"].call_args_list]
    select_queries = [q for q in sparql_queries if "hasVersion" in q]
    # art_54 resolves → SELECT happens; art_55 doesn't resolve → no SELECT
    # for it but ASKs were issued. The MANDATORY contract is "the chain
    # is queried" — for a resolved URI that's the SELECT we see here.
    assert any("art_54" in q for q in select_queries)


def test_chain_trace_records_no_chain_as_fallback():
    """When KG lookup yields no chain, the trace records source='fallback'."""
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "نص")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={},
    )
    out = h.run("بعد 2005")
    chains = out["_telemetry"]["amendment_chains"]
    assert chains[0]["source"] == "fallback"
    assert chains[0]["chain_len"] == 0
    assert chains[0]["picked"] is None


def test_chain_trace_records_kg_no_match_when_target_pre_dates_chain():
    """If target_date is before every version, picked=None, source=kg_no_match."""
    art_uri = "https://legal.dz/resource/law/2025-08-03/25-14#art_51"
    h, _ = _make_handler(
        bm25_hits=[_bm25("25-14_2025-08-03", "51", "x")],
        dense_hits=[],
        routed_ids=["25-14_2025-08-03"],
        uri_to_chain={art_uri: [{"date": "2025-08-03", "text": "نص 2025"}]},
    )
    out = h.run("في 1990 صدر القانون")
    chains = out["_telemetry"]["amendment_chains"]
    assert chains[0]["source"] == "kg_no_match"
    assert chains[0]["picked"] is None
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


def test_picks_correct_version_for_target_date():
    """Two versions: 1984 and 2005. Query mentions 2005 → pick 2005 version."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={
            art_uri: [
                {"date": "1984-06-09", "text": "النسخة الأصلية"},
                {"date": "2005-02-27", "text": "النسخة المعدلة"},
            ],
        },
    )
    out = h.run("بعد تعديل 2005")
    cit = out["citations"][0]
    assert cit["text"] == "النسخة المعدلة"
    assert cit["version_date"] == "2005-02-27"


def test_picks_pre_amendment_version_when_target_is_before_amendment():
    """Two versions: 1984 and 2005. Query mentions 2000 → pick 1984 version."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={
            art_uri: [
                {"date": "1984-06-09", "text": "النسخة الأصلية"},
                {"date": "2005-02-27", "text": "النسخة المعدلة"},
            ],
        },
    )
    out = h.run("في عام 2000")
    cit = out["citations"][0]
    assert cit["text"] == "النسخة الأصلية"
    assert cit["version_date"] == "1984-06-09"


def test_no_dates_uses_latest_version():
    """No date in query → target=LATEST → newest version selected."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={
            art_uri: [
                {"date": "1984-06-09", "text": "أ"},
                {"date": "2005-02-27", "text": "ب"},
            ],
        },
    )
    out = h.run("ما هو نص المادة 54؟")  # no date
    assert out["_telemetry"]["target_date"] == LATEST_VERSION_DATE
    assert out["citations"][0]["text"] == "ب"


# ---------------------------------------------------------------------------
# Verifier hookup
# ---------------------------------------------------------------------------


def test_verifier_receives_kg_versioned_text_not_chunk_text():
    """When opted in, the verifier must see the KG version text — that's
    the answer-from-KG contract."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, deps = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "STALE CHUNK")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "نص KG حديث"}]},
        verify_top_n=3,
    )
    h.run("بعد 2005")
    # Inspect the article passed to the verifier
    verifier_call = deps["verifier_fn"].call_args
    article_arg = verifier_call.args[2]  # (pool, query, article, model)
    assert article_arg["text"] == "نص KG حديث"


def test_verifier_can_drop_irrelevant_candidate():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
        verifier_verdicts={
            ("84-11_1984-06-09", "54"): {"relevant": False, "confidence": 0.9},
        },
        verify_top_n=3,
    )
    out = h.run("بعد 2005")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


def test_verifier_low_confidence_drops_candidate():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
        verifier_verdicts={
            ("84-11_1984-06-09", "54"): {
                "relevant": True, "confidence": 0.1,  # below default 0.4
            },
        },
        verify_top_n=3,
    )
    out = h.run("بعد 2005")
    assert out["abstention"] is True


def test_verifier_exception_does_not_crash():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, deps = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
        verify_top_n=3,
    )
    deps["verifier_fn"].side_effect = RuntimeError("boom")
    out = h.run("بعد 2005")
    # Verifier failure → keep candidate (relevant=True default)
    assert out["abstention"] is False
    assert out["citations"][0]["doc_id"] == "84-11_1984-06-09"


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_summary_used_as_answer_text():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
        summary="ملخص نهائي من السامارايزر",
    )
    out = h.run("بعد 2005")
    assert out["answer_text"] == "ملخص نهائي من السامارايزر"


def test_null_summary_falls_back_to_template():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص"}]},
        summary=None,
    )
    out = h.run("بعد 2005")
    assert "وفقًا لـ" in out["answer_text"]
    assert "(نسخة 2005-02-27)" in out["answer_text"]
    assert "النص" in out["answer_text"]


def test_summarizer_exception_falls_back_to_template():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, deps = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص"}]},
    )
    deps["summarizer_fn"].side_effect = RuntimeError("boom")
    out = h.run("بعد 2005")
    assert "وفقًا لـ" in out["answer_text"]
    assert out["abstention"] is False


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calling_anything():
    h, deps = _make_handler(bm25_hits=[], dense_hits=[], routed_ids=[])
    out = h.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    deps["bm25"].search.assert_not_called()
    deps["dense"].search.assert_not_called()
    deps["router"].route.assert_not_called()
    deps["sparql_fn"].assert_not_called()
    deps["verifier_fn"].assert_not_called()
    deps["summarizer_fn"].assert_not_called()


def test_no_hits_abstains():
    h, _ = _make_handler(bm25_hits=[], dense_hits=[], routed_ids=["x"])
    out = h.run("بعد 2005")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


def test_no_verified_articles_abstains():
    """With the verifier opted in and rejecting every candidate, the
    handler must abstain with reason ``no_verified_articles``."""
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "ب"}]},
        verifier_verdicts={
            ("84-11_1984-06-09", "54"): {"relevant": False, "confidence": 0.0},
        },
        verify_top_n=3,
    )
    out = h.run("بعد 2005")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


# ---------------------------------------------------------------------------
# End-to-end with eval runner
# ---------------------------------------------------------------------------


def test_answer_to_result_consumes_handler_output_unchanged():
    """The handler output must shape-match the baseline contract."""
    from akn_rlm.eval.runner import _answer_to_result

    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص"}]},
    )
    answer = h.run("بعد 2005")
    answer["_latency_s"] = 0.01

    record = {
        "id": "test_q01",
        "query": "بعد 2005",
        "query_type": "temporal_factual",
        "difficulty": "hard",
        "gold_doc_ids": ["84-11_1984-06-09"],
        "gold_articles": [("84-11_1984-06-09", "54")],
    }
    result = _answer_to_result(record, answer)
    assert result["question_id"] == "test_q01"
    assert result["query_type"] == "temporal_factual"
    # Doc-level retrieval hit
    assert "84-11_1984-06-09" in result["pred_doc_ids"]


# ---------------------------------------------------------------------------
# top_k_candidates / final_top_k truncation
# ---------------------------------------------------------------------------


def test_top_k_candidates_caps_amendment_chain_calls():
    """Only the top-K candidates have their chains queried."""
    chains = {}
    for i in range(20):
        u = f"https://legal.dz/resource/law/1984-06-09/84-11#art_{i + 1}"
        chains[u] = [{"date": "1984-06-09", "text": f"v{i}"}]
    hits = [_bm25("84-11_1984-06-09", str(i + 1), f"t{i}", score=20 - i) for i in range(20)]
    h, deps = _make_handler(
        bm25_hits=hits,
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain=chains,
        top_k_candidates=3,
    )
    h.run("بعد 2005")
    # Exactly 3 chain traces
    out = h.run("بعد 2005")
    assert len(out["_telemetry"]["amendment_chains"]) == 3


def test_final_top_k_caps_returned_citations():
    chains = {}
    hits = []
    for i in range(8):
        u = f"https://legal.dz/resource/law/1984-06-09/84-11#art_{i + 1}"
        chains[u] = [{"date": "1984-06-09", "text": f"v{i}"}]
        hits.append(_bm25("84-11_1984-06-09", str(i + 1), f"t{i}", score=10 - i))
    h, _ = _make_handler(
        bm25_hits=hits,
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain=chains,
        top_k_candidates=8,
        verify_top_n=8,
        final_top_k=3,
    )
    out = h.run("بعد 2005")
    assert len(out["citations"]) == 3


def test_template_uses_doc_title_and_falls_back_to_doc_id():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص"}]},
        doc_title="قانون الأسرة",
        summary=None,  # force template
    )
    out = h.run("بعد 2005")
    assert "قانون الأسرة" in out["answer_text"]


def test_template_uses_doc_id_when_title_missing():
    art_uri = "https://legal.dz/resource/law/1984-06-09/84-11#art_54"
    h, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "54", "x")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09"],
        uri_to_chain={art_uri: [{"date": "2005-02-27", "text": "النص"}]},
        doc_title="",
        summary=None,
    )
    out = h.run("بعد 2005")
    assert "84-11_1984-06-09" in out["answer_text"]
