"""Pervasive Argument Mining (Phase C).

Shared helper that ADU-extracts Toulmin structure from a list of
finalised citations and attaches an ``argumentation`` dict to each.

Used by every handler EXCEPT ``unanswerable`` (no citations) and
``conceptual_definitional`` (CD does ADU pre-citation-construction so
the claim+ground can feed into ``supporting_span`` at build time and so
the same extract feeds its KG-bias re-rank).

Returned ``argumentation`` shape::

    {"claim": str, "ground": str, "warrant": str,
     "rebuttal": str, "backing": str}

Citations beyond ``top_n`` receive an empty argumentation dict so the
field is uniformly present and the downstream
:func:`am_faithfulness_score` metric can iterate citations without
key-existence guards.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from akn_rlm.adu import extract as default_adu_extract

log = logging.getLogger(__name__)

#: Default number of citations the pervasive helper ADU-extracts before
#: zero-filling the rest. 5 covers the top of every handler's typical
#: emit list while keeping the per-query sub-LM call budget bounded
#: (handler emit-list × 5 ADU calls).
DEFAULT_ADU_EXTRACT_TOP_N: int = 5

#: Max characters retained in the ``supporting_span`` rebuilt from
#: ``claim + ground``. Matches the per-handler ``SUPPORT_SPAN_LEN``
#: default so span shape stays uniform across handlers.
DEFAULT_ADU_SPAN_LEN: int = 280


AduExtractFn = Callable[..., dict[str, Any]]


def empty_argumentation() -> dict[str, str]:
    """Return a fresh empty Toulmin dict with the canonical key set."""
    return {
        "claim":    "",
        "ground":   "",
        "warrant":  "",
        "rebuttal": "",
        "backing":  "",
    }


def attach_argumentation(
    citations: list[dict[str, Any]],
    llm_pool,
    *,
    sub_model: str,
    top_n: int = DEFAULT_ADU_EXTRACT_TOP_N,
    adu_extract_fn: Optional[AduExtractFn] = None,
    span_len: int = DEFAULT_ADU_SPAN_LEN,
) -> tuple[list[dict[str, Any]], int]:
    """ADU-extract the top-``top_n`` citations and attach ``argumentation``.

    Returns ``(new_citations, sub_call_count)``. Original citation dicts
    are shallow-copied; callers can replace their list reference with
    the returned one without worrying about aliasing.
    """
    if not citations or top_n <= 0:
        out = [dict(c, argumentation=c.get("argumentation") or empty_argumentation())
               for c in citations]
        return out, 0

    fn = adu_extract_fn or default_adu_extract
    sub_calls = 0
    out: list[dict[str, Any]] = []
    for i, cit in enumerate(citations):
        new = dict(cit)
        if i < top_n:
            text = cit.get("text", "") or ""
            if text.strip():
                try:
                    adu = fn(text, llm_pool, model=sub_model)
                    sub_calls += 1
                except Exception as exc:
                    log.warning("pervasive ADU extract failed: %s", exc)
                    adu = {}
            else:
                adu = {}
        else:
            adu = {}

        if not isinstance(adu, dict):
            adu = {}

        claim    = str(adu.get("claim", "")    or "").strip()
        ground   = str(adu.get("ground", "")   or "").strip()
        warrant  = str(adu.get("warrant", "")  or "").strip()
        rebuttal = str(adu.get("rebuttal", "") or "").strip()
        backing  = str(adu.get("backing", "")  or "").strip()

        new["argumentation"] = {
            "claim":    claim,
            "ground":   ground,
            "warrant":  warrant,
            "rebuttal": rebuttal,
            "backing":  backing,
        }

        # Replace supporting_span with claim+ground when both extracted.
        # Empty-out path: keep whatever supporting_span the handler set.
        if claim and ground:
            new["supporting_span"] = f"{claim} {ground}"[:span_len]
        elif claim:
            new["supporting_span"] = claim[:span_len]

        out.append(new)
    return out, sub_calls
