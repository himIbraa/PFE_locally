"""Production API for AKN-RLM.

The :mod:`akn_rlm.api.answer` module exposes a single user-facing
function :func:`answer_query` that the UI calls.  It wraps the
deployable Phase E SOTA pipeline (classifier-typed dispatch, selective
HyDE, pervasive Toulmin ADU, gap-driven recursion, corrective retry,
``recursion_coverage_min=4`` for MH/RA) and returns:

  * the synthesised Arabic answer,
  * a list of citations with Toulmin argumentation and version dates,
  * a human-readable trajectory ("explainability"),
  * a formatted reference list ready for display.

The KG-CONTAINS retrieval channels (E5 / E6 / E7) shipped behind
feature flags in Phase E are **NOT** used by this endpoint: the
2026-05-14 benchmark showed they slightly hurt Cite F1 and add
substantial latency (E.7 alone: 17.5 ms → 7708 ms per query for the
doc-router). They remain in the codebase only for the thesis ablation
table.
"""

from akn_rlm.api.answer import (
    AnswerResponse,
    Citation,
    TrajectoryStep,
    answer_query,
    get_dispatcher,
    reset_dispatcher,
)

__all__ = [
    "AnswerResponse",
    "Citation",
    "TrajectoryStep",
    "answer_query",
    "get_dispatcher",
    "reset_dispatcher",
]
