"""Corrective retry on faithfulness failure — Phase D.

Wraps the per-handler answer-assembly step so that a faithfulness-gate
failure triggers ONE regeneration of the summary with explicit
"unsupported claims" feedback. After one retry we accept whatever the
summariser produces — the gate stays a *quality flag* rather than a
hard reject (HCR is the hard contract; faithfulness is a soft signal).

Pipeline:

  1. Handler computes ``answer_text`` via its summariser.
  2. :func:`maybe_corrective_retry` runs
     :func:`akn_rlm.gates.faithfulness_nli.run_gate` on
     ``(answer_text, citations)``.
  3. If ``gate.passed is False`` AND we haven't retried yet:
       a. Build a **feedback prompt** that lists the unsupported claims
          and instructs the summariser to regenerate using ONLY the
          cited articles' text.
       b. Call the summariser with that augmented question.
       c. If the new summary is non-empty, replace ``answer_text``;
          otherwise keep the original (a failed retry must not silently
          clear an answer that at least had content).
  4. Return ``(answer_text, retry_telemetry)``.

The retry never re-runs the gate — that would be a second NLI pass on
every retried answer, doubling the gate cost for marginal value. The
final faithfulness telemetry that lands in ``predictions.jsonl`` is the
*post-retry* gate score, which is added to telemetry by the helper.

Fail-open semantics: any LLM exception, NLI failure, or empty retry
output preserves the pre-retry ``answer_text`` and records the failure
mode in telemetry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from akn_rlm.gates.faithfulness_nli import run_gate as _default_run_gate

log = logging.getLogger(__name__)


def _resolve_default_gate() -> Callable[..., Any]:
    """Resolve the default gate function at *call* time so that tests can
    monkeypatch ``akn_rlm.rlm.corrective_retry.run_gate`` (or
    ``akn_rlm.gates.faithfulness_nli.run_gate``) and have handlers that
    omit the ``gate_fn=`` kwarg pick up the patched version.
    """
    # Prefer a module-level ``run_gate`` symbol if one is set (e.g. by a
    # monkeypatch); otherwise fall back to the import-time default.
    return globals().get("run_gate") or _default_run_gate


# Module-level alias so monkeypatch.setattr(cr_mod, "run_gate", ...) works.
run_gate = _default_run_gate


# ---------------------------------------------------------------------------
# Telemetry record
# ---------------------------------------------------------------------------

@dataclass
class CorrectiveRetryTrace:
    """Outcome of one corrective-retry attempt.

    ``fired`` distinguishes "we evaluated the gate and it was already
    passing" (False) from "we tried to retry but the LLM failed"
    (True with ``retry_succeeded=False``).
    """
    fired: bool = False
    pre_gate_score: float = 1.0
    post_gate_score: float = 1.0
    pre_passed: bool = True
    post_passed: bool = True
    unsupported_claims: list[str] = field(default_factory=list)
    retry_succeeded: bool = False
    sub_call_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "pre_gate_score": round(self.pre_gate_score, 3),
            "post_gate_score": round(self.post_gate_score, 3),
            "pre_passed": self.pre_passed,
            "post_passed": self.post_passed,
            "unsupported_claims": list(self.unsupported_claims),
            "retry_succeeded": self.retry_succeeded,
            "sub_call_count": self.sub_call_count,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Feedback prompt — wraps the summariser
# ---------------------------------------------------------------------------

_FEEDBACK_PREAMBLE_AR = """\
سؤال:
{original_question}

ملاحظات تصحيح: التلخيص السابق قدّم ادعاءات غير مؤيدة بنصوص المواد المستشهدة. \
أعد صياغة الإجابة باستخدام نصوص هذه المواد فقط، ولا تذكر أي معلومة لا تظهر \
صراحةً في النصوص أدناه. الادعاءات غير المؤيدة التي يجب تجنبها أو إعادة \
صياغتها بدقة:

{unsupported_block}

السؤال الأصلي مرة أخرى:
{original_question}
"""


def build_corrective_question(
    original_question: str, unsupported_claims: list[str], *, max_claims: int = 6,
) -> str:
    """Wrap the original question with corrective feedback for the summariser.

    The summariser sees this as the ``question`` field; its prompt
    template otherwise stays untouched (so we don't have to fork
    ``call_summarizer``). The Arabic preamble is what the LLM will
    actually condition on.
    """
    capped = [c.strip() for c in unsupported_claims if c and c.strip()][:max_claims]
    if capped:
        block = "\n".join(f"- {c}" for c in capped)
    else:
        block = "- (لا توجد قائمة محددة، تجنّب أي إضافات خارج النصوص)"
    return _FEEDBACK_PREAMBLE_AR.format(
        original_question=original_question.strip(),
        unsupported_block=block,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

SummarizerFn = Callable[..., dict[str, Any]]


def maybe_corrective_retry(
    *,
    answer_text: str,
    citations: list[dict[str, Any]],
    original_question: str,
    summarizer_fn: SummarizerFn,
    llm_pool,
    sub_model: str,
    enabled: bool = True,
    max_retries: int = 1,
    template_fallback: Optional[str] = None,
    gate_fn: Optional[Callable[..., Any]] = None,
) -> tuple[str, CorrectiveRetryTrace]:
    """Run the faithfulness gate; if it fails, regenerate ONCE with feedback.

    Args:
        answer_text: the summariser's first-pass answer.
        citations: list of citation dicts (already final-ranked).
        original_question: the user's original query string.
        summarizer_fn: the same callable the handler uses for the
            first-pass summary; signature
            ``(llm_pool, question, articles, model) -> {"summary": str|None, ...}``.
        llm_pool: passed straight through to the summariser.
        sub_model: model name forwarded to the summariser.
        enabled: master toggle. When False the function does nothing
            and returns ``(answer_text, fired=False)``.
        max_retries: hard cap on retry attempts. Phase D spec calls for
            1; tests can pass higher to exercise the loop.
        template_fallback: if the retried summary is empty AND this is
            non-empty, use it instead of preserving the pre-retry
            answer. Handlers pass their deterministic Arabic template
            here.
        gate_fn: injectable for tests (defaults to
            :func:`gates.faithfulness_nli.run_gate`).

    Returns:
        ``(final_answer_text, telemetry_trace)``.
    """
    trace = CorrectiveRetryTrace()
    if gate_fn is None:
        gate_fn = _resolve_default_gate()
    if not enabled:
        return answer_text, trace
    if not citations:
        # No citations means the gate skips trivially; nothing to correct.
        return answer_text, trace
    if not isinstance(answer_text, str) or not answer_text.strip():
        # No answer text to grade — nothing to retry against either.
        return answer_text, trace

    try:
        gate_result = gate_fn(answer_text, citations, llm_pool=llm_pool, model=sub_model)
    except Exception as exc:
        log.warning("corrective_retry: pre-gate failed (%s) — skipping", exc)
        trace.error = f"pre_gate_error:{exc}"
        return answer_text, trace

    pre_score = float(getattr(gate_result, "score", 0.0) or 0.0)
    pre_passed = bool(getattr(gate_result, "passed", True))
    trace.pre_gate_score = pre_score
    trace.post_gate_score = pre_score
    trace.pre_passed = pre_passed
    trace.post_passed = pre_passed

    if pre_passed:
        return answer_text, trace

    # Pull the unsupported claim list. ``run_gate`` returns ``details``
    # as a list of dicts on failure, each containing ``"claim"``.
    raw_details = getattr(gate_result, "details", None) or []
    unsupported: list[str] = []
    for d in raw_details:
        if isinstance(d, dict) and isinstance(d.get("claim"), str):
            unsupported.append(d["claim"])
    trace.unsupported_claims = unsupported

    current_answer = answer_text
    for attempt in range(max_retries):
        feedback_q = build_corrective_question(original_question, unsupported)
        try:
            synth = summarizer_fn(
                llm_pool, feedback_q, citations, sub_model,
            )
            trace.sub_call_count += 1
        except Exception as exc:
            log.warning("corrective_retry: retry summariser failed (%s)", exc)
            trace.error = f"summarizer_error:{exc}"
            break

        new_summary = synth.get("summary") if isinstance(synth, dict) else None
        if isinstance(new_summary, str) and new_summary.strip():
            current_answer = new_summary.strip()
            trace.retry_succeeded = True
            trace.fired = True
            break
        # Empty retry — try template fallback if provided, then give up.
        if template_fallback and template_fallback.strip():
            current_answer = template_fallback.strip()
            trace.retry_succeeded = False
            trace.fired = True
        else:
            trace.fired = True
        # Only attempt once unless caller explicitly bumped max_retries.
    else:
        # Exhausted attempts without break — record but keep current answer.
        trace.fired = True

    # Re-run gate on the new answer to record the post-retry score so
    # telemetry shows whether the retry actually moved the needle.
    if trace.retry_succeeded and current_answer != answer_text:
        try:
            post_result = gate_fn(
                current_answer, citations, llm_pool=llm_pool, model=sub_model,
            )
            trace.post_gate_score = float(getattr(post_result, "score", 0.0) or 0.0)
            trace.post_passed = bool(getattr(post_result, "passed", False))
        except Exception as exc:
            log.warning("corrective_retry: post-gate failed (%s)", exc)
            trace.error = f"post_gate_error:{exc}"

    return current_answer, trace
