"""Genuine recursion — Phase D.

Adds depth-2+ gap-driven re-retrieval to the typed handlers. Justifies
the *Recursive* in *Recursive Language Model* by replacing the F5
single-shot retrieve-and-verify pass with an iterative loop:

  1. Run the handler's normal retrieve+verify pass at depth=1.
  2. Decide whether the *result so far* covers the question
     (:func:`_should_recurse`). The decision is hybrid:
       a. cheap pre-check on the accumulator size + top-citation
          confidence — recurse early when retrieval is genuinely thin;
       b. if neither thin nor strong, fire a single ``gpt-oss-120b``
          probe that returns *both* the binary recurse/stop signal AND a
          gap sub-question in one call (saves a round-trip vs. the
          two-call design in the Phase D spec).
  3. On recurse, **additively merge** the new candidates into the
     accumulator. Never replace — the prior failure mode for restrict-
     style supervisors (R9.6) was narrowing the candidate pool.
  4. Stop at ``max_depth`` (default 3) or when the probe says "covered".

The module is handler-agnostic: each handler hands in two closures —

  * ``retrieve_verify_fn(query) -> dict[(doc, ref) -> citation]``
    runs whatever retrieval+verification step the handler already does
    for ONE query string. The handler decides what "verify" means
    (Qwen verifier for RA/MH, KG amendment chain for TF, ADU+KG-bias
    for CD).

  * ``identify_gap_fn(original_query, accumulator) -> tuple[bool, str]``
    returns ``(should_recurse, gap_sub_question)``. Closure builds
    whatever prompt the handler wants over its own citation shape;
    failure must return ``(False, "")``.

Telemetry: ``RecursiveRetriever.run`` returns a list of
:class:`RecursionStep` records that the caller stamps into
``_telemetry["recursion_trace"]`` and into the per-answer
``trajectory`` list. ``depth_max_reached`` is the max depth the
handler reached.

Fail-open semantics throughout: any exception inside the helpers
collapses recursion to depth 1, which is the F5 baseline behaviour.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from akn_rlm.rlm.sub_worker import parse_strict_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Probe model. gpt-oss-120b is the strongest model in the AI Grid pool
#: and the same one the R9.5 supervisor uses; reusing it amortises the
#: sub-LM-call envelope review (we already account for this model's
#: latency in HANDOFF §1).
DEFAULT_PROBE_MODEL: str = "gpt-oss-120b"

#: Maximum recursion depth. 3 = original + 2 re-retrieval passes; 4+ adds
#: latency without empirical lift on the 244-q benchmark (each extra
#: depth at most surfaces 1-2 new gold articles). Phase D spec asks for
#: depth-2; we cap at 3 so tests can verify the cap works.
DEFAULT_MAX_DEPTH: int = 3

#: Cheap pre-check thresholds.
#: - ``coverage_min``: fewer accumulated citations than this triggers
#:   the probe. 2 catches the documented "got the right doc but not
#:   enough articles" failure mode (multi-hop especially).
#: - ``confidence_weak``: top-citation confidence below this ALSO
#:   triggers the probe even when ``coverage_min`` is met — covers the
#:   case where retrieval surfaced a wide pool of low-confidence
#:   candidates (RA's typical failure mode).
#: - ``confidence_strong``: top-citation confidence above this short-
#:   circuits to "no recurse" without firing the probe — saves the
#:   gpt-oss-120b call when we're already confident.
DEFAULT_COVERAGE_MIN: int = 2
DEFAULT_CONFIDENCE_WEAK: float = 0.55
DEFAULT_CONFIDENCE_STRONG: float = 0.90


# ---------------------------------------------------------------------------
# Telemetry record
# ---------------------------------------------------------------------------

@dataclass
class RecursionStep:
    """One iteration of the recursive loop.

    Attributes:
        depth: 1-indexed; depth=1 is the original handler retrieve pass.
        sub_question: the query string used for this depth's retrieval
            (depth=1 == the original query; depth>1 == a gap question).
        pre_count: accumulator size BEFORE this step ran.
        post_count: accumulator size AFTER this step's merge.
        new_citations: unique (doc, ref) keys added this step.
        top_confidence_pre: max citation confidence in accumulator
            before the step (0.0 when empty).
        top_confidence_post: max citation confidence after the merge.
        gap_decision: one of ``"strong_skip"``, ``"covered"``,
            ``"thin_force"``, ``"weak_probe_yes"``, ``"weak_probe_no"``,
            ``"max_depth"``, or ``"depth_1"`` (initial pass).
        sub_call_count: sub-LM calls billed *for this step*
            (probe + gap-question generation; retrieve_verify_fn is the
            handler's own budget and is counted there).
    """
    depth: int
    sub_question: str
    pre_count: int = 0
    post_count: int = 0
    new_citations: int = 0
    top_confidence_pre: float = 0.0
    top_confidence_post: float = 0.0
    gap_decision: str = "depth_1"
    sub_call_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "sub_question": self.sub_question,
            "pre_count": self.pre_count,
            "post_count": self.post_count,
            "new_citations": self.new_citations,
            "top_confidence_pre": round(self.top_confidence_pre, 3),
            "top_confidence_post": round(self.top_confidence_post, 3),
            "gap_decision": self.gap_decision,
            "sub_call_count": self.sub_call_count,
        }


# ---------------------------------------------------------------------------
# Combined gap probe (gpt-oss-120b)
# ---------------------------------------------------------------------------

#: Schema we ask the probe to return. We bundle the recurse/stop
#: decision with the gap sub-question to save one round-trip.
_GAP_PROBE_PROMPT = """\
You are a legal-research recursion controller. Given an Algerian-law \
QUESTION and a list of CANDIDATE articles already retrieved, decide:

  1. Are these candidates *together* sufficient to answer the QUESTION?
  2. If NOT sufficient, what specific aspect is still missing — phrase \
it as a single short sub-question in Modern Standard Arabic that, if \
answered by ONE more article, would close the gap.

Return ONLY a JSON object exactly of the form:
{{"sufficient": true}}
or
{{"sufficient": false, "gap_question": "..."}}

Do not add prose, comments, or markdown.

QUESTION:
{question}

CANDIDATES (each: [doc_id art.ref] confidence: ... text excerpt):
{candidate_block}

Return only the JSON object.
"""


_MAX_PROBE_TEXT_CHARS: int = 800


def _build_candidate_block(citations: list[dict[str, Any]]) -> str:
    """Compact, bounded representation of the accumulator for the probe."""
    lines: list[str] = []
    for i, c in enumerate(citations):
        doc = str(c.get("doc_id", ""))
        ref = str(c.get("article_ref", ""))
        conf = float(c.get("confidence", 0.0) or 0.0)
        text = (c.get("supporting_span") or c.get("text") or "")[:_MAX_PROBE_TEXT_CHARS]
        lines.append(f"[{i}] [{doc} art.{ref}] conf={conf:.2f}\n{text}")
    return "\n\n".join(lines) if lines else "(none)"


def call_gap_probe(
    llm_pool,
    query: str,
    citations: list[dict[str, Any]],
    *,
    model: str = DEFAULT_PROBE_MODEL,
) -> tuple[bool, str]:
    """Ask the strong model whether the accumulator covers the question.

    Returns ``(should_recurse, gap_sub_question)``. ``should_recurse``
    is True only when the probe explicitly says ``sufficient=false``
    AND returns a non-empty ``gap_question`` — anything else (parse
    failure, JSON missing keys, exception) is treated as ``sufficient``
    so a flaky probe never *forces* extra recursion.
    """
    if not isinstance(query, str) or not query.strip():
        return False, ""

    block = _build_candidate_block(citations)
    prompt = _GAP_PROBE_PROMPT.format(
        question=query.strip(),
        candidate_block=block,
    )

    try:
        raw = llm_pool.call(prompt, model=model, max_tokens=384, temperature=0.0)
    except Exception as exc:
        log.warning("recursion gap probe failed (%s) — assuming sufficient", exc)
        return False, ""

    parsed = parse_strict_json(raw, default={})
    if not isinstance(parsed, dict):
        return False, ""

    sufficient = parsed.get("sufficient")
    if sufficient is True:
        return False, ""

    gap = (parsed.get("gap_question") or "").strip()
    if not gap:
        # Probe said "not sufficient" but gave no actionable gap — treat
        # as covered so we don't recurse aimlessly.
        return False, ""
    return True, gap


# ---------------------------------------------------------------------------
# Decision logic (cheap pre-check + probe)
# ---------------------------------------------------------------------------

def _top_confidence(citations: list[dict[str, Any]]) -> float:
    if not citations:
        return 0.0
    best = 0.0
    for c in citations:
        conf = float(c.get("confidence", 0.0) or 0.0)
        if conf > best:
            best = conf
    return best


def _should_recurse(
    query: str,
    citations: list[dict[str, Any]],
    *,
    llm_pool,
    coverage_min: int,
    confidence_weak: float,
    confidence_strong: float,
    probe_fn: Callable[..., tuple[bool, str]],
    probe_model: str,
) -> tuple[bool, str, str, int]:
    """Return ``(should_recurse, gap_question, decision_label, sub_calls)``.

    Decision tree:
      - **strong_skip**: top conf >= confidence_strong  → no recurse.
      - **thin_force**: len(citations) < coverage_min AND probe -> gap;
        if probe declines, we still skip (no point recursing without a
        gap question).
      - **weak_probe**: top conf < confidence_weak → fire probe.
      - otherwise (mid-band): fire probe.
      - **probe**: returns (yes, gap) or (no, "").
    """
    n = len(citations)
    top_conf = _top_confidence(citations)

    if top_conf >= confidence_strong and n >= coverage_min:
        return False, "", "strong_skip", 0

    # Thin retrieval: explicitly probe for a gap question, but the
    # *decision* to recurse already favours yes — we just need a
    # sensible sub-question to retrieve on.
    decision_label = "thin_force" if n < coverage_min else "weak_probe"
    if n >= coverage_min and top_conf >= confidence_weak:
        # Mid-band: not strong, not thin — let the probe decide.
        decision_label = "weak_probe"

    try:
        recurse, gap = probe_fn(llm_pool, query, citations, model=probe_model)
    except Exception as exc:
        log.warning("recursion probe wrapper raised (%s) — skipping recursion", exc)
        return False, "", f"{decision_label}_error", 1

    sub_calls = 1
    if recurse and gap:
        return True, gap, f"{decision_label}_yes", sub_calls
    return False, "", f"{decision_label}_no", sub_calls


# ---------------------------------------------------------------------------
# Recursive retriever
# ---------------------------------------------------------------------------

CitationKey = tuple[str, str]
RetrieveVerifyFn = Callable[[str], dict[CitationKey, dict[str, Any]]]
GapProbeFn = Callable[..., tuple[bool, str]]


@dataclass
class RecursiveRetriever:
    """Iterate retrieve+verify until covered or ``max_depth`` reached.

    ``retrieve_verify_fn`` is the handler's own pipeline step. It MUST
    accept a single query string and return a ``{(doc, ref): citation}``
    dict shaped like the handler's existing ``accumulated`` accumulator.
    Re-running with the same query string is a no-op (the merge will
    add zero new citations) — handlers should make sure each depth's
    sub-question is genuinely different.

    ``depth1_fn`` is an optional override for the first pass. Useful
    for handlers like multi_hop whose depth-1 is "decompose +
    per-sub-question sweep" but whose depth>1 is "one gap sub-question".
    Defaults to ``retrieve_verify_fn`` so RA / TF / CD don't have to
    pass it.

    Caller-injected ``probe_fn`` defaults to :func:`call_gap_probe`;
    tests pass a stub.

    ``seed_accumulator`` lets a caller supply a pre-built depth-1
    accumulator (e.g. multi_hop's decomposition output) so the loop
    skips its own depth-1 retrieve. When set, ``depth1_fn`` is ignored.
    """
    llm_pool: Any
    retrieve_verify_fn: RetrieveVerifyFn
    max_depth: int = DEFAULT_MAX_DEPTH
    coverage_min: int = DEFAULT_COVERAGE_MIN
    confidence_weak: float = DEFAULT_CONFIDENCE_WEAK
    confidence_strong: float = DEFAULT_CONFIDENCE_STRONG
    probe_fn: GapProbeFn = field(default=call_gap_probe)
    probe_model: str = DEFAULT_PROBE_MODEL
    depth1_fn: Optional[RetrieveVerifyFn] = None
    seed_accumulator: Optional[dict[CitationKey, dict[str, Any]]] = None

    def run(
        self, query: str,
    ) -> tuple[dict[CitationKey, dict[str, Any]], list[RecursionStep], int]:
        """Run retrieve+verify recursively. Returns merged accumulator,
        per-depth telemetry, and total probe sub-LM calls.
        """
        if self.max_depth < 1:
            return {}, [], 0

        # Depth 1 — either a caller-supplied seed accumulator (handler
        # already ran its own decomposition pass) or run the depth-1
        # function (defaults to retrieve_verify_fn).
        accumulator: dict[CitationKey, dict[str, Any]] = {}
        steps: list[RecursionStep] = []
        probe_calls = 0

        seen_questions: set[str] = set()
        seen_questions.add(_normalise_question(query))

        if self.seed_accumulator is not None:
            new_acc = dict(self.seed_accumulator)
        else:
            depth1 = self.depth1_fn or self.retrieve_verify_fn
            try:
                raw = depth1(query)
            except Exception as exc:
                log.warning("recursive depth1_fn raised (%s) — empty step", exc)
                raw = {}
            new_acc = raw if isinstance(raw, dict) else {}
        merged, added = _merge_into(accumulator, new_acc)
        accumulator = merged
        steps.append(RecursionStep(
            depth=1,
            sub_question=query,
            pre_count=0,
            post_count=len(accumulator),
            new_citations=added,
            top_confidence_pre=0.0,
            top_confidence_post=_top_confidence(list(accumulator.values())),
            gap_decision="depth_1",
            sub_call_count=0,
        ))

        if self.max_depth == 1:
            return accumulator, steps, probe_calls

        depth = 1
        while depth < self.max_depth:
            cits_list = list(accumulator.values())
            recurse, gap, decision, calls = _should_recurse(
                query=query,
                citations=cits_list,
                llm_pool=self.llm_pool,
                coverage_min=self.coverage_min,
                confidence_weak=self.confidence_weak,
                confidence_strong=self.confidence_strong,
                probe_fn=self.probe_fn,
                probe_model=self.probe_model,
            )
            probe_calls += calls
            if not recurse:
                # Stamp a zero-action step so the trace records WHY we
                # stopped — useful for telemetry inspection during the
                # gate review (Phase D requires depth>1 on >=30% of MH).
                steps.append(RecursionStep(
                    depth=depth + 1,
                    sub_question="",
                    pre_count=len(accumulator),
                    post_count=len(accumulator),
                    new_citations=0,
                    top_confidence_pre=_top_confidence(cits_list),
                    top_confidence_post=_top_confidence(cits_list),
                    gap_decision=decision,
                    sub_call_count=calls,
                ))
                break

            # Avoid infinite loops on identical gap questions —
            # the probe sometimes recommends paraphrases of the
            # original on edge cases.
            normalised = _normalise_question(gap)
            if normalised in seen_questions:
                steps.append(RecursionStep(
                    depth=depth + 1,
                    sub_question=gap,
                    pre_count=len(accumulator),
                    post_count=len(accumulator),
                    new_citations=0,
                    top_confidence_pre=_top_confidence(cits_list),
                    top_confidence_post=_top_confidence(cits_list),
                    gap_decision=f"{decision}_duplicate",
                    sub_call_count=calls,
                ))
                break
            seen_questions.add(normalised)

            depth += 1
            pre_count = len(accumulator)
            top_pre = _top_confidence(cits_list)

            new_acc = self._safe_retrieve(gap)
            merged, added = _merge_into(accumulator, new_acc)
            accumulator = merged
            steps.append(RecursionStep(
                depth=depth,
                sub_question=gap,
                pre_count=pre_count,
                post_count=len(accumulator),
                new_citations=added,
                top_confidence_pre=top_pre,
                top_confidence_post=_top_confidence(list(accumulator.values())),
                gap_decision=decision,
                sub_call_count=calls,
            ))

        # If we exited because of max_depth, stamp a marker so the trace
        # records the cap (helps debug "why didn't we recurse further").
        if depth >= self.max_depth and steps and steps[-1].depth == self.max_depth:
            steps[-1].gap_decision = f"{steps[-1].gap_decision}_at_max_depth"

        return accumulator, steps, probe_calls

    # ------------------------------------------------------------------
    def _safe_retrieve(self, query: str) -> dict[CitationKey, dict[str, Any]]:
        try:
            result = self.retrieve_verify_fn(query)
        except Exception as exc:
            log.warning("recursive retrieve_verify_fn raised (%s) — empty step", exc)
            return {}
        if not isinstance(result, dict):
            log.warning(
                "recursive retrieve_verify_fn returned non-dict %r — empty step",
                type(result).__name__,
            )
            return {}
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_question(text: str) -> str:
    """Stable hash of a question for dedup detection."""
    return hashlib.sha256(
        (text or "").strip().lower().encode("utf-8")
    ).hexdigest()


def _merge_into(
    existing: dict[CitationKey, dict[str, Any]],
    incoming: dict[CitationKey, dict[str, Any]],
) -> tuple[dict[CitationKey, dict[str, Any]], int]:
    """Additive merge: keep all existing keys; for each incoming key
    either add (new) or keep the higher-confidence entry.

    Returns ``(merged, n_new_keys_added)``.
    """
    merged = dict(existing)
    new_keys = 0
    for key, cit in (incoming or {}).items():
        prior = merged.get(key)
        if prior is None:
            merged[key] = dict(cit)
            new_keys += 1
            continue
        new_conf = float(cit.get("confidence", 0.0) or 0.0)
        prior_conf = float(prior.get("confidence", 0.0) or 0.0)
        if new_conf > prior_conf:
            replaced = dict(cit)
            # Preserve already-attached argumentation if the new entry
            # didn't extract one (Phase C extracts ADU AFTER recursion;
            # this path keeps that data flowing).
            if not replaced.get("argumentation") and prior.get("argumentation"):
                replaced["argumentation"] = prior["argumentation"]
            merged[key] = replaced
    return merged, new_keys
