"""Faithfulness gate: sentence-level entailment check of the answer vs cited articles.

Strategy:
  1. Split answer_text into claims (sentence boundaries).
  2. For each claim: check if at least one cited article text entails it.
     Uses MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual — handles Arabic).
  3. Gate passes if ≥ SUPPORT_THRESHOLD fraction of claims are supported.
  4. Falls back to a sub-LM call when the NLI model is unavailable or scores
     are in the ambiguous zone (0.3 < score < CLAIM_THRESHOLD).

Gate is silently skipped (returns passed=True) when:
  - No citations present (abstention answers).
  - NLI model could not be loaded AND no llm_pool provided.
"""
from __future__ import annotations

from importlib import metadata
import logging
import re

from akn_rlm.gates.base import GateResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLI model (lazy, module-level singleton)
# ---------------------------------------------------------------------------

_NLI_MODEL = None
_NLI_MODEL_FAILED = False
_NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
_NLI_ENTAILMENT_IDX: int | None = None   # resolved on first use


def _has_known_abi_issue() -> bool:
    """Avoid importing heavy ML stacks when NumPy/SciPy are known-incompatible."""
    try:
        numpy_v = metadata.version("numpy")
        scipy_v = metadata.version("scipy")
    except metadata.PackageNotFoundError:
        return False
    except Exception:
        return False

    def _major_minor(version: str) -> tuple[int, int]:
        parts = version.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return major, minor

    np_major, _ = _major_minor(numpy_v)
    sp_major, sp_minor = _major_minor(scipy_v)
    return np_major >= 2 and sp_major == 1 and sp_minor < 14


def _get_model():
    """Return loaded CrossEncoder or None on failure."""
    global _NLI_MODEL, _NLI_MODEL_FAILED
    if _NLI_MODEL_FAILED:
        return None
    if _NLI_MODEL is None:
        if _has_known_abi_issue():
            log.warning(
                "Skipping NLI model load because the installed NumPy/SciPy stack is ABI-incompatible"
            )
            _NLI_MODEL_FAILED = True
            return None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            log.info("Loading NLI model: %s", _NLI_MODEL_NAME)
            _NLI_MODEL = CrossEncoder(_NLI_MODEL_NAME)
        except Exception as exc:
            log.warning("Could not load NLI model %s: %s", _NLI_MODEL_NAME, exc)
            _NLI_MODEL_FAILED = True
    return _NLI_MODEL


def _get_entailment_idx(model) -> int:
    """Find which output index corresponds to 'entailment' for this model."""
    global _NLI_ENTAILMENT_IDX
    if _NLI_ENTAILMENT_IDX is not None:
        return _NLI_ENTAILMENT_IDX
    try:
        id2label = model.config.id2label
        for idx, label in id2label.items():
            if "entail" in label.lower():
                _NLI_ENTAILMENT_IDX = int(idx)
                return _NLI_ENTAILMENT_IDX
    except Exception:
        pass
    _NLI_ENTAILMENT_IDX = 0  # mDeBERTa-v3-base-mnli-xnli default: [entailment, neutral, contradiction]
    return _NLI_ENTAILMENT_IDX


# ---------------------------------------------------------------------------
# Sentence / claim splitter
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"[.!?؟。\n]+\s*")
_MIN_CLAIM_LEN = 15   # characters; shorter fragments are skipped


def _split_claims(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_CLAIM_LEN]


# ---------------------------------------------------------------------------
# Per-claim NLI scoring
# ---------------------------------------------------------------------------

def entailment_score(premise: str, hypothesis: str) -> float:
    """Return probability that premise entails hypothesis (0..1).

    Returns 0.5 (neutral) if model unavailable or scoring fails.
    """
    model = _get_model()
    if model is None:
        return 0.5

    try:
        import numpy as np  # type: ignore
        # ``show_progress_bar=False`` — single-pair scoring doesn't need
        # the tqdm bar, and the bar interacts badly with stdout capture
        # (wandb / pytest) producing spurious "I/O operation on closed
        # file" errors that degrade scoring to the 0.5 fallback.
        scores = model.predict(
            [[premise, hypothesis]],
            show_progress_bar=False,
        )
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        idx = _get_entailment_idx(model)
        return float(probs[0][idx])
    except Exception as exc:
        log.error("NLI scoring failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_LLM_FAITHFULNESS_PROMPT = """\
Does the following article text SUPPORT, CONTRADICT, or is NEUTRAL to the claim?

Article text:
{article_text}

Claim:
{claim}

Answer with exactly one word: SUPPORT, NEUTRAL, or CONTRADICT."""


def _llm_support_check(claim: str, article_text: str, llm_pool, model: str) -> bool:
    """Return True if LLM says the article supports the claim."""
    prompt = _LLM_FAITHFULNESS_PROMPT.format(
        article_text=article_text[:1500],
        claim=claim[:500],
    )
    try:
        raw = llm_pool.call(prompt, model=model, max_tokens=10, temperature=0.0)
        return "support" in raw.lower()
    except Exception as exc:
        log.warning("LLM faithfulness check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------

CLAIM_THRESHOLD   = 0.5    # entailment score ≥ this → claim is supported by NLI
LLM_FALLBACK_MIN  = 0.3    # score in [LLM_FALLBACK_MIN, CLAIM_THRESHOLD) → try LLM
# R8: lowered from 0.80 → 0.55 to match per-citation NLI ("at least one cited
# article entails this claim"). 0.80 was too strict for legal Arabic where
# definitional articles often state premises that the answer paraphrases —
# semantic entailment is a noisy signal. The gate is now record-only on
# failure (pipeline.py does not retry on faithfulness alone), so the
# threshold acts as a quality flag in telemetry, not a hard gate.
SUPPORT_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Legacy helper (backwards compat)
# ---------------------------------------------------------------------------

def check_faithfulness(
    answer_text: str,
    citations: list[dict],
    threshold: float = 0.3,
) -> tuple[bool, float]:
    """Return (faithful, max_entailment_score). Backwards-compatible API."""
    if not citations:
        return False, 0.0
    best = 0.0
    for c in citations:
        text = c.get("text", "") or c.get("supporting_span", "")
        if not text:
            continue
        score = entailment_score(text, answer_text)
        if score > best:
            best = score
    return best >= threshold, best


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def _citation_to_article_text(citation: dict) -> str:
    """Extract usable article text from a citation dict."""
    return (
        citation.get("text", "")
        or citation.get("supporting_span", "")
        or citation.get("article_text", "")
    )


def _citation_key(citation: dict) -> str:
    """Stable key for a citation (doc_id + article_ref)."""
    return f"{citation.get('doc_id', '')}#{citation.get('article_ref', '')}"


def run_gate(
    answer_text: str,
    citations: list[dict],
    *,
    llm_pool=None,
    model: str = "Qwen3-30B-A3B-Thinking",
    support_threshold: float = SUPPORT_THRESHOLD,
    claim_threshold: float = CLAIM_THRESHOLD,
) -> GateResult:
    """Run per-citation faithfulness gate.

    Strategy (per-citation NLI):
      1. Extract ADU claims from the answer (sentence splitter).
      2. For each claim, identify which citation it most plausibly belongs to
         by finding the citation whose article text gives the highest NLI score.
      3. A claim is considered supported only if the NLI score against ITS OWN
         best-matching citation meets claim_threshold — not pooled across all
         citations (which would let one strong article cover another's claims).
      4. Gate passes if ≥ support_threshold fraction of claims are supported.

    Args:
        answer_text       : the model's answer string.
        citations         : list of citation dicts (need "text"/"supporting_span").
        llm_pool          : optional LLMPool for ambiguous-zone fallback.
        model             : model to use for LLM fallback.
        support_threshold : fraction of claims that must be supported (default 0.80).
        claim_threshold   : NLI score ≥ this counts as supported (default 0.50).
    """
    if not citations:
        return GateResult(passed=True, score=1.0, details=[{
            "note": "no_citations_faithfulness_skipped"
        }])

    nli_available = _get_model() is not None
    if not nli_available and llm_pool is None:
        log.warning("Faithfulness gate skipped: NLI model unavailable and no llm_pool")
        return GateResult(passed=True, score=0.5, details=[{
            "note": "nli_model_unavailable_skipped"
        }])

    # Build per-citation article text map (only citations with usable text)
    cit_texts: list[tuple[str, str]] = []   # [(cit_key, article_text), ...]
    for c in citations:
        art_text = _citation_to_article_text(c)
        if art_text:
            cit_texts.append((_citation_key(c), art_text))

    if not cit_texts:
        return GateResult(passed=True, score=1.0, details=[{
            "note": "no_article_texts_available_skipped"
        }])

    claims = _split_claims(answer_text)
    if not claims:
        return GateResult(passed=True, score=1.0, details=[{
            "note": "no_claims_extracted_skipped"
        }])

    # Per-citation NLI: each claim is checked against all citations but must
    # meet threshold against the ONE citation it scores highest on.
    unsupported: list[dict] = []
    for claim in claims:
        # Find the citation that best supports this specific claim
        best_score = 0.0
        best_cit_key = ""
        for cit_key, art_text in cit_texts:
            score = entailment_score(art_text, claim)
            if score > best_score:
                best_score = score
                best_cit_key = cit_key

        # Ambiguous zone → LLM fallback against the best-matching citation only
        if best_score < claim_threshold and best_score >= LLM_FALLBACK_MIN and llm_pool is not None:
            # Use the best-match citation's text for the LLM check
            best_art_text = next(
                (txt for key, txt in cit_texts if key == best_cit_key), ""
            )
            if best_art_text and _llm_support_check(claim, best_art_text, llm_pool, model):
                continue

        if best_score < claim_threshold:
            unsupported.append({
                "claim": claim,
                "best_cit": best_cit_key,
                "best_score": round(best_score, 3),
                "reason": "not_supported_by_attributed_citation",
            })

    n_claims = len(claims)
    n_supported = n_claims - len(unsupported)
    coverage = n_supported / n_claims

    if coverage >= support_threshold:
        return GateResult(
            passed=True,
            score=coverage,
            details=[{"note": f"{n_supported}/{n_claims} claims supported (per-citation NLI)"}],
        )

    log.warning(
        "Faithfulness gate failed: %d/%d claims unsupported (threshold %.0f%%)",
        len(unsupported), n_claims, support_threshold * 100,
    )
    return GateResult(
        passed=False,
        score=coverage,
        details=unsupported,
    )
