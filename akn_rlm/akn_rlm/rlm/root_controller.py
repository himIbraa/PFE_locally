"""Root RLM controller.

Executes the generate-execute-observe loop:
  1. Send system prompt + few-shot + query to root LLM.
  2. Parse ```python blocks from response, exec each in REPL namespace.
  3. Feed captured stdout back as the next user turn.
  4. Repeat until ```answer block found or timeout / max_rounds reached.
  5. Apply citation existence gate; return final answer object.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from akn_rlm.config import ROOT_LLM_MODEL, SUB_LLM_MODEL
from akn_rlm.gates import citation_existence, span_existence
from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.recursion_budget import RecursionBudget, RecursionBudgetExceeded

log = logging.getLogger(__name__)

_PROMPTS_DIR  = Path(__file__).parent / "prompts"
_PYTHON_RE    = re.compile(r"```python\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_ANSWER_RE    = re.compile(r"```answer\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)
# Some LLMs emit ```json instead of ```answer when the system prompt is misread
_ANSWER_JSON_RE = re.compile(r"```json\s*\n(\{.*?\})\s*\n?```", re.DOTALL)
_TIMEOUT_DEFAULT = 60.0
_TIMEOUT_MULTIHOP = 180.0
_MAX_ROUNDS = 20

# ---------------------------------------------------------------------------
# Query-type classifier (lightweight keyword rules)
# ---------------------------------------------------------------------------

_TEMPORAL_KW = re.compile(
    r"(?:متى|قبل التعديل|بعد التعديل|النص الأصلي|"
    r"version|amendment|before.*amend|after.*amend|"
    r"تاريخ النفاذ|prior to|as of \d{4}|"
    r"تاريخ.*قانون|سريان.*القانون|النفاذ الزمني)",
    re.IGNORECASE,
)
_TEMPORAL_FACTUAL_KW = re.compile(
    r"(?:منذ متى|تاريخ صدور|سنة.*إصدار|"
    r"متى.*صدر|متى.*نشر|متى.*أُصدر)",
    re.IGNORECASE,
)
_MULTIHOP_KW = re.compile(
    r"(?:و(?:القانون|الق[اأ]نون)|both.*and|"
    r"العلاقة بين|interaction between|combined|"
    r"متعدد القوانين|multi.?law)",
    re.IGNORECASE,
)
# Patterns matching explicit article citations in the query itself
_EXACT_ARTICLE_KW = re.compile(
    r"(?:(?:الم[اآ]دة|art(?:icle)?\.?)\s*\d+|"
    r"نص الم[اآ]دة\s*\d+|"
    r"\bالم[اآ]دة\s+(?:الأولى|الثانية|الثالثة|\d+)\b)",
    re.IGNORECASE | re.UNICODE,
)
_CONCEPTUAL_KW = re.compile(
    r"(?:ما هو تعريف|عرّف|ما المقصود|ماذا يُقصد|"
    r"define|definition of|what is meant by|"
    r"ما مفهوم|مفهوم|ما معنى)",
    re.IGNORECASE,
)
_LAYMAN_KW = re.compile(
    r"(?:بكلمات بسيطة|بشكل مبسط|للمواطن العادي|"
    r"in simple terms|simply|للعموم|باختصار غير قانوني)",
    re.IGNORECASE,
)


def classify_query_type(query: str) -> str:
    """Heuristic query-type classifier.  Returns one of:
      rule_application | exact_article | temporal | temporal_factual |
      multi_hop | conceptual_definitional | layman | unanswerable
    """
    from akn_rlm.gates.jurisdiction import is_infected
    if is_infected(query):
        return "unanswerable"
    if _TEMPORAL_FACTUAL_KW.search(query):
        return "temporal_factual"
    if _TEMPORAL_KW.search(query):
        return "temporal"
    if _MULTIHOP_KW.search(query):
        return "multi_hop"
    if _EXACT_ARTICLE_KW.search(query):
        return "exact_article"
    if _CONCEPTUAL_KW.search(query):
        return "conceptual_definitional"
    if _LAYMAN_KW.search(query):
        return "layman"
    return "rule_application"


# ---------------------------------------------------------------------------
# REPL execution
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "list": list,
    "dict": dict, "str": str, "int": int, "float": float,
    "bool": bool, "isinstance": isinstance, "type": type,
    "enumerate": enumerate, "zip": zip, "sorted": sorted,
    "min": min, "max": max, "sum": sum, "abs": abs,
    "round": round, "__import__": __import__,
}


def _exec_block(code: str, namespace: dict) -> tuple[str, str]:
    """Execute a Python code block; return (stdout, stderr).

    If the block is a single bare expression (e.g. ``env.search_bm25(...)``), it
    is compiled in "single" mode so Python auto-prints the result via displayhook,
    matching interactive-REPL behaviour.  All other code uses "exec" mode.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), \
             contextlib.redirect_stderr(stderr_buf):
            try:
                tree = ast.parse(code, mode="exec")
                # Use "single" only when the block is a single expression statement
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    compiled = compile(code, "<repl>", "single")
                else:
                    compiled = compile(code, "<repl>", "exec")
            except SyntaxError:
                compiled = compile(code, "<repl>", "exec")
            exec(compiled, namespace)  # noqa: S102
    except RecursionBudgetExceeded as exc:
        stderr_buf.write(f"RecursionBudgetExceeded: {exc}\n")
    except Exception as exc:
        stderr_buf.write(f"{type(exc).__name__}: {exc}\n")
    return stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_EMPTY_ANSWER = {
    "answer_text": "",
    "abstention": True,
    "abstention_reason": "out_of_corpus",
    "citations": [],
    "reasoning_chain": [],
    "trajectory": [],
    "tokens_used": 0,
    "depth_max_reached": 0,
}


def _parse_answer_block(text: str) -> dict | None:
    for pattern in (_ANSWER_RE, _ANSWER_JSON_RE):
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            result = json.loads(raw)
            # Require at least answer_text or abstention key to accept as answer
            if "answer_text" in result or "abstention" in result:
                return result
        except json.JSONDecodeError:
            pass
    return None


def _extract_python_blocks(text: str) -> list[str]:
    return _PYTHON_RE.findall(text)


_ENV_CALL_RE = re.compile(r"\benv\.\w+\s*\(|\bprint\s*\(", re.IGNORECASE)


def _try_extract_raw_python(text: str) -> list[str]:
    """Fallback: if the response has no fences but parses as valid Python
    and contains env.* or print() calls, treat the whole response as one code block.
    """
    stripped = text.strip()
    if not stripped or not _ENV_CALL_RE.search(stripped):
        return []
    try:
        import ast
        ast.parse(stripped)
        return [stripped]
    except SyntaxError:
        return []


_TOOL_OUTPUT_RE = re.compile(r"\n\nTool output:\n", re.IGNORECASE)


def _split_few_shot_turn(assistant_turn: str) -> list[tuple[str, str]]:
    """Split a monolithic few-shot assistant turn (with embedded 'Tool output:' sections)
    into a proper multi-turn list of (role, content) pairs.

    Pattern expected in assistant_turn:
      ```python...```       ← assistant turn
      Tool output:          ← becomes a user turn
      <output text>
      ```python...```       ← next assistant turn
      ...
      ```answer...```       ← final assistant turn
    """
    # Split at "Tool output:" boundaries
    segments = _TOOL_OUTPUT_RE.split(assistant_turn)
    result: list[tuple[str, str]] = []
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue
        if i == 0:
            # First segment is always an assistant turn (code blocks)
            result.append(("assistant", seg))
        else:
            # Subsequent segments: everything up to the next code block is tool output,
            # then the code block (and anything after) is the next assistant turn.
            # Find where the next ```python or ```answer block starts
            code_match = re.search(r"```(?:python|answer)", seg)
            if code_match:
                tool_output = seg[:code_match.start()].strip()
                next_assistant = seg[code_match.start():].strip()
                if tool_output:
                    result.append(("user", f"Tool output:\n{tool_output}"))
                if next_assistant:
                    result.append(("assistant", next_assistant))
            else:
                # Entire segment is tool output (no more code blocks)
                result.append(("user", f"Tool output:\n{seg}"))
    return result


# ---------------------------------------------------------------------------
# Root controller
# ---------------------------------------------------------------------------

class RootController:
    """Wraps the root LLM in a generate-execute-observe loop."""

    def __init__(
        self,
        env: LegalEnv,
        system_prompt: str | None = None,
        few_shot: list[dict] | None = None,
        root_model: str = ROOT_LLM_MODEL,
        timeout_default: float = _TIMEOUT_DEFAULT,
        timeout_multihop: float = _TIMEOUT_MULTIHOP,
        max_rounds: int = _MAX_ROUNDS,
    ) -> None:
        self.env = env
        self.system_prompt = system_prompt or self._load_system_prompt()
        self.few_shot = few_shot if few_shot is not None else self._load_few_shot()
        self.root_model = root_model
        self.timeout_default = timeout_default
        self.timeout_multihop = timeout_multihop
        self.max_rounds = max_rounds

    # ------------------------------------------------------------------
    @staticmethod
    def _load_system_prompt() -> str:
        p = _PROMPTS_DIR / "root_system.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    @staticmethod
    def _load_few_shot() -> list[dict]:
        p = _PROMPTS_DIR / "few_shot_trajectories.json"
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        query_type: str | None = None,
    ) -> dict:
        """Execute the RLM loop and return the final answer object."""
        query_type = query_type or classify_query_type(query)
        # long_context and multi_hop get extended timeout; temporal types get default
        _LONG_TYPES = {"multi_hop", "long_context"}
        timeout = self.timeout_multihop if query_type in _LONG_TYPES else self.timeout_default
        start = time.time()

        budget = self.env.budget
        sys_prompt = self.system_prompt.replace(
            "{max_sub_calls}", str(budget.max_sub_calls)
        )

        # Inject corpus manifest so the LLM always knows the canonical doc IDs.
        # This prevents hallucinated IDs like "Constitution_1996" or "Penal_Code_1971".
        corpus_manifest = self._build_corpus_manifest()

        # Build conversation
        messages: list[dict] = [{"role": "system", "content": sys_prompt + corpus_manifest}]
        for ex in self.few_shot:
            qt = ex.get("query_type", "rule_application")
            # Support both old monolithic format (assistant_turn) and new multi-turn format (messages)
            if "messages" in ex:
                # New format: list of {role, content} pairs (already interleaved user/assistant)
                # First message should be the user query — prefix it with query type
                for i, msg in enumerate(ex["messages"]):
                    if i == 0 and msg["role"] == "user":
                        messages.append({"role": "user",
                                         "content": f"[TYPE: {qt}]\n{msg['content']}"})
                    else:
                        messages.append(msg)
            else:
                # Old monolithic format: single assistant turn with embedded Tool output: sections
                # Split on "Tool output:" to reconstruct multi-turn exchange
                raw = ex.get("assistant_turn", "")
                turns = _split_few_shot_turn(raw)
                messages.append({"role": "user",
                                 "content": f"[TYPE: {qt}]\n{ex['query']}"})
                for role, content in turns:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user",
                         "content": f"[TYPE: {query_type}]\n{query}"})

        # REPL namespace (persistent across blocks)
        namespace: dict[str, Any] = {
            "env": self.env,
            "json": json,
            "re": re,
            **_SAFE_BUILTINS,
        }

        trajectory: list[dict] = []
        answer: dict | None = None

        # Bootstrap: pre-execute list_documents + search_hybrid and inject as
        # if the model ran the code.  This gives the LLM real search data even
        # if it resists writing code blocks.
        bootstrap_code = (
            f"print(env.list_documents())\n"
            f"results = env.search_hybrid({query!r}, k=10)\n"
            f"print(results[:5])"
        )
        bootstrap_stdout, bootstrap_stderr = _exec_block(bootstrap_code, namespace)
        if bootstrap_stdout:
            messages.append({
                "role": "assistant",
                "content": f"```python\n{bootstrap_code}\n```",
            })
            messages.append({
                "role": "user",
                "content": f"Tool output:\n{bootstrap_stdout}",
            })
            trajectory.append({"round": -1, "block": 0,
                                "stdout": bootstrap_stdout,
                                "stderr": bootstrap_stderr})
        root_tokens_estimated = 0

        # ------------------------------------------------------------------
        client = self.env.llm_pool._route(self.root_model)

        for round_num in range(self.max_rounds):
            elapsed = time.time() - start
            if elapsed > timeout:
                log.warning("RLM timeout after %.1fs (round %d)", elapsed, round_num)
                break

            # LLM call
            try:
                input_chars = sum(len(m.get("content", "")) for m in messages)
                root_tokens_estimated += input_chars // 4 + 4096
                response_text = client.chat(
                    messages=messages,
                    model=self.root_model,
                    max_tokens=4096,
                    temperature=0.0,
                )
            except Exception as exc:
                log.error("Root LLM call failed (round %d): %s", round_num, exc)
                break

            # Strip <think>...</think> reasoning traces some models emit
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

            trajectory.append({"round": round_num, "assistant": response_text})

            # Check for answer block
            answer = _parse_answer_block(response_text)
            if answer is not None:
                # Reject premature abstention: if the LLM claims abstention but
                # has never called any search function, force it to search first.
                # Check if a real search ran: all search results include 'retriever' key.
                # list_documents() does NOT include 'retriever', so this correctly
                # distinguishes between listing and actually searching.
                _did_search = any(
                    "retriever" in t.get("stdout", "")
                    for t in trajectory
                )
                if not _did_search and round_num < self.max_rounds - 2:
                    # Force search for BOTH premature abstentions AND premature answers
                    # (model answered from knowledge without running any REPL code)
                    if answer.get("abstention"):
                        log.warning("Round %d: premature abstention (no search yet) — forcing search", round_num)
                        msg = (
                            "VIOLATION: You abstained without executing a single search. "
                            "Rule 4 strictly forbids this. Write a ```python search block now."
                        )
                    else:
                        log.warning("Round %d: answer emitted without any search — forcing search", round_num)
                        msg = (
                            "VIOLATION: You answered without running any search. "
                            "You MUST first call env.search_hybrid() or env.search_bm25() "
                            "and verify articles before answering. Write a ```python search block now."
                        )
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": msg})
                    answer = None
                    continue
                break

            # Execute python blocks
            python_blocks = _extract_python_blocks(response_text)
            if not python_blocks:
                # Fallback: model wrote raw Python without fences — auto-wrap it
                raw_blocks = _try_extract_raw_python(response_text)
                if raw_blocks:
                    log.warning("Round %d: auto-wrapping unfenced Python code", round_num)
                    python_blocks = raw_blocks
                else:
                    log.warning("Round %d: no python blocks and no answer block", round_num)
                    # Don't add empty assistant messages — they confuse the model
                    if response_text:
                        messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": (
                        "Your response must use triple-backtick code fences. Example:\n\n"
                        "```python\n"
                        "results = env.search_hybrid('your query', k=10)\n"
                        "print(results[:3])\n"
                        "```\n\n"
                        "Write a ```python code block now, or emit a ```answer block "
                        "if you have verified enough articles."
                    )})
                    continue

            tool_outputs = []
            for i, code in enumerate(python_blocks):
                stdout, stderr = _exec_block(code, namespace)
                block_output = stdout
                if stderr:
                    block_output += f"\n[stderr]: {stderr}"
                tool_outputs.append(block_output)
                trajectory.append({"round": round_num, "block": i, "stdout": stdout, "stderr": stderr})

            combined_output = "\n---\n".join(tool_outputs)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user",
                             "content": f"Tool output:\n{combined_output}"})

        # ------------------------------------------------------------------
        # Post-process answer
        if answer is None:
            answer = dict(_EMPTY_ANSWER)
            answer["abstention_reason"] = "timeout" if time.time() - start > timeout else "no_answer"

        # Apply citation existence gate (article must exist in registry)
        # Set AKN_NO_CITATION_GATE=1 to bypass for pre-gate HCR measurement.
        raw_citations = answer.get("citations", []) or []
        import os as _os
        if _os.getenv("AKN_NO_CITATION_GATE"):
            valid_citations, rejected = raw_citations, []
            log.info("AKN_NO_CITATION_GATE: gate bypassed, %d citations pass through", len(valid_citations))
        else:
            valid_citations, rejected = citation_existence.filter_citations(
                self.env.registry, raw_citations
            )

        # Apply span-existence gate to surviving citations: supporting_span
        # must occur in the actual article text (catches LLM-fabricated spans
        # that pass the existence check).
        if valid_citations and not _os.getenv("AKN_NO_CITATION_GATE"):
            valid_citations, span_rejected = span_existence.filter_citations(
                self.env, valid_citations
            )
            rejected.extend(span_rejected)

        # stash as normalized dicts so downstream hint builders can always call .get()
        answer["_raw_citations"] = valid_citations + rejected
        answer["citations"] = valid_citations
        if rejected and not answer.get("abstention"):
            answer.setdefault("abstention_reason", "fictitious_citation")

        # Attach telemetry
        answer["trajectory"]        = trajectory
        answer["tokens_used"]       = budget.tokens_used + root_tokens_estimated
        answer["depth_max_reached"] = budget.max_depth_reached
        answer["sub_call_count"]    = budget.sub_calls_used

        return answer

    # ------------------------------------------------------------------
    def _build_corpus_manifest(self) -> str:
        """Return a compact list of all canonical doc IDs for the system prompt.

        Injected at the end of the system prompt so the LLM always has the
        exact doc IDs it must use in citations — no hallucination possible.
        """
        try:
            docs = self.env.list_documents()
            if not docs:
                return ""
            lines = [
                "\n\n═══════════════════════════════════════════════════════",
                "CORPUS MANIFEST — USE THESE EXACT doc_id VALUES IN CITATIONS",
                "═══════════════════════════════════════════════════════",
            ]
            for d in sorted(docs, key=lambda x: x.get("doc_id", "")):
                lines.append(
                    f"  {d['doc_id']:<38}  {d.get('doc_title', '')[:60]}"
                )
            lines.append(
                "\nNEVER invent doc_ids. Only use the exact strings listed above."
            )
            return "\n".join(lines)
        except Exception as exc:
            log.warning("Could not build corpus manifest: %s", exc)
            return ""

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, env: LegalEnv, query_type: str = "rule_application") -> "RootController":
        """Convenience factory: budget max_depth=2 for multi_hop, 1 otherwise."""
        env.budget.max_depth = 2 if query_type == "multi_hop" else 1
        return cls(env=env)
