"""Single-query demo for the AKN-RLM UI endpoint.

Run a single legal query through the deployable Phase E pipeline and
print the human-readable trajectory + answer + citations + references.
Useful for smoke-testing the API before wiring it to the front-end.

Usage::

    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"

    # One-shot
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\answer_query.py `
        --query "ما هي شروط الزواج في قانون الأسرة؟"

    # Interactive REPL
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\answer_query.py --repl

    # Emit JSON for piping into a UI
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\answer_query.py `
        --query "..." --json
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# UTF-8 stdout on Windows so Arabic prints without `\uXXXX`.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.api import answer_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("answer_query")


def _print_response(response, *, show_detail: bool = False) -> None:
    """Pretty-print an :class:`AnswerResponse` to stdout."""
    print()
    print("=" * 72)
    print(f"QUERY:    {response.query}")
    print(f"TYPE:     {response.query_type_predicted}   "
          f"HANDLER: {response.handler_used}")
    print(f"LATENCY:  {response.latency_s:.2f}s   "
          f"SUB-LM CALLS: {response.sub_call_count}   "
          f"RECURSION DEPTH: {response.recursion_depth_max}   "
          f"CORRECTIVE RETRY: {response.corrective_retry_fired}")
    print("=" * 72)

    print()
    print("=== EXPLAINABILITY (pipeline trajectory) ===")
    for i, step in enumerate(response.trajectory, start=1):
        print(f"  {i:2d}. [{step.step}/d{step.depth}] {step.summary}")
        if show_detail and step.detail:
            for k, v in step.detail.items():
                vs = repr(v)
                if len(vs) > 120:
                    vs = vs[:117] + "…"
                print(f"        - {k}: {vs}")

    print()
    if response.abstained:
        print("=== ABSTAINED ===")
        print(f"Reason: {response.abstention_reason}")
    else:
        print("=== ANSWER ===")
        print(response.answer_text)

    if response.citations:
        print()
        print("=== CITATIONS ===")
        for i, c in enumerate(response.citations, start=1):
            head = f"  [{i}] {c.doc_title}, art. {c.article_ref}"
            if c.version_date:
                head += f" (نسخة {c.version_date})"
            head += f"   conf={c.confidence:.3f}"
            if c.kg_source:
                head += f"   kg_source={c.kg_source}"
            print(head)
            if c.supporting_span:
                span = c.supporting_span
                if len(span) > 240:
                    span = span[:237] + "…"
                print(f"        supporting_span: {span}")
            if c.argumentation:
                claim = (c.argumentation.get("claim") or "").strip()
                ground = (c.argumentation.get("ground") or "").strip()
                if claim:
                    print(f"        Toulmin.claim:  {claim[:200]}")
                if ground:
                    print(f"        Toulmin.ground: {ground[:200]}")

    if response.references:
        print()
        print("=== REFERENCES ===")
        for ref in response.references:
            print(f"  {ref}")
    print()


def _repl() -> None:
    """Interactive REPL for ad-hoc query testing."""
    print("AKN-RLM REPL — enter a query, blank line / Ctrl-D to quit.")
    while True:
        try:
            line = input("\nquery> ").strip()
        except EOFError:
            print()
            break
        if not line:
            break
        try:
            response = answer_query(line)
        except Exception as exc:
            log.exception("answer_query failed: %s", exc)
            continue
        _print_response(response, show_detail=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo the AKN-RLM UI endpoint for a single legal query."
    )
    parser.add_argument("--query", "-q", default=None,
                        help="One-shot query string.")
    parser.add_argument("--repl", action="store_true",
                        help="Drop into an interactive query REPL.")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON to stdout instead of pretty-printing. "
                             "Useful for piping into a UI / curl.")
    parser.add_argument("--detail", action="store_true",
                        help="Show the raw detail dict for each trajectory step.")
    args = parser.parse_args()

    if not args.query and not args.repl:
        parser.error("supply either --query or --repl")

    if args.repl:
        _repl()
        return

    response = answer_query(args.query)
    if args.json:
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_response(response, show_detail=args.detail)


if __name__ == "__main__":
    main()
