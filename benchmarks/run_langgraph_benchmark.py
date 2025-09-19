#!/usr/bin/env python3
"""Run a benchmark set of questions through the LangGraph or legacy pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_cli import build_core, chat_loop, _extract_message_and_citations, llm_loop
from src.langgraph.runner import LangGraphRunner
from src.langgraph.executor_stub import execute_turn as langgraph_stub_executor
from src.utils.profiler import execution_profiler

QUESTIONS_PATH = Path(__file__).parent / "langgraph_benchmark_questions.json"
RESULTS_PATH = Path(__file__).parent / "langgraph_benchmark_results.json"


def ensure_flag_active(use_langgraph: bool) -> None:
    if not use_langgraph:
        print("[warning] use_langgraph_multi_agent is disabled in configs/llm_config.yaml.")
        print("          Enable the flag for a full LangGraph preview run.")


def run_question(question: str, mode: str) -> dict:
    llm, tool_manager, conversation_history, use_langgraph = build_core()

    if mode == "legacy":
        runner = LangGraphRunner(use_graph=False, legacy_llm_loop=llm_loop)
    else:
        ensure_flag_active(use_langgraph)
        runner = LangGraphRunner(use_graph=True, legacy_llm_loop=llm_loop)
        runner.set_graph_executor(langgraph_stub_executor)

    response, final_message = chat_loop(question, llm, tool_manager, conversation_history, runner)

    _, citations = _extract_message_and_citations(response)
    report = {
        "question": question,
        "answer": final_message,
        "citations": citations,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
    }

    if conversation_history.iteration:
        report["iterations"] = conversation_history.iteration

    execution_profiler.clean()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph benchmark questions.")
    parser.add_argument(
        "--mode",
        choices=["preview", "legacy"],
        default="preview",
        help="'preview' uses the LangGraph stub executor, 'legacy' uses the existing llm_loop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not QUESTIONS_PATH.exists():
        print(f"Question file missing: {QUESTIONS_PATH}")
        return 1

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []

    for idx, question in enumerate(questions, start=1):
        print(f"\n[{idx}/{len(questions)}] {question}")
        try:
            result = run_question(question, mode=args.mode)
            print("  → Completed\n")
        except Exception as exc:  # pylint: disable=broad-except
            result = {
                "question": question,
                "error": str(exc),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": args.mode,
            }
            print(f"  → Error: {exc}\n")
        results.append(result)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Benchmark results saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
