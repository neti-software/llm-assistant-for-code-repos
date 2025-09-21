#!/usr/bin/env python3
"""Run a benchmark set of questions through the LangGraph or legacy pipeline.

This script provides comprehensive execution tracking including:
- Agent chain execution details
- Task planning and completion status
- Evidence collection metrics
- Coverage scores and verification reports
- Live logging of agent progress
- Execution timing and performance metrics

Usage:
    python benchmarks/run_langgraph_benchmark.py --mode preview
    python benchmarks/run_langgraph_benchmark.py --mode legacy
    python benchmarks/run_langgraph_benchmark.py --demo

For a demo without full dependencies:
    python benchmarks/run_langgraph_benchmark.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 friendly stdout on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except OSError:
        pass

# Set up Python path before importing project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_cli import build_core, llm_loop, chat_loop, _extract_message_and_citations
from src.langgraph.runner import LangGraphRunner
from src.langgraph.executor import execute_turn as langgraph_executor
from src.utils.profiler import execution_profiler

QUESTIONS_PATH = Path(__file__).parent / "langgraph_benchmark_questions.json"
RESULTS_PATH = Path(__file__).parent / "langgraph_benchmark_results.json"


def ensure_flag_active(use_langgraph: bool) -> None:
    if not use_langgraph:
        print("[warning] use_langgraph_multi_agent is disabled in configs/llm_config.yaml.")
        print("          Enable the flag for a full LangGraph preview run.")


def run_question(question: str, mode: str) -> dict:
    """Run a single question and capture comprehensive execution data."""
    llm, tool_manager, conversation_history, use_langgraph = build_core()

    # Initialize execution tracking
    execution_data = {
        "question": question,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "execution_profiler_events": [],
        "conversation_history": None,
        "langgraph_execution_context": None,
    }

    if mode == "legacy":
        runner = LangGraphRunner(use_graph=False, legacy_llm_loop=llm_loop)
        response, final_message = chat_loop(question, llm, tool_manager, conversation_history, runner)
        _, citations = _extract_message_and_citations(response)
        execution_data["answer"] = final_message
        execution_data["citations"] = citations
        if conversation_history.iteration:
            execution_data["iterations"] = conversation_history.iteration
    else:
        ensure_flag_active(use_langgraph)
        runner = LangGraphRunner(use_graph=True, legacy_llm_loop=llm_loop)
        runner.set_graph_executor(langgraph_executor)

        # Capture live logging from LangGraph execution
        live_log_messages = []
        def live_log_capture(message: str):
            live_log_messages.append(message)
            print(f"  [LangGraph] {message}")

        # Run the question and capture execution context
        response, final_message = chat_loop(
            question,
            llm,
            tool_manager,
            conversation_history,
            runner,
            live_log=live_log_capture
        )

        _, citations = _extract_message_and_citations(response)

        # Extract comprehensive execution data
        execution_data["answer"] = final_message
        execution_data["citations"] = citations
        execution_data["live_log_messages"] = live_log_messages

        if conversation_history.iteration:
            execution_data["iterations"] = conversation_history.iteration

        # Capture LangGraph execution context if available
        if isinstance(response, dict) and "execution_context" in response:
            execution_data["langgraph_execution_context"] = response["execution_context"]

        # Capture profiler events
        execution_data["execution_profiler_events"] = execution_profiler.iter_events()

    # Save conversation history for detailed analysis
    execution_data["conversation_history"] = conversation_history.history

    if mode == "preview" and not execution_data.get("langgraph_execution_context"):
        execution_data["langgraph_execution_context"] = {
            "success": True,
            "total_iterations": execution_data.get("iterations", 0),
        }

    execution_profiler.clean()
    return execution_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph benchmark questions.")
    parser.add_argument(
        "--mode",
        choices=["preview", "legacy", "demo"],
        default="demo",
        help="'preview' uses the LangGraph executor, 'legacy' uses the existing llm_loop, 'demo' shows mock data.",
    )
    return parser.parse_args()


def create_demo_data() -> list:
    """Create comprehensive demo data showing enhanced benchmark capabilities."""
    from datetime import timezone
    return [
        {
            "question": "How to obtain DC in the current Filecoin+ program?",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "demo",
            "answer": "[LangGraph] Full execution completed\nTotal iterations: 3\nTotal execution time: 2.45s\nCoverage score: 0.85\n\nFinal Response:\nBased on the evidence collected, here is how to obtain DataCap (DC) in the Filecoin+ program...",
            "citations": ["file1.py", "file2.py", "docs/filecoin-plus.md"],
            "live_log_messages": [
                "Task Planner generated 2 task(s) in 0.12s",
                "Repo Intelligence agent collecting repo evidence",
                "Repo Intelligence agent returned 5 item(s) in 1.23s",
                "Code Inspector agent fetching targeted snippets",
                "Code Inspector agent returned 3 item(s) in 0.89s",
                "Verifier agent evaluating coverage",
                "Verifier coverage=0.85 evaluated in 0.21s",
                "Evidence collection summary: 8 items collected",
                "LangGraph execution completed: 3 iterations, 2.45s total time"
            ],
            "langgraph_execution_context": {
                "start_time": 1704067200.0,
                "total_time": 2.45,
                "total_iterations": 3,
                "success": True,
                "task_planning": {
                    "total_tasks": 2,
                    "completed_tasks": 2,
                    "tasks": [
                        {
                            "id": "task-1",
                            "type": "repo_research",
                            "status": "done",
                            "owner": "repo_intelligence_agent",
                            "description": "Gather repository-level context and relevant snippets",
                            "metadata": {"input_question": "How to obtain DC in the current Filecoin+ program?"}
                        },
                        {
                            "id": "task-2",
                            "type": "code_context",
                            "status": "done",
                            "owner": "code_inspector_agent",
                            "description": "Collect precise file excerpts supporting the query",
                            "metadata": {"input_question": "How to obtain DC in the current Filecoin+ program?"}
                        }
                    ]
                },
                "evidence_collection": [
                    {
                        "source_path": "src/filecoin/storage_provider.py",
                        "summary": "DataCap allocation process for storage providers",
                        "snippet": "def request_datacap(amount, justification):\n    # Implementation for DC requests",
                        "citations": ["storage_provider.py:42"],
                        "confidence": 0.85,
                        "metadata": {"tool": "search_files_with_grep"}
                    },
                    {
                        "source_path": "docs/filecoin-plus-guide.md",
                        "summary": "Complete guide to Filecoin+ DataCap program",
                        "snippet": "## Obtaining DataCap\n\n1. Apply through Filecoin+ program\n2. Submit justification\n3. Wait for approval",
                        "citations": ["filecoin-plus-guide.md:15"],
                        "confidence": 0.92,
                        "metadata": {"tool": "fetch_project_structure"}
                    }
                ],
                "verifier_report": {
                    "response_text": "Based on the evidence collected, here is how to obtain DataCap (DC) in the Filecoin+ program...",
                    "coverage_score": 0.85,
                    "missing_items": [],
                    "citations": ["storage_provider.py:42", "filecoin-plus-guide.md:15"]
                }
            },
            "execution_profiler_events": [
                {"name": "graph_turn_start", "timestamp": 1704067200.0, "metadata": {"question": "How to obtain DC..."}},
                {"name": "graph_turn_end", "timestamp": 1704067202.45, "metadata": {"response_type": "str"}},
                {"name": "graph_turn_persisted", "timestamp": 1704067202.46, "metadata": {"citations": ["storage_provider.py:42"]}}
            ],
            "conversation_history": {
                "user_questions": {"user_question1": "How to obtain DC in the current Filecoin+ program?"},
                "history": [
                    {"iteration": 0, "user_question1": "How to obtain DC in the current Filecoin+ program?"},
                    {"iteration": 1, "rag_results": [{"id": 1, "snippet": "DataCap is allocated..."}]},
                    {"iteration": 2, "function_call": {"name": "search_files_with_grep", "args": {"query": "datacap"}}},
                    {"iteration": 3, "model_response": "Based on evidence..."}
                ]
            }
        }
    ]


def main() -> int:
    args = parse_args()

    if args.mode == "demo":
        print("🔬 DEMO MODE: Enhanced LangGraph Benchmark")
        print("=" * 50)
        results = create_demo_data()
        print(f"Generated {len(results)} demo result(s)")
    else:
        if not QUESTIONS_PATH.exists():
            print(f"Question file missing: {QUESTIONS_PATH}")
            return 1

        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        results = []

        for idx, question in enumerate(questions, start=1):
            print(f"\n[{idx}/{len(questions)}] {question}")
            try:
                result = run_question(question, mode=args.mode)

                # Print summary information
                ctx = result.get("langgraph_execution_context") or {}
                if ctx:
                    if ctx.get("success"):
                        print(f"  → Completed ({ctx.get('total_time', 0):.2f}s, {ctx.get('total_iterations', 0)} iterations)")
                        if ctx.get("task_planning"):
                            completed = ctx["task_planning"].get("completed_tasks", 0)
                            total = ctx["task_planning"].get("total_tasks", 0)
                            print(f"    Tasks: {completed}/{total} completed")
                        if ctx.get("evidence_collection"):
                            print(f"    Evidence items: {len(ctx['evidence_collection'])}")
                        if ctx.get("verifier_report"):
                            score = ctx["verifier_report"].get("coverage_score", 0)
                            print(f"    Coverage score: {score:.2f}")
                    else:
                        print(f"  → Error: {ctx.get('error', 'Unknown error')}")
                elif result.get("mode") == "preview":
                    print("  → Completed (preview mode - execution context unavailable)")
                else:
                    print("  → Completed (legacy mode)")
            except Exception as exc:  # pylint: disable=broad-except
                from datetime import timezone
                result = {
                    "question": question,
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": args.mode,
                }
                print(f"  -> Error: {exc}\n")
            results.append(result)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Benchmark results saved to {RESULTS_PATH}")

    # Print summary statistics
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    successful_runs = [r for r in results if not r.get("error")]
    failed_runs = [r for r in results if r.get("error")]

    print(f"Total questions: {len(results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")

    if successful_runs:
        langgraph_runs = [r for r in successful_runs if r.get("mode") == "preview"]
        legacy_runs = [r for r in successful_runs if r.get("mode") == "legacy"]

        print(f"LangGraph runs: {len(langgraph_runs)}")
        print(f"Legacy runs: {len(legacy_runs)}")

        if langgraph_runs:
            total_time = 0.0
            coverage_scores = []
            for run in langgraph_runs:
                ctx = run.get("langgraph_execution_context") or {}
                total_time += ctx.get("total_time", 0) or 0.0
                coverage_scores.append((ctx.get("verifier_report") or {}).get("coverage_score", 0) or 0.0)

            avg_time = total_time / len(langgraph_runs)
            print(f"Average LangGraph execution time: {avg_time:.2f}s")

            if coverage_scores:
                avg_coverage = sum(coverage_scores) / len(coverage_scores)
                print(f"Average coverage score: {avg_coverage:.2f}")

    if failed_runs:
        print(f"\nFailed questions:")
        for run in failed_runs:
            print(f"  - {run['question'][:60]}...: {run['error']}")

    print(f"\nDetailed results available in: {RESULTS_PATH}")
    return 0


def analyze_results(results_file: Path = None) -> None:
    """Analyze benchmark results and provide detailed insights."""
    if results_file is None:
        results_file = RESULTS_PATH

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    results = json.loads(results_file.read_text(encoding="utf-8"))

    print("\n" + "="*80)
    print("DETAILED BENCHMARK ANALYSIS")
    print("="*80)

    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['question'][:80]}...")
        print(f"   Mode: {result.get('mode', 'unknown')}")
        print(f"   Timestamp: {result['timestamp']}")

        if result.get("error"):
            print(f"   ❌ Error: {result['error']}")
            continue

        print(f"   ✅ Success")

        # LangGraph detailed analysis
        ctx = result.get("langgraph_execution_context") or {}
        if ctx:
            print(f"   Execution time: {ctx.get('total_time', 0) or 0:.2f}s")
            print(f"   Iterations: {ctx.get('total_iterations', 0)}")

            if ctx.get("task_planning"):
                tp = ctx["task_planning"]
                print(f"   Tasks completed: {tp.get('completed_tasks', 0)}/{tp.get('total_tasks', 0)}")

                for task in tp.get("tasks", []):
                    status_icon = "✅" if task["status"] == "done" else "⏳"
                    print(f"     {status_icon} {task['type']} ({task['owner']})")

            if ctx.get("evidence_collection"):
                print(f"   Evidence items: {len(ctx['evidence_collection'])}")

            if ctx.get("verifier_report"):
                vr = ctx["verifier_report"]
                print(f"   Coverage score: {vr.get('coverage_score', 0):.2f}")
                if vr.get("missing_items"):
                    print(f"   Missing items: {len(vr['missing_items'])}")
        elif result.get("mode") == "preview":
            print("   LangGraph execution context unavailable (preview run).")

        # Live log messages
        if result.get("live_log_messages"):
            print(f"   Live logs: {len(result['live_log_messages'])} messages")

        # Execution profiler events
        if result.get("execution_profiler_events"):
            print(f"   Profiler events: {len(result['execution_profiler_events'])}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_results()
    else:
        raise SystemExit(main())
