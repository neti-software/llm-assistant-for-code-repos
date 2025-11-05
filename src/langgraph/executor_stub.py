"""Full LangGraph executor with detailed execution tracking.

This module implements the complete LangGraph pipeline with all agents,
providing comprehensive execution data for benchmarking purposes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .state_models import ConversationState
from .orchestrator import Orchestrator
from .agents import TaskPlannerAgent, RepoIntelligenceAgent, CodeInspectorAgent, VerifierAgent, ResponderAgent
from .tool_nodes.base import ToolNodeAdapter


def build_detailed_executor(tool_manager: Any, live_log=None, llm=None) -> Orchestrator:
    """Build a fully configured orchestrator with all agents and tools."""

    # Build tool nodes
    tool_nodes = []
    if hasattr(tool_manager, '_available_tools'):
        for tool_name in tool_manager._available_tools:
            tool_nodes.append(ToolNodeAdapter(tool_manager, tool_name))

    # Create agents
    task_planner = TaskPlannerAgent(llm=llm)
    repo_agent = RepoIntelligenceAgent(tool_nodes, llm=llm)
    code_agent = CodeInspectorAgent()
    verifier = VerifierAgent(coverage_threshold=0.7, llm=llm)
    responder = ResponderAgent(llm=llm)

    # Create orchestrator
    orchestrator = Orchestrator(
        task_planner=task_planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=5,
        llm=llm,
    )

    return orchestrator


def execute_turn(
    llm: Any,
    tool_manager: Any,
    state: ConversationState,
    live_log=None
) -> Dict[str, Any]:
    """Execute a full LangGraph turn with comprehensive tracking."""

    # Build execution context
    execution_context = {
        "start_time": time.time(),
        "agent_executions": [],
        "task_planning": {},
        "evidence_collection": [],
        "verifier_report": {},
        "total_iterations": 0,
    }

    # Build orchestrator
    orchestrator = build_detailed_executor(tool_manager, live_log, llm=llm)

    # Run orchestrator and capture detailed execution data
    try:
        final_report = orchestrator.run(state, live_log=live_log)
        execution_context["total_iterations"] = state.control_flags.iteration
        execution_context["verifier_report"] = {
            "response_text": getattr(final_report, 'response_text', ''),
            "coverage_score": getattr(final_report, 'coverage_score', 0.0),
            "missing_items": getattr(final_report, 'missing_items', []),
            "citations": getattr(final_report, 'citations', []),
        }

        # Capture task planning data
        if state.tasks:
            execution_context["task_planning"] = {
                "total_tasks": len(state.tasks),
                "completed_tasks": len([t for t in state.tasks if t.status == "done"]),
                "tasks": [
                    {
                        "id": task.id,
                        "type": task.type,
                        "status": task.status,
                        "owner": task.owner,
                        "description": task.description,
                        "metadata": task.metadata,
                    }
                    for task in state.tasks
                ]
            }

        # Capture evidence collection data
        if state.evidence_store:
            execution_context["evidence_collection"] = [
                {
                    "source_path": item.source_path,
                    "full_content": item.full_content,
                    "citations": item.citations,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                }
                for item in state.evidence_store
            ]

        execution_context["total_time"] = time.time() - execution_context["start_time"]
        execution_context["success"] = True

    except Exception as e:
        execution_context["error"] = str(e)
        execution_context["total_time"] = time.time() - execution_context["start_time"]
        execution_context["success"] = False

        # Return a basic response on error
        return {
            "response": f"LangGraph execution failed: {e}",
            "state": state,
            "execution_context": execution_context,
        }

    # Generate comprehensive response
    response_parts = [
        "[LangGraph] Full execution completed",
        f"Total iterations: {execution_context['total_iterations']}",
        f"Total execution time: {execution_context['total_time']:.2f}s",
        f"Coverage score: {execution_context['verifier_report'].get('coverage_score', 0):.2f}",
    ]

    if execution_context["verifier_report"].get("response_text"):
        response_parts.append("\nFinal Response:")
        response_parts.append(execution_context["verifier_report"]["response_text"])

    response_text = "\n".join(response_parts)

    return {
        "response": response_text,
        "state": state,
        "execution_context": execution_context,
    }
