"""LangGraph executor that wires agents together for a single turn."""

from __future__ import annotations

import time
from typing import Any, List

from .state_models import ConversationState
from .tool_nodes.base import ToolNodeAdapter
from .agents.task_planner import TaskPlannerAgent
from .agents.repo_intelligence import RepoIntelligenceAgent
from .agents.code_inspector import CodeInspectorAgent
from .agents.verifier import VerifierAgent
from .agents.responder import ResponderAgent, FinalResponse
from .orchestrator import Orchestrator


def _maybe_tool(tool_manager, name: str):
    if hasattr(tool_manager, "tools") and name in tool_manager.tools:
        return ToolNodeAdapter(tool_manager, name)
    return None


def execute_turn(llm: Any, tool_manager: Any, state: ConversationState, live_log=None) -> dict:
    """Execute the LangGraph agents and return the final response plus updated state."""

    planner = TaskPlannerAgent()
    # Pass the LLM client directly to the planner
    if llm:
        planner.llm = llm

    repo_nodes: List[Any] = []
    for tool_name in ("rag_search", "rag_search_project_readme", "search_files_with_grep"):
        node = _maybe_tool(tool_manager, tool_name)
        if node is not None:
            repo_nodes.append(node)
    repo_agent = RepoIntelligenceAgent(tool_nodes=repo_nodes, llm=llm)

    structure_node = None  # full structure support to be added with richer tasks
    file_node = _maybe_tool(tool_manager, "fetch_file_from_patch")
    code_agent = CodeInspectorAgent(structure_node=structure_node, file_node=file_node)

    verifier = VerifierAgent(coverage_threshold=0.7, llm=llm)
    responder = ResponderAgent()

    orchestrator = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        llm=llm,
        max_iterations=4,
    )

    # Capture detailed execution context
    execution_context = {
        "start_time": time.perf_counter(),
        "task_planning": {},
        "evidence_collection": [],
        "verifier_report": None,
        "total_iterations": 0,
        "success": False,
    }

    final_report, iteration_log = orchestrator.run(state, live_log=live_log)
    final_response: FinalResponse = responder.respond(state)

    # Extract detailed execution information
    execution_context["total_iterations"] = state.control_flags.iteration + 1
    execution_context["total_time"] = time.perf_counter() - execution_context["start_time"]
    execution_context["success"] = True

    # Extract task planning details
    if state.tasks:
        execution_context["task_planning"] = {
            "total_tasks": len(state.tasks),
            "completed_tasks": len([t for t in state.tasks if t.status == "done"]),
            "tasks": [task.dict() for task in state.tasks]
        }

    # Extract evidence collection details
    execution_context["evidence_collection"] = [item.dict() for item in state.evidence_store]
    execution_context["iteration_log"] = iteration_log

    # Extract verifier report
    if hasattr(state.control_flags, 'last_verifier_report') and state.control_flags.last_verifier_report:
        execution_context["verifier_report"] = state.control_flags.last_verifier_report

    return {
        "response": final_response,
        "state": state,
        "execution_context": execution_context,
    }
