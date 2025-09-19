"""LangGraph executor that wires agents together for a single turn."""

from __future__ import annotations

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

    repo_nodes: List[Any] = []
    for tool_name in ("rag_search", "rag_search_project_readme", "search_files_with_grep"):
        node = _maybe_tool(tool_manager, tool_name)
        if node is not None:
            repo_nodes.append(node)
    repo_agent = RepoIntelligenceAgent(tool_nodes=repo_nodes)

    structure_node = None  # full structure support to be added with richer tasks
    file_node = _maybe_tool(tool_manager, "fetch_file_from_patch")
    code_agent = CodeInspectorAgent(structure_node=structure_node, file_node=file_node)

    verifier = VerifierAgent(coverage_threshold=0.7)
    responder = ResponderAgent()

    orchestrator = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        max_iterations=4,
    )

    orchestrator.run(state, live_log=live_log)
    final_response: FinalResponse = responder.respond(state)

    return {
        "response": final_response,
        "state": state,
    }
