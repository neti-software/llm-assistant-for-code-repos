import pytest

from src.langgraph.agents.task_planner import TaskPlannerAgent
from src.langgraph.agents.repo_intelligence import RepoIntelligenceAgent
from src.langgraph.agents.code_inspector import CodeInspectorAgent
from src.langgraph.agents.verifier import VerifierAgent, VerifierReport
from src.langgraph.agents.responder import ResponderAgent
from src.langgraph.orchestrator import Orchestrator
from src.langgraph.state_models import ConversationState, ConversationBuffer, EvidenceItem


class StubToolNode:
    def __init__(self, tool_name, payload, *, confidence=0.8):
        self.tool_name = tool_name
        self.payload = payload
        self.confidence = confidence
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "tool_name": self.tool_name,
            "data": list(self.payload),
            "citations": [f"{self.tool_name}:{idx}" for idx, _ in enumerate(self.payload)],
            "confidence": self.confidence,
            "error": None,
            "metadata": {"args": kwargs},
        }


def build_state(question: str) -> ConversationState:
    buffer = ConversationBuffer(
        user_questions={"user_question1": question},
        history=[{"iteration": 0, "user_question1": question}],
    )
    return ConversationState(conversation=buffer)


def test_langgraph_stub_flow_runs_end_to_end():
    search_node = StubToolNode(
        "search_files_with_grep",
        [
            {"path": "src/foo.py", "snippet": "class Foo:", "score": 0.9},
        ],
    )
    rag_node = StubToolNode(
        "rag_search",
        [
            {"path": "docs/README.md", "snippet": "Foo architecture overview with detailed explanation of the Foo class implementation."},
        ],
    )
    repo_agent = RepoIntelligenceAgent(tool_nodes=[search_node, rag_node])

    structure_node = StubToolNode("fetch_project_structure", ["."], confidence=1.0)
    file_node = StubToolNode(
        "fetch_file_from_patch",
        [
            {
                "path": "src/foo.py",
                "snippet": "class Foo:\n    pass",
                "line_start": 1,
                "line_end": 2,
                "score": 0.95,
            }
        ],
        confidence=0.9,
    )
    code_agent = CodeInspectorAgent(structure_node=structure_node, file_node=file_node)

    task_planner = TaskPlannerAgent()
    verifier = VerifierAgent(coverage_threshold=0.6)
    responder = ResponderAgent()

    orchestrator = Orchestrator(
        task_planner=task_planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        max_iterations=3,
    )

    state = build_state("Where is the Foo class implemented?")

    final_report, timeline = orchestrator.run(state)
    assert isinstance(final_report, VerifierReport)
    assert final_report.coverage_score >= 0.6
    assert not final_report.missing_items

    response = responder.respond(state)
    assert "Foo architecture overview with detailed explanation" in response.message
    assert "class Foo" in response.message
    assert response.citations

    # ensure evidence recorded
    assert any(item.source_path == "src/foo.py" for item in state.evidence_store)

    # ensure planner added tasks and they were marked
    assert any(task.type == "repo_research" for task in state.tasks)
    assert any(task.type == "code_context" for task in state.tasks)
