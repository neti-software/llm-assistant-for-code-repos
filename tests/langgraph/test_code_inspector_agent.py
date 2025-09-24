import pytest

from src.langgraph.state_models import ConversationState, Task, EvidenceItem
from src.langgraph.agents.code_inspector import CodeInspectorAgent


class RecordingToolNode:
    def __init__(self, tool_name, payload):
        self.tool_name = tool_name
        self.payload = payload
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "tool_name": self.tool_name,
            "data": self.payload,
            "citations": [f"{self.tool_name}:{idx}" for idx, _ in enumerate(self.payload)],
            "confidence": 0.7,
            "error": None,
            "metadata": {"args": kwargs},
        }


@pytest.fixture
def agent():
    structure_node = RecordingToolNode(
        "fetch_project_structure",
        [
            {"path": "src/foo.py"},
            {"path": "src/bar.py"},
        ],
    )

    file_node = RecordingToolNode(
        "fetch_file_from_patch",
        [
            {
                "path": "src/foo.py",
                "snippet": "class Foo:\n    pass",
                "line_start": 1,
                "line_end": 2,
                "score": 0.95,
            },
        ],
    )

    return CodeInspectorAgent(structure_node=structure_node, file_node=file_node)


def test_code_inspector_agent_collects_file_snippets(agent):
    task = Task(
        id="task-5",
        type="code_context",
        owner="code_inspector_agent",
        metadata={
            "target_paths": ["src/foo.py"],
            "project_root": ".",
            "input_question": "Where is Foo defined?",
        },
    )
    state = ConversationState()

    updated_state, evidence = agent.run(state, task=task)

    assert len(evidence) == 1
    item = evidence[0]
    assert isinstance(item, EvidenceItem)
    assert item.source_path == "src/foo.py"
    assert item.snippet.startswith("class Foo")
    assert item.metadata["line_start"] == 1
    assert item.metadata["tool"] == "fetch_file_from_patch"

    # Ensure calls captured expected arguments
    assert agent.structure_node.calls[0]["root"] == "."
    assert agent.file_node.calls[0]["file_path"] == "src/foo.py"

    assert len(updated_state.evidence_store) == 1


def test_code_inspector_skips_errors():
    class ErrorNode(RecordingToolNode):
        def invoke(self, **kwargs):
            return {
                "tool_name": self.tool_name,
                "data": None,
                "citations": [],
                "confidence": None,
                "error": "failure",
                "metadata": {"args": kwargs},
            }

    error_file_node = ErrorNode("fetch_file_from_patch", [])
    agent = CodeInspectorAgent(structure_node=None, file_node=error_file_node)

    task = Task(id="task-2", type="code_context", metadata={"target_paths": ["src/foo.py"]})
    state = ConversationState()

    updated_state, evidence = agent.run(state, task=task)

    assert evidence == []
    assert updated_state.evidence_store == []


def test_code_inspector_handles_missing_target_paths(agent):
    task = Task(
        id="task-6",
        type="code_context",
        metadata={"input_question": "Show Foo"},
    )
    state = ConversationState()

    updated_state, evidence = agent.run(state, task=task)

    assert evidence == []
    assert updated_state.evidence_store == []
