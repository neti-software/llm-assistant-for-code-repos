import pytest

from src.langgraph.state_models import ConversationState, EvidenceItem
from src.langgraph.agents.repo_intelligence import RepoIntelligenceAgent


class StubToolNode:
    def __init__(self, tool_name, payload):
        self.tool_name = tool_name
        self.payload = payload

    def invoke(self, **kwargs):
        return {
            "tool_name": self.tool_name,
            "data": self.payload,
            "citations": [f"{self.tool_name}:{idx}" for idx, _ in enumerate(self.payload)],
            "confidence": 0.8,
            "error": None,
            "metadata": {"args": kwargs},
        }


@pytest.fixture
def agent():
    search_stub = StubToolNode(
        "search_files_with_grep",
        [
            {"path": "src/foo.py", "snippet": "class Foo:", "score": 0.9},
            {"path": "src/bar.py", "snippet": "class Bar:", "score": 0.75},
        ],
    )
    rag_stub = StubToolNode(
        "rag_search",
        [
            {"path": "docs/README.md", "summary": "Overview of Foo"},
        ],
    )
    agent = RepoIntelligenceAgent(tool_nodes=[search_stub, rag_stub])
    return agent


def test_repo_intelligence_agent_collects_evidence(agent):
    state = ConversationState()
    updated_state, evidence = agent.run(state, query="Find Foo class")

    assert len(evidence) == 3
    assert all(isinstance(item, EvidenceItem) for item in evidence)

    paths = {item.source_path for item in evidence}
    assert "src/foo.py" in paths
    assert "src/bar.py" in paths
    assert "docs/README.md" in paths

    scores = {item.metadata.get("score") for item in evidence if item.metadata}
    assert 0.9 in scores
    assert 0.75 in scores

    assert len(updated_state.evidence_store) == len(evidence)


def test_agent_skips_tool_errors():
    failing_stub = StubToolNode("search_files_with_grep", [])

    def invoke(**kwargs):
        return {
            "tool_name": "search_files_with_grep",
            "data": None,
            "citations": [],
            "confidence": None,
            "error": "boom",
            "metadata": {"args": kwargs},
        }

    failing_stub.invoke = invoke

    agent = RepoIntelligenceAgent(tool_nodes=[failing_stub])
    state = ConversationState()
    updated_state, evidence = agent.run(state, query="test")

    assert evidence == []
    assert updated_state.evidence_store == []
