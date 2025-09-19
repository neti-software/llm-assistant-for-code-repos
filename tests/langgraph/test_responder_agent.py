import pytest

from src.langgraph.state_models import ConversationState, EvidenceItem
from src.langgraph.agents.responder import ResponderAgent


@pytest.fixture
def agent():
    return ResponderAgent()


def build_state():
    state = ConversationState()
    state.evidence_store.extend(
        [
            EvidenceItem(source_path="src/foo.py", snippet="class Foo", citations=["file:0"], confidence=0.8),
            EvidenceItem(source_path="docs/README.md", summary="Foo overview", citations=["rag:0"], confidence=0.7),
        ]
    )
    state.control_flags.last_verifier_report = {
        "coverage_score": 0.8,
        "missing_items": [],
        "citations": ["file:0", "rag:0"],
    }
    return state


def test_responder_formats_final_message(agent):
    state = build_state()
    response = agent.respond(state)

    assert response.message.startswith("Final Answer:")
    assert "class Foo" in response.message
    assert "Foo overview" in response.message
    assert response.citations == ["file:0", "rag:0"]


def test_responder_updates_conversation_history(agent):
    state = build_state()
    dummy_history = type("History", (), {
        "add_model_response": lambda self, payload: setattr(self, "last", payload),
    })()

    response = agent.respond(state)
    agent.persist_response(dummy_history, response)

    assert hasattr(dummy_history, "last")
    assert dummy_history.last == response.message
