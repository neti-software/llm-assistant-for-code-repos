import pytest

from src.langgraph.state_models import ConversationState, EvidenceItem
from src.langgraph.agents.verifier import VerifierAgent


@pytest.fixture
def agent():
    return VerifierAgent(coverage_threshold=0.6)


def build_state_with_evidence():
    state = ConversationState()
    state.evidence_store.extend(
        [
            EvidenceItem(
                source_path="src/foo.py",
                summary="Foo class handles requests",
                snippet="class Foo:\n    pass",
                confidence=0.8,
                citations=["search:0"],
            ),
            EvidenceItem(
                source_path="docs/README.md",
                summary="Overview of Foo architecture",
                confidence=0.7,
                citations=["rag:0"],
            ),
        ]
    )
    return state


def test_verifier_agent_produces_response_and_coverage(agent):
    state = build_state_with_evidence()
    report = agent.evaluate(state, question="Explain Foo class")

    assert report.response_text.startswith("Summary")
    assert report.coverage_score >= 0.6
    assert report.missing_items == []
    assert report.citations == ["search:0", "rag:0"]


def test_verifier_agent_flags_missing_evidence(agent):
    state = ConversationState()
    report = agent.evaluate(state, question="Explain Foo class")

    assert report.coverage_score == 0.0
    assert "Foo class" in report.missing_items[0]
    assert report.response_text.startswith("Insufficient evidence")


def test_verifier_agent_applies_state_updates(agent):
    state = build_state_with_evidence()
    report = agent.evaluate(state, question="Explain Foo class")

    agent.apply_report(state, report)

    assert state.control_flags.last_verifier_report is not None
    assert state.control_flags.iteration == 0  # unchanged by verifier
