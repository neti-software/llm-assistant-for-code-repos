import pytest

from src.langgraph.state_models import ConversationState, Task, EvidenceItem
from src.langgraph.orchestrator import Orchestrator
from src.langgraph.agents.responder import FinalResponse


class FakeTaskPlanner:
    def __init__(self, task_batches):
        self.task_batches = task_batches  # list of lists returned per call
        self.calls = 0

    def plan(self, state):
        batch = []
        if self.calls < len(self.task_batches):
            batch = self.task_batches[self.calls]
            for task in batch:
                state.tasks.append(task)
        self.calls += 1
        return state, list(batch)


class FakeRepoAgent:
    def __init__(self):
        self.calls = 0

    def run(self, state, *, query):
        self.calls += 1
        state.evidence_store.append(
            EvidenceItem(source_path="src/foo.py", summary="Foo class", citations=["search:0"], confidence=0.8)
        )
        return state, list(state.evidence_store)


class FakeCodeAgent:
    def __init__(self):
        self.calls = 0

    def run(self, state, *, task):
        self.calls += 1
        state.evidence_store.append(
            EvidenceItem(source_path="src/foo.py", snippet="class Foo", citations=["file:0"], confidence=0.9)
        )
        task.status = "done"
        return state, [state.evidence_store[-1]]


class FakeVerifier:
    def __init__(self, coverage_sequence, missing_items_sequence=None):
        self.coverage_sequence = coverage_sequence
        self.missing_items_sequence = missing_items_sequence or []
        self.calls = 0
        self.coverage_threshold = 0.7

    def evaluate(self, state, *, question):
        coverage = self.coverage_sequence[min(self.calls, len(self.coverage_sequence) - 1)]
        
        # Allow explicit control of missing_items
        if self.missing_items_sequence:
            missing = self.missing_items_sequence[min(self.calls, len(self.missing_items_sequence) - 1)]
        else:
            missing = [] if coverage >= 0.7 else ["Need more"]
        
        self.calls += 1
        return type("Report", (), {
            "coverage_score": coverage,
            "missing_items": missing,
            "response_text": "summary",
            "citations": [],
        })()

    def apply_report(self, state, report):
        state.control_flags.last_verifier_report = {
            "coverage_score": report.coverage_score,
            "missing_items": report.missing_items,
        }


class FakeResponder:
    def __init__(self):
        self.calls = 0

    def respond(self, state):
        self.calls += 1
        return FinalResponse(
            message=f"Final answer based on {len(state.evidence_store)} evidence items",
            citations=[]
        )


@pytest.fixture
def orchestrator():
    planner = FakeTaskPlanner(
        [
            [
                Task(id="task-1", type="repo_research", owner="repo_intelligence_agent"),
                Task(id="task-2", type="code_context", owner="code_inspector_agent", metadata={"target_paths": ["src/foo.py"]}),
            ],
            [],
        ]
    )
    repo_agent = FakeRepoAgent()
    code_agent = FakeCodeAgent()
    verifier = FakeVerifier([0.4, 0.8])
    responder = FakeResponder()
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )
    return orch


def test_orchestrator_runs_until_verifier_succeeds(orchestrator):
    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report, timeline = orchestrator.run(state)

    assert isinstance(final_report, FinalResponse)
    assert orchestrator.repo_agent.calls == 1
    assert orchestrator.code_agent.calls == 1
    assert orchestrator.verifier.calls == 2
    assert len(state.tasks) == 2
    assert state.tasks[-1].status == "done"
    assert state.control_flags.iteration == 1
    # Check the final coverage was good from state
    assert state.control_flags.last_verifier_report["coverage_score"] >= 0.7


def test_orchestrator_stops_when_iteration_cap_hit():
    planner = FakeTaskPlanner([
        [Task(id="task-1", type="repo_research", owner="repo_intelligence_agent")],
        [],
    ])
    repo_agent = FakeRepoAgent()
    code_agent = FakeCodeAgent()
    verifier = FakeVerifier([0.2, 0.3, 0.2])
    responder = FakeResponder()
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=2,
    )

    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report, timeline = orch.run(state)

    assert isinstance(final_report, FinalResponse)
    assert state.control_flags.last_verifier_report["coverage_score"] == 0.3
    assert state.control_flags.iteration == 1  # 0-indexed, so 2 iterations = iteration 1


def test_orchestrator_continues_on_low_coverage_even_with_empty_missing_items():
    """Test Fix #1: Orchestrator should continue when coverage is low even if missing_items is empty."""
    planner = FakeTaskPlanner([
        [Task(id="task-1", type="repo_research", owner="repo_intelligence_agent")],
        [Task(id="task-2", type="code_context", owner="code_inspector_agent")],
        [],
    ])
    repo_agent = FakeRepoAgent()
    code_agent = FakeCodeAgent()
    # First iteration: low coverage (0.5) with empty missing_items
    # Second iteration: good coverage (0.8) with empty missing_items
    verifier = FakeVerifier(
        coverage_sequence=[0.5, 0.8],
        missing_items_sequence=[[], []]  # Empty list both times
    )
    responder = FakeResponder()
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )

    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report, timeline = orch.run(state)

    # Should have continued to iteration 2 because coverage was low
    assert verifier.calls == 2
    assert state.control_flags.iteration == 1
    assert state.control_flags.last_verifier_report["coverage_score"] >= 0.7
    assert len(timeline) == 2


def test_orchestrator_continues_on_low_coverage_with_none_missing_items():
    """Test Fix #1: Orchestrator should handle None missing_items gracefully."""
    planner = FakeTaskPlanner([
        [Task(id="task-1", type="repo_research", owner="repo_intelligence_agent")],
        [Task(id="task-2", type="code_context", owner="code_inspector_agent")],
        [],
    ])
    repo_agent = FakeRepoAgent()
    code_agent = FakeCodeAgent()
    # First iteration: low coverage (0.4) with None missing_items
    # Second iteration: good coverage (0.9)
    verifier = FakeVerifier(
        coverage_sequence=[0.4, 0.9],
        missing_items_sequence=[None, None]  # None both times
    )
    responder = FakeResponder()
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )

    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report, timeline = orch.run(state)

    # Should have continued to iteration 2 because coverage was low
    assert verifier.calls == 2
    assert state.control_flags.iteration == 1
    assert state.control_flags.last_verifier_report["coverage_score"] >= 0.7


def test_orchestrator_stops_when_both_coverage_and_missing_items_good():
    """Test that orchestrator stops when coverage is good AND no missing items."""
    planner = FakeTaskPlanner([
        [Task(id="task-1", type="repo_research", owner="repo_intelligence_agent")],
        [],
    ])
    repo_agent = FakeRepoAgent()
    code_agent = FakeCodeAgent()
    # High coverage and no missing items on first iteration
    verifier = FakeVerifier(
        coverage_sequence=[0.85],
        missing_items_sequence=[[]]
    )
    responder = FakeResponder()
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )

    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report, timeline = orch.run(state)

    # Should stop after first iteration
    assert verifier.calls == 1
    assert state.control_flags.iteration == 0
    assert responder.calls == 1

