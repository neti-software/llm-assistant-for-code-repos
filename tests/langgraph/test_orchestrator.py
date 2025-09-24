import pytest

from src.langgraph.state_models import ConversationState, Task, EvidenceItem
from src.langgraph.orchestrator import Orchestrator


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
    def __init__(self, coverage_sequence):
        self.coverage_sequence = coverage_sequence
        self.calls = 0

    def evaluate(self, state, *, question):
        coverage = self.coverage_sequence[min(self.calls, len(self.coverage_sequence) - 1)]
        self.calls += 1
        missing = [] if coverage >= 0.7 else ["Need more"]
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
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        max_iterations=3,
    )
    return orch


def test_orchestrator_runs_until_verifier_succeeds(orchestrator):
    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report = orchestrator.run(state)

    assert final_report.coverage_score >= 0.7
    assert orchestrator.repo_agent.calls == 1
    assert orchestrator.code_agent.calls == 1
    assert orchestrator.verifier.calls == 2
    assert len(state.tasks) == 2
    assert state.tasks[-1].status == "done"
    assert state.control_flags.iteration == 1


def test_orchestrator_stops_when_iteration_cap_hit():
    planner = FakeTaskPlanner([
        [Task(id="task-1", type="repo_research", owner="repo_intelligence_agent")],
        [],
    ])
    repo_agent = FakeRepoAgent()
    code_agent = FakeCodeAgent()
    verifier = FakeVerifier([0.2, 0.3, 0.2])
    orch = Orchestrator(
        task_planner=planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        max_iterations=2,
    )

    state = ConversationState()
    state.conversation.history.append({"user_question1": "Explain Foo"})

    final_report = orch.run(state)

    assert final_report.coverage_score == 0.3
    assert state.control_flags.iteration == 2

