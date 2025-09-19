import pytest

from src.langgraph.runner import LangGraphRunner


class FakeState:
    def __init__(self, label="snapshot"):
        self.label = label


class FakeHistory:
    def __init__(self):
        self.snapshot_requested = False
        self.applied_state = None
        self.snapshot = FakeState()

    def to_state_snapshot(self):
        self.snapshot_requested = True
        return self.snapshot

    def apply_state_delta(self, state):
        self.applied_state = state


def test_runner_uses_legacy_when_flag_disabled():
    calls = []

    def legacy(llm, tool_manager, conversation_history):
        calls.append((llm, tool_manager, conversation_history))
        return "legacy"

    runner = LangGraphRunner(use_graph=False, legacy_llm_loop=legacy)
    history = FakeHistory()

    result = runner.run_turn("llm", "tools", history)

    assert result == "legacy"
    assert len(calls) == 1
    assert not history.snapshot_requested


def test_runner_invokes_graph_executor_and_applies_state():
    legacy_calls = []

    def legacy(llm, tool_manager, conversation_history):
        legacy_calls.append(True)
        return "legacy"

    graph_calls = []

    def graph_executor(llm, tool_manager, state):
        graph_calls.append(state)
        return {
            "response": "graph",
            "state": FakeState("updated"),
        }

    runner = LangGraphRunner(use_graph=True, legacy_llm_loop=legacy)
    runner.set_graph_executor(graph_executor)

    history = FakeHistory()
    result = runner.run_turn("llm", "tools", history)

    assert result == "graph"
    assert graph_calls == [history.snapshot]
    assert isinstance(history.applied_state, FakeState)
    assert history.applied_state.label == "updated"
    assert legacy_calls == []


def test_runner_falls_back_when_graph_missing():
    legacy_calls = []

    def legacy(llm, tool_manager, conversation_history):
        legacy_calls.append(True)
        return "legacy"

    runner = LangGraphRunner(use_graph=True, legacy_llm_loop=legacy)
    history = FakeHistory()
    result = runner.run_turn("llm", "tools", history)

    assert result == "legacy"
    assert history.snapshot_requested is False
    assert legacy_calls == [True]


def test_runner_handles_plain_response_and_no_state():
    legacy_calls = []

    def legacy(llm, tool_manager, conversation_history):
        legacy_calls.append(True)
        return "legacy"

    def graph_executor(llm, tool_manager, state):
        return "graph"

    runner = LangGraphRunner(use_graph=True, legacy_llm_loop=legacy)
    runner.set_graph_executor(graph_executor)

    history = FakeHistory()
    result = runner.run_turn("llm", "tools", history)

    assert result == "graph"
    assert history.applied_state is None
    assert legacy_calls == []
