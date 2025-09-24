import pytest

from src.langgraph.executor import execute_turn
from src.langgraph.state_models import ConversationState, ConversationBuffer


class DummyToolManager:
    def __init__(self):
        self.tools = {
            "search_files_with_grep": self._search,
            "fetch_file_from_patch": self._fetch,
        }

    def call_tool(self, payload):
        action = payload["action"]
        args = payload.get("args", {})
        func = self.tools.get(action)
        if func is None:
            return {"error": f"unknown tool {action}"}
        return func(**args)

    def _search(self, query: str, **kwargs):  # noqa: D401
        return ["src/foo.py"]

    def _fetch(self, file_path: str, **kwargs):  # noqa: D401
        if file_path == "src/foo.py":
            return "class Foo:\n    pass"
        return ""


@pytest.fixture
def state():
    buffer = ConversationBuffer(
        user_questions={"user_question1": "Explain Foo class"},
        history=[{"iteration": 0, "user_question1": "Explain Foo class"}],
    )
    return ConversationState(conversation=buffer)


def test_execute_turn_returns_final_response(state):
    tool_manager = DummyToolManager()
    result = execute_turn(llm=None, tool_manager=tool_manager, state=state)

    assert "response" in result
    final_response = result["response"]
    assert "Final Answer" in final_response.message
    assert "src/foo.py" in final_response.message or "Foo" in final_response.message
