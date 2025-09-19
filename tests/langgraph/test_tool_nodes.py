import pytest

from src.tools_to_call.tool_manager import ToolManager

from src.langgraph.tool_nodes.base import ToolNodeAdapter


@pytest.fixture
def tool_manager(tmp_path):
    return ToolManager(str(tmp_path), {"default": []})


def test_tool_manager_lists_registered_tools(tool_manager):
    def stub_tool(example: str):
        return {"example": example}

    tool_manager.add_tool_pointer("stub_tool", stub_tool)

    tool_names = {entry["name"] for entry in tool_manager.list_tools()}

    assert {
        "fetch_file_from_patch",
        "fetch_project_structure",
        "search_files_with_grep",
        "stub_tool",
    }.issubset(tool_names)


def test_tool_node_adapter_returns_structured_payload(tool_manager):
    def stub_tool(query: str):
        return {"results": [query]}

    tool_manager.add_tool_pointer("stub_tool", stub_tool)

    adapter = ToolNodeAdapter(tool_manager, "stub_tool")
    result = adapter.invoke(query="hello")

    assert result["tool_name"] == "stub_tool"
    assert result["data"] == {"results": ["hello"]}
    assert result["error"] is None
    assert result["metadata"]["args"] == {"query": "hello"}
    assert result["citations"] == []


def test_tool_node_adapter_handles_error_payload(tool_manager):
    def failing_tool():
        raise RuntimeError("boom")

    tool_manager.add_tool_pointer("fail_tool", failing_tool)

    adapter = ToolNodeAdapter(tool_manager, "fail_tool")
    payload = adapter.invoke()

    assert payload["data"] is None
    assert payload["error"] is not None
    assert "fail_tool" in payload["tool_name"]
    assert payload["metadata"]["args"] == {}
