"""Common adapters wrapping ToolManager for LangGraph usage."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_tool_result(
    *,
    tool_name: str,
    args: Dict[str, Any],
    raw_result: Any,
) -> Dict[str, Any]:
    """Normalise tool outputs into a LangGraph-friendly payload."""

    error: Optional[str] = None
    data: Optional[Any] = raw_result

    if isinstance(raw_result, dict) and "error" in raw_result:
        error = str(raw_result.get("error"))
        data = None

    return {
        "tool_name": tool_name,
        "data": data,
        "citations": [],  # citations are collected by specialist agents later
        "confidence": None,
        "error": error,
        "metadata": {
            "args": dict(args),
        },
    }


class ToolNodeAdapter:
    """Thin wrapper around ToolManager for LangGraph nodes."""

    def __init__(self, tool_manager, tool_name: str):
        self._tool_manager = tool_manager
        self._tool_name = tool_name

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def invoke(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the underlying tool via ToolManager and normalise output."""

        payload = {"action": self._tool_name, "args": kwargs}
        raw = self._tool_manager.call_tool(payload)
        return build_tool_result(tool_name=self._tool_name, args=kwargs, raw_result=raw)
