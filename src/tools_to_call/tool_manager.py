import json
from typing import Any, Dict, Callable
import inspect

from src.tools_to_call.fetch_file_from_patch import fetch_file_from_patch
from src.tools_to_call.fetch_file_from_query import fetch_file_from_query
from src.tools_to_call.fetch_metadata_from_query import fetch_metadata_from_query


class ToolManager:
    def __init__(self):
        # Registry of available tools
        self.tools: Dict[str, Callable[..., Any]] = {
            "fetch_file_from_patch": fetch_file_from_patch,
            "fetch_file_from_query": fetch_file_from_query,
            "fetch_metadata_from_query": fetch_metadata_from_query,
        }

    def call_tool(self, rag_response: list, tool_output: Dict[str, Any]) -> Any:
        """
        Call the correct tool, safeguarding against incorrect arguments.

        Args:
            rag_response (list): Full RAG results.
            tool_output (dict): LLM's tool call.
        """
        action = tool_output.get("action")
        llm_args = tool_output.get("args", {})

        if not action:
            raise ValueError("Tool output missing 'action' field.")

        func = self.tools.get(action)
        if func is None:
            raise ValueError(f"Unknown tool requested: {action}")

        # --- The Guardian Logic ---
        # 1. Inspect the function signature to get its parameters
        func_params = inspect.signature(func).parameters

        # 2. Prepare the arguments to be passed, only including what the function expects
        call_args = {}
        for param_name, _ in func_params.items():
            if param_name in llm_args:
                call_args[param_name] = llm_args[param_name]
            elif param_name == "rag_response":
                # Special case: Inject RAG results if the tool explicitly asks for them
                call_args[param_name] = rag_response

        try:
            # 3. Call the function with the filtered, correct arguments
            return func(**call_args)
        except Exception as e:
            return {"error": f"Tool '{action}' failed: {e}"}