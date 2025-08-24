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

    # def call_tool(self, rag_response: list, tool_output: Dict[str, Any]) -> Any:
    #     """
    #     Call the correct tool, safeguarding against incorrect arguments.
    #
    #     Args:
    #         rag_response (list): Full RAG results.
    #         tool_output (dict): LLM's tool call.
    #     """
    #     action = tool_output.get("action")
    #     llm_args = tool_output.get("args", {})
    #
    #     if not action:
    #         raise ValueError("Tool output missing 'action' field.")
    #
    #     func = self.tools.get(action)
    #     if func is None:
    #         raise ValueError(f"Unknown tool requested: {action}")
    #
    #     # --- The Guardian Logic ---
    #     # 1. Inspect the function signature to get its parameters
    #     func_params = inspect.signature(func).parameters
    #
    #     # 2. Prepare the arguments to be passed, only including what the function expects
    #     call_args = {}
    #     for param_name, _ in func_params.items():
    #         if param_name in llm_args:
    #             call_args[param_name] = llm_args[param_name]
    #         elif param_name == "rag_response":
    #             # Special case: Inject RAG results if the tool explicitly asks for them
    #             call_args[param_name] = rag_response
    #
    #     try:
    #         # 3. Call the function with the filtered, correct arguments
    #         return func(**call_args)
    #     except Exception as e:
    #         return {"error": f"Tool '{action}' failed: {e}"}

    def call_tool(self, tool_output: Dict[str, Any]) -> Any:
        """
        Call the correct tool, safeguarding against incorrect arguments.

        Args:
            tool_output (dict): LLM's tool call, expected shape:
                {"action": "<tool_name>", "args": { ... }}
        Returns:
            Whatever the tool returns, or an error dict on failure.
        """
        action = tool_output.get("action")
        llm_args = tool_output.get("args", {}) or {}

        if not action:
            raise ValueError("Tool output missing 'action' field.")

        func = self.tools.get(action)
        if func is None:
            raise ValueError(f"Unknown tool requested: {action}")

        # --- Guardian logic: inspect function signature and pass only expected args ---
        try:
            func_params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            # If we can't introspect the function signature, call with raw args (best-effort)
            try:
                return func(**llm_args)
            except Exception as e:
                return {"error": f"Tool '{action}' failed (no-signature fallback): {e}"}

        call_args = {}
        for param_name in func_params.keys():
            if param_name in llm_args:
                call_args[param_name] = llm_args[param_name]

        try:
            return func(**call_args)
        except Exception as e:
            return {"error": f"Tool '{action}' failed: {e}"}


    def add_tool_pointer(self, name: str, func: Callable[..., Any]) -> None:
        """
        Dynamically add a new tool to the manager.

        Args:
            name (str): The action name that the LLM will call.
            func (Callable): The function to register.
        """
        if not callable(func):
            raise ValueError(f"Tool '{name}' must be callable.")
        if name in self.tools:
            raise ValueError(f"Tool '{name}' already exists in the registry.")
        self.tools[name] = func
        print(f"✅ Registered new tool: {name}")