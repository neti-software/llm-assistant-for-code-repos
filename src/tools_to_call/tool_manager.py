from typing import Any, Dict, Callable
import inspect
from pathlib import Path

from src.tools_to_call.fetch_file_from_patch import fetch_file_from_patch
from src.tools_to_call.fetch_file_from_query import fetch_file_from_query
from src.tools_to_call.fetch_metadata_from_query import fetch_metadata_from_query
from src.tools_to_call.fetch_project_structure import fetch_project_structure
from src.tools_to_call.search_files_with_grep import search_files_with_grep


class ToolManager:
    def __init__(self, path_to_repos: str):
        self._path_to_repos = Path(path_to_repos)

        # Registry of available tools
        self.tools: Dict[str, Callable[..., Any]] = {
            "fetch_file_from_patch": fetch_file_from_patch,
            "fetch_file_from_query": fetch_file_from_query,
            "fetch_metadata_from_query": fetch_metadata_from_query,
            "fetch_project_structure": fetch_project_structure,
            "search_files_with_grep": search_files_with_grep,
        }

        # tools that require root injection
        self._tools_requiring_root = [
            "fetch_file_from_patch",
            "fetch_project_structure",
            "search_files_with_grep",
        ]

        # Guardian check
        self._check_tools_requiring_root()

    def _check_tools_requiring_root(self):
        """Ensure all tools in _tools_requiring_root accept a `root` parameter."""
        for tool_name in self._tools_requiring_root:
            func = self.tools.get(tool_name)
            if func is None:
                raise ValueError(f"Tool '{tool_name}' listed as requiring root but not registered")
            params = inspect.signature(func).parameters
            if "root" not in params:
                raise ValueError(
                    f"Tool '{tool_name}' must accept a 'root' parameter. Found params: {list(params.keys())}"
                )

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

        # auto-inject root if needed
        if action in self._tools_requiring_root:
            llm_args["root"] = self._path_to_repos

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
