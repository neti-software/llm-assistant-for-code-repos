"""Bridge module between the legacy llm_loop and the future LangGraph graph."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Dict


class LangGraphRunner:
    """Dispatch chat turns either to the legacy loop or a LangGraph executor."""

    def __init__(
        self,
        *,
        use_graph: bool,
        legacy_llm_loop: Callable[[Any, Any, Any], Any],
    ) -> None:
        if legacy_llm_loop is None:
            raise ValueError("legacy_llm_loop must be provided")
        self.use_graph = use_graph
        self._legacy_llm_loop = legacy_llm_loop
        self._graph_executor: Optional[Callable[[Any, Any, Any], Any]] = None

    def set_graph_executor(
        self, executor: Callable[[Any, Any, Any], Any]
    ) -> None:
        """Register a callable that runs the LangGraph pipeline."""

        self._graph_executor = executor

    def run_turn(self, llm: Any, tool_manager: Any, conversation_history: Any, live_log=None) -> Any:
        """Execute a single chat turn, delegating to the appropriate backend."""

        if self.use_graph and self._graph_executor is not None:
            state = conversation_history.to_state_snapshot()
            response, new_state = self._run_graph(llm, tool_manager, state, live_log)
            if new_state is not None:
                conversation_history.apply_state_delta(new_state)
            return response

        return self._legacy_llm_loop(llm, tool_manager, conversation_history)

    def _run_graph(
        self, llm: Any, tool_manager: Any, state: Any, live_log=None
    ) -> Tuple[Any, Optional[Any]]:
        """Call the graph executor and normalise its output."""

        result = self._graph_executor(llm, tool_manager, state, live_log=live_log)

        response = result
        new_state = None

        if isinstance(result, dict):
            response = result.get("response", result)
            new_state = result.get("state")
        elif isinstance(result, tuple) and len(result) == 2:
            response, new_state = result

        return response, new_state
