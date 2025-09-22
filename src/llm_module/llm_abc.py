from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from src.utils.helper import SingletonABCMeta


class LLMABC(ABC, metaclass=SingletonABCMeta):
    """Abstract base class for all LLM clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """Generate text from LLM.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional arguments including:
                - json_schema: dict for structured JSON output
                - response_format: dict for structured output format

        Returns:
            Tuple of (want_tool: bool, response: dict)
            want_tool is True if the LLM wants to call a tool, False for direct response
        """
        pass
