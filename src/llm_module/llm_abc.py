from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.utils.helper import SingletonABCMeta


class LLMABC(ABC, metaclass=SingletonABCMeta):
    """Abstract base class for all LLM clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from LLM."""
        pass
