from typing import Dict, Any
from src.llm_module.llm_abc import LLMABC
from openai import OpenAI
from src.utils.profiler import execution_profiler


class CloudLLM(LLMABC):
    """LLM client for cloud providers like OpenAI."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(api_key=config["api_key"])
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.7)

    @execution_profiler
    def generate(self, prompt: str, **kwargs) -> str:
        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return resp.choices[0].text.strip()
