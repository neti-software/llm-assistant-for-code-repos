from typing import List, Dict, Any, Union
from openai import OpenAI
from langsmith.wrappers import wrap_openai  # minimal tracing for OpenAI embeddings
import tiktoken
from src.embedding_module.embedding_abc import EmbeddingABC
from src.utils.helper import load_yaml
import os


class OpenaiEmbedding(EmbeddingABC):
    """Minimal OpenAI-only cloud embedding wrapper with dim validation."""

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("config must contain 'model_name'")

        api_key = config.get("api_key")
        key_path = config.get("api_key_path") or config.get("path_to_api_key")
        if not api_key and key_path:
            data = load_yaml(key_path)
            api_key = data.get("key") or data.get("api_key") or data.get("openai_api_key")

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key not found in config or OPENAI_API_KEY env")

        self.client = OpenAI(api_key=api_key)
        # Enable LangSmith tracing for OpenAI calls (minimal, env-driven)
        try:
            self.client = wrap_openai(self.client)
        except Exception:
            pass
        self.dim_size = int(config.get("dim")) if config.get("dim") is not None else None

        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8192 - 2000 # TODO , some spaces....

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        items = self._ensure_list(texts)
        if len(items) == 0:
            return [] if isinstance(texts, list) else []

        items = self._trim_texts_to_max_tokens(items)
        resp = self.client.embeddings.create(model=self.model_name, input=items, dimensions=self.dim_size)

        embeds = [data.embedding for data in resp.data]

        return embeds[0] if isinstance(texts, str) else embeds


    def _trim_texts_to_max_tokens(self, texts: List[str]) -> List[str]: # TODO ...
        def trim_one(text: str) -> str:
            if len(self.enc.encode(text)) <= self.max_tokens:
                return text

            lo, hi = 0, len(text)
            best = ""
            while lo < hi:
                mid = (lo + hi) // 2
                cand = text[:mid]
                if len(self.enc.encode(cand)) <= self.max_tokens:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid
            return best if best else text[:1]

        return [trim_one(t) for t in texts]

    @staticmethod
    def _ensure_list(x: Union[str, List[str]]) -> List[str]: # TODO ... blend it to embed func?
        return [x] if isinstance(x, str) else list(x)
