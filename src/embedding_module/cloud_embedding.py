from typing import List, Dict, Any, Union
from openai import OpenAI
import tiktoken
from src.embedding_module.embedding_abc import EmbeddingABC
from src.utils.helper import load_yaml
import os


class CloudEmbedding(EmbeddingABC):
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
        self.batch_size = int(config.get("batch_size", 256))
        self.dim_size = int(config.get("dim")) if config.get("dim") is not None else None

        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8192 - 2000 # TODO , some spaces....

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        items = self._ensure_list(texts)
        if len(items) == 0:
            return [] if isinstance(texts, list) else []

        out: List[List[float]] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch = self._trim_texts_to_max_tokens(batch)
            resp = self.client.embeddings.create(model=self.model_name, input=batch, dimensions=self.dim_size)
            for entry in resp.data:
                vec = entry.embedding
                if self.dim_size is not None and len(vec) != self.dim_size:
                    raise ValueError(
                        f"Embedding dimension mismatch: model returned {len(vec)} but config.dim={self.dim_size}"
                    )
                out.append(vec)

        return out[0] if isinstance(texts, str) else out


    def _trim_texts_to_max_tokens(self, texts: List[str]) -> List[str]:
        def trim_one(text: str) -> str:
            s = text or ""
            if len(self.enc.encode(s)) <= self.max_tokens:
                return s

            lo, hi = 0, len(s)
            best = ""
            while lo < hi:
                mid = (lo + hi) // 2
                cand = s[:mid]
                if len(self.enc.encode(cand)) <= self.max_tokens:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid
            return best if best else s[:1]

        return [trim_one(t) for t in texts]

    @staticmethod
    def _ensure_list(x: Union[str, List[str]]) -> List[str]: # TODO ... blend it to embed func?
        return [x] if isinstance(x, str) else list(x)
