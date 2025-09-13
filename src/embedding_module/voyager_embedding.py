from typing import List, Dict, Any, Union
import voyageai

from src.embedding_module.embedding_abc import EmbeddingABC
from src.utils.helper import load_yaml


class VoyagerEmbedding(EmbeddingABC):
    def __init__(self, config: Dict[str, Any]):
        # Create Voyage client
        self.client = voyageai.Client(api_key=load_yaml(config["api_key_path"])["key"])
        self.model_name = config.get("model_name")
        self.dim_size = int(config["dim"])
        self.max_tokens = int(config["max_tokens"])

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        items = self._ensure_list(texts)
        if len(items) == 0:
            return []

        resp = self.client.embed(model=self.model_name,
                                 texts=texts,
                                 truncation=True,
                                 output_dimension=self.dim_size)
        embeds = getattr(resp, "embeddings", None)

        return embeds[0] if isinstance(texts, str) else embeds

    @staticmethod
    def _ensure_list(x: Union[str, List[str]]) -> List[str]:
        return [x] if isinstance(x, str) else list(x)




