from typing import List, Dict, Any, Union
from src.embedding_module.embedding_abc import EmbeddingABC


class CloudEmbedding(EmbeddingABC):
    """Cloud embedding class for API-based embeddings."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

        self.provider = self.cfg["provider"]
        self.model_name = self.cfg["model_name"]
        self.endpoint = self.cfg["endpoint"]
        self.api_key = self.cfg["api_key"]

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        TODO: Implement actual request logic here.
        """
        raise NotImplementedError(
            f"Cloud embedding not yet implemented for provider={self.provider}"
        )
