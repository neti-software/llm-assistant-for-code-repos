from fastembed import TextEmbedding
from typing import List, Dict, Any, Union
from src.embedding_module.embedding_abc import EmbeddingABC



class LocalEmbedding(EmbeddingABC):
    """Local embedding class using fastembed.TextEmbedding with strict model_name and safe defaults."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding model from a config dict.

        Parameters
        ----------
        config : Dict[str, Any]
            Required:
              - model_name (str)
            Optional (falls back to fastembed defaults if not provided):
              - cache_dir (str)
              - max_length (int)
              - device (str: "cpu" or "cuda")
        """
        self.cfg = config

        # Required → enforce crash if missing
        model_name = self.cfg["model_name"]

        # Optional → fall back to fastembed defaults if not present
        cache_dir = self.cfg["cache_dir"] if "cache_dir" in self.cfg else None
        max_length = self.cfg["max_length"] if "max_length" in self.cfg else None
        self.device = self.cfg["device"] if "device" in self.cfg else "cpu"

        # Initialize model with only provided values
        kwargs: Dict[str, Any] = {"model_name": model_name}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if max_length is not None:
            kwargs["max_length"] = max_length

        self.model = TextEmbedding(**kwargs)
        self.dim_size = self.model.embedding_size

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for a single string or a list of strings."""
        if isinstance(texts, str):
            return next(iter(self.model.embed([texts]))).tolist()
        return [emb.tolist() for emb in self.model.embed(texts)]

