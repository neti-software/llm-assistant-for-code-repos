from typing import Dict, Any, Union, List
from pathlib import Path
from src.embedding_module.local_embedding import LocalEmbedding
from src.embedding_module.cloud_embedding import CloudEmbedding


class EmbeddingBuilder:
    """Builder that constructs both Text and Code embeddings from YAML config."""

    def __init__(self, cfg: Dict):
        """
        Initialize the builder and create embeddings.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the YAML configuration file.
        """
        self.cfg: Dict[str, Any] = cfg

        if "text" not in self.cfg or "code" not in self.cfg:
            raise ValueError("Config must contain both 'text' and 'code' sections")

        self._text = self._build_one(self.cfg["text"])
        self._code = self._build_one(self.cfg["code"])

    def _build_one(self, cfg: Dict[str, Any]):
        """Helper to construct one embedding model (local or cloud)."""
        etype = cfg["type"]
        if etype == "local":
            return LocalEmbedding(cfg)
        elif etype == "cloud":
            return CloudEmbedding(cfg)
        else:
            raise ValueError(f"Unknown embedding type: {etype}")

    def text_embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using the text model."""
        return self._text.embed(texts)

    def code_embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using the code model."""
        return self._code.embed(texts)

    def get_dim_size_text(self) -> int:
        return self._text.dim_size

    def get_dim_size_code(self) -> int:
        return self._code.dim_size
