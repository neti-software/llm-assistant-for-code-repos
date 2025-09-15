from typing import Dict, Any, Union, List
from pathlib import Path
from src.embedding_module.local_embedding import LocalEmbedding
from src.embedding_module.openai_embedding import OpenaiEmbedding
from src.embedding_module.voyager_embedding import VoyagerEmbedding
from src.utils.logger import logger


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
        logger.info("EmbeddingBuilder initializing with config keys: %s", list(self.cfg.keys()))

        if "text" not in self.cfg or "code" not in self.cfg:
            logger.error("EmbeddingBuilder config missing 'text' or 'code' sections")
            raise ValueError("Config must contain both 'text' and 'code' sections")

        self._text = self._build_one(self.cfg["text"])
        self._code = self._build_one(self.cfg["code"])
        logger.info("EmbeddingBuilder initialized text=%s code=%s",
                    getattr(self._text, "model_name", type(self._text).__name__),
                    getattr(self._code, "model_name", type(self._code).__name__))

    def _build_one(self, cfg: Dict[str, Any]):
        """Helper to construct one embedding model (local or cloud)."""
        etype = cfg["type"]
        logger.debug("_build_one called type=%s provider=%s", etype, cfg.get("provider"))
        if etype == "local":
            logger.info("Building LocalEmbedding")
            return LocalEmbedding(cfg)
        elif etype == "cloud":
            provider = cfg["provider"]
            if provider == "openai":
                logger.info("Building OpenaiEmbedding")
                return OpenaiEmbedding(cfg)
            elif provider == "voyager":
                logger.info("Building VoyagerEmbedding")
                return VoyagerEmbedding(cfg)
            else:
                logger.error("Unknown provider type: %s", provider)
                raise ValueError(f"Unknown provider type: {provider}")
        else:
            logger.error("Unknown embedding type: %s", etype)
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
