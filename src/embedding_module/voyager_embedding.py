from typing import List, Dict, Any, Union
import voyageai
from src.utils.logger import logger
from src.embedding_module.embedding_abc import EmbeddingABC
from src.utils.helper import load_yaml
from src.utils.profiler import time_it


class VoyagerEmbedding(EmbeddingABC):
    def __init__(self, config: Dict[str, Any]):
        # Create Voyage client
        logger.info("Initializing VoyagerEmbedding with config path=%s", config.get("api_key_path"))
        self.client = voyageai.Client(api_key=load_yaml(config["api_key_path"])["key"])
        self.model_name = config.get("model_name")
        self.dim_size = int(config["dim"])
        self.max_tokens = int(config["max_tokens"])
        logger.debug("VoyagerEmbedding model=%s dim=%d max_tokens=%d",
                     self.model_name, self.dim_size, self.max_tokens)

    # TODO fix voyager when using batch.. max is 120k tokens on one go. Safe are batch = 8
    @time_it
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        items = self._ensure_list(texts)
        logger.debug("VoyagerEmbedding.embed called items=%d", len(items))
        if len(items) == 0:
            logger.debug("VoyagerEmbedding.embed received empty items")
            return []

        resp = self.client.embed(model=self.model_name,
                                 texts=texts,
                                 truncation=True,
                                 output_dimension=self.dim_size)
        embeds = getattr(resp, "embeddings", None)
        logger.debug("VoyagerEmbedding.embed returned embeddings=%s", "present" if embeds is not None else "missing")

        return embeds[0] if isinstance(texts, str) else embeds

    @staticmethod
    def _ensure_list(x: Union[str, List[str]]) -> List[str]:
        return [x] if isinstance(x, str) else list(x)
