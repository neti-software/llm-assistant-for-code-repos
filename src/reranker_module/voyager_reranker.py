from typing import List, Dict, Any, Optional
import voyageai

from src.utils.helper import load_yaml
from src.utils.logger import logger
from src.utils.profiler import time_it


class VoyagerReranker:
    """Thin wrapper around VoyageAI reranker tailored for RAG search hits.

    Responsibilities
    - load small YAML config (model, api_key path, prompt template)
    - prepare document list from search hits
    - call voyageai.Client.rerank and map relevance scores back into hits

    Notes
    - The class never mutates the original search index mapping. It assumes
      `hits` is a list and that document order is the stable input order.
    - The caller controls whether to call `_minimalize_rag_results` afterwards.
    """

    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing VoyagerReranker with model=%s", config.get("model_name"))
        self.client = voyageai.Client(api_key=load_yaml(config["api_key_path"])["key"])
        self.model_name: str = config["model_name"]
        self.prompt_template: str = config["prompt"]
        logger.debug("VoyagerReranker prompt template length=%d", len(self.prompt_template or ""))

    def _hit_to_document(self, h: Dict) -> str:
        """Compose a single document string from a hit dict.

        Heuristic: include collection, field and value. If metadata contains
        a short 'text' or 'content' field prefer that.
        """
        # prefer explicit content-like fields if present
        md = h.get("metadata", {})
        for candidate in ("text", "content", "body", "snippet"):
            v = md.get(candidate)
            if isinstance(v, str) and v.strip():
                logger.debug("_hit_to_document: using candidate '%s' from metadata", candidate)
                return v.strip()

        # fallback to indexed value + field + collection
        collection = h.get("collection", "<collection>")
        field = h.get("field", "<field>")
        value = h.get("value", "<missing>")
        logger.debug("_hit_to_document: falling back to collection/field/value for collection=%s field=%s", collection,
                     field)
        return f"[{collection}] {field}: {value}"

    @time_it
    def rerank_hits(
            self,
            hits: List[Dict],
            query: str,
            instruction: Optional[str] = None,
            top_k: Optional[int] = None,
            truncation: bool = True,
    ) -> List[Dict]:
        """Rerank hits and return exactly `top_k` results in the same structure.

        Steps:
        - Vector DB may return N × factor candidates.
        - We rerank them all with VoyageAI.
        - Replace their `score` with the reranker score.
        - Return exactly `top_k` hits sorted by rerank score, preserving dict structure.
        """
        logger.info("rerank_hits called hits=%d top_k=%s truncation=%s", len(hits) if hits is not None else 0, top_k,
                    truncation)
        if not hits:
            logger.debug("rerank_hits: empty hits -> returning []")
            return []

        documents = [self._hit_to_document(h) for h in hits]
        logger.debug("rerank_hits: built %d documents for reranker", len(documents))

        q = query
        if instruction:
            q = f"Instruction: {instruction}\nQuery: {query}"
            logger.debug("rerank_hits: using provided instruction")
        elif self.prompt_template:
            q = f"Instruction: {self.prompt_template}\nQuery: {query}"
            logger.debug("rerank_hits: using configured prompt_template")

        call_top_k = top_k or len(documents)
        logger.info("Calling voyageai.rerank model=%s call_top_k=%d", self.model_name, call_top_k)

        resp = self.client.rerank(
            q,
            documents,
            model=self.model_name,
            top_k=call_top_k,
            truncation=truncation,
        )

        results = getattr(resp, "results", None)
        if results is None:
            logger.warning("voyageai.rerank returned no results attribute on response")
        else:
            logger.debug("voyageai.rerank returned %d result entries", len(results))

        reranked_hits = []
        for r in results:
            idx = int(getattr(r, "index", -1))
            if 0 <= idx < len(hits):
                hit = hits[idx].copy()
                hit["score"] = getattr(r, "relevance_score", float("-inf"))
                reranked_hits.append(hit)
            else:
                logger.debug("rerank result index out of range or invalid: %s", getattr(r, "index", None))

        logger.info("rerank_hits completed. returning %d reranked hits", len(reranked_hits))
        return reranked_hits
