from typing import List, Dict, Optional, Any
from src.utils.helper import load_yaml
import voyageai


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
        self.client = voyageai.Client(api_key=load_yaml(config["api_key_path"])["key"])
        self.model_name: str = config["model_name"]
        self.prompt_template: str = config["prompt"]

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
                return v.strip()

        # fallback to indexed value + field + collection
        collection = h.get("collection", "<collection>")
        field = h.get("field", "<field>")
        value = h.get("value", "<missing>")
        return f"[{collection}] {field}: {value}"

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
        if not hits:
            return []

        documents = [self._hit_to_document(h) for h in hits]
        q = query
        if instruction:
            q = f"Instruction: {instruction}\nQuery: {query}"
        elif self.prompt_template:
            q = f"Instruction: {self.prompt_template}\nQuery: {query}"

        call_top_k = top_k or len(documents)

        resp = self.client.rerank(
            q,
            documents,
            model=self.model_name,
            top_k=call_top_k,
            truncation=truncation,
        )

        results = getattr(resp, "results", None)
        reranked_hits = []
        for r in results:
            idx = int(getattr(r, "index", -1))
            if 0 <= idx < len(hits):
                hit = hits[idx].copy()
                hit["score"] = getattr(r, "relevance_score", float("-inf"))
                reranked_hits.append(hit)

        return reranked_hits
