from __future__ import annotations
from typing import Dict, Optional, Sequence, Union, Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.embedding_module.emmbeding_builder import EmbeddingBuilder


class QdrantVectorDB:
    _DISTANCE_MAP = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "ip": Distance.DOT,
        "l2": Distance.EUCLID,
        "euclid": Distance.EUCLID,
        "euclidean": Distance.EUCLID,
    }

    """
    Minimal, YAML-driven Qdrant wrapper.

    Required config keys (no defaults; missing key => KeyError):
      cfg["connection"]["host_url"]              -> str (e.g., "http://127.0.0.1:6333")
      cfg["connection"]["collection_name"]       -> str
      cfg["connection"]["embedding_model"]       -> str (e.g., "BAAI/bge-small-en-v1.5")
      cfg["connection"]["batch_size"]            -> int
      cfg["connection"]["request_timeout_sec"]   -> float/int (seconds)

      cfg["collection_settings"]["distance_metric"]  -> "cosine" | "dot" | "l2"
      cfg["collection_settings"]["vector_size"]      -> int or "auto"
    """

    def __init__(self, cfg: Dict, embedding_model: EmbeddingBuilder):
        connection_cfg = cfg["connection"]
        collection_cfg = cfg["collection_settings"]

        self.qdrant_client = QdrantClient(
            url=connection_cfg["host_url"],
            timeout=float(connection_cfg["request_timeout_sec"])
        )
        self.collection_name = connection_cfg["collection_name"]

        # Embedding model
        self.embedding_model = embedding_model

        # Distance metric
        self.distance_metric = self._DISTANCE_MAP[collection_cfg["distance_metric"]]

    # ------------- public API -------------
    def create_collection_with_data(
        self,
        documents: Sequence[Dict[str, Any]],
        overwrite_existing: bool = False,
        create_payload_indexes: bool = True,
    ) -> None:
        # ---------------- Validate ----------------
        if not documents:
            raise ValueError("No documents provided")

        # Find all embedding fields dynamically
        embedding_keys = {
            key
            for doc in documents
            for key in doc.keys()
            if key.endswith("_to_embedded")
        }
        if not embedding_keys:
            raise KeyError("No '*_to_embedded' fields found in documents")

        # ---------------- Create collection ----------------
        if overwrite_existing and self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)

        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    key: VectorParams(
                        size=(self.embedding_model.get_dim_size_code() if key.startswith("code")
                              else self.embedding_model.get_dim_size_text()),
                        distance=Distance.COSINE,
                    )
                    for key in embedding_keys
                },
            )

        vectors_config = {}
        for key in embedding_keys:
            # decide which embedding model to use
            if key.startswith("code"):
                dim = self.embedding_model.get_dim_size_code()
                vectors_config[key] = VectorParams(size=dim, distance=Distance.COSINE)
            else:
                dim = self.embedding_model.get_dim_size_text()
                vectors_config[key] = VectorParams(size=dim, distance=Distance.COSINE)

        # ---------------- Prepare data ----------------
        points = []
        for doc in documents:
            payload = doc.get("metadata", {})
            vectors = {}

            for key in embedding_keys:
                value = doc.get(key)

                if value is None or str(value).strip() == "":
                    # dummy vector if missing
                    dim = vectors_config[key].size
                    vectors[key] = [0.0] * dim
                else:
                    if key.startswith("code"):
                        vectors[key] = self.embedding_model.code_embed(str(value))
                    else:
                        vectors[key] = self.embedding_model.text_embed(str(value))

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,  # ✅ always dict (multi-vector mode)
                    payload=payload,
                )
            )

        # ---------------- Upsert ----------------
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        # ---------------- Payload indexes ----------------
        if create_payload_indexes:
            fields_to_index = {
                "repo": "keyword",
                "path": "keyword",
                "file_ext": "keyword",
                "language": "keyword",
                "namespace": "keyword",
                "doc_kind": "keyword",
                "tag": "keyword",
                "imports": "keyword",
                "calls": "keyword",
                "symbol_name": "keyword",
            }
            for field_name, schema in fields_to_index.items():
                self.qdrant_client.create_payload_index(
                    self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )

    def search_collection(
            self,
            query_text: str,
            top_k: int = 5,
            filter_conditions: Optional[Dict[str, Union[str, int, float, bool]]] = None,
            include_payload: bool = True,
            per_field: bool = False,  # ✅ new flag
    ):
        query_code_vector = self.embedding_model.code_embed(query_text)
        query_doc_vector = self.embedding_model.text_embed(query_text)

        query_filter = (
            self._build_eq_filter(filter_conditions) if filter_conditions else None
        )

        if per_field:
            # ✅ return top-k per field
            code_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=("code_to_embedded", query_code_vector),
                limit=top_k,
                with_payload=include_payload,
                query_filter=query_filter,
            )
            doc_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=("doc_to_embedded", query_doc_vector),
                limit=top_k,
                with_payload=include_payload,
                query_filter=query_filter,
            )
            return {"code": code_results, "doc": doc_results}

        else:
            # ❌ Qdrant does not support multi-vector search directly.
            # So here we just pick one field as "primary".
            # Later we could implement weighted fusion.
            return self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=("code_to_embedded", query_code_vector),
                limit=top_k,
                with_payload=include_payload,
                query_filter=query_filter,
            )

    # ------------- internals -------------
    def _create_collection(self) -> None:
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_model.get_dim_size_code(), distance=self.distance_metric),
        )

    @staticmethod
    def _build_eq_filter(filters: Dict[str, Union[str, int, float, bool]]) -> Filter:
        return Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        )

    def erase_database(self) -> None:
        """
        Permanently delete ALL collections from the connected Qdrant instance.

        DANGER: This cannot be undone. It removes every collection, not just `self.collection_name`.
        """
        collections = self.qdrant_client.get_collections()
        names = [c.name for c in getattr(collections, "collections", [])]
        for name in names:
            self.qdrant_client.delete_collection(name)
