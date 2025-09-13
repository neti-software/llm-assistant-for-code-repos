from __future__ import annotations
from typing import Dict, Optional, Sequence, Union, Any
import uuid
from tqdm import tqdm
import numpy as np

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
from src.utils.profiler import execution_profiler
from src.utils.helper import load_yaml


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

    @execution_profiler
    def __init__(self, cfg: Dict, embedding_model: EmbeddingBuilder):
        connection_cfg = cfg["connection"]
        collection_cfg = cfg["collection_settings"]

        if connection_cfg["type"] == "local":
            self.qdrant_client = QdrantClient(
                url=connection_cfg["url"],
                timeout=10
            )
        elif connection_cfg["type"] == "cloud":
            self.qdrant_client = QdrantClient(
                url=connection_cfg["url"],
                api_key=load_yaml(connection_cfg["api_key_path"])["key"],
                timeout=30,
                prefer_grpc=True
            )

        # Embedding model
        self.embedding_model = embedding_model

        # Distance metric
        self.distance_metric = self._DISTANCE_MAP[collection_cfg["distance_metric"]]

        self.upsert_batch_size = collection_cfg["upsert_batch_size"]
        self.embedding_batch_size = collection_cfg["embedding_batch_size"]

    # ------------- public API -------------
    @execution_profiler
    def search_collection(
            self,
            collection_name: str,
            query_code_vector: np.ndarray,
            query_doc_vector: np.ndarray,
            top_k: int = 5,
            filter_conditions: Optional[Dict[str, Union[str, int, float, bool]]] = None,
            include_payload: bool = True,
    ):

        query_filter = (
            self._build_eq_filter(filter_conditions) if filter_conditions else None
        )

        # ✅ return top-k per field
        code_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=("code_to_embedded", query_code_vector),
            limit=top_k,
            with_payload=include_payload,
            query_filter=query_filter,
        )

        if query_doc_vector:
            doc_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=("doc_to_embedded", query_doc_vector),
                limit=top_k,
                with_payload=include_payload,
                query_filter=query_filter,
            )
        else:
            doc_results = None

        return {"code": code_results, "doc": doc_results}

    def set_collection_name(self,
                            collection_name):  # TODO remove self.collection_name , make it fully dynamic per project
        self.collection_name = collection_name

    @execution_profiler
    def create_collection_with_data(
            self,
            documents: Sequence[Dict[str, Any]],
            only_code: bool,
            overwrite_existing: bool = False,
            create_payload_indexes: bool = True,
    ) -> None:
        # Validate
        embedding_keys = self._validate_documents(documents, only_code)

        # Create/recreate collection
        vectors_config = self._ensure_collection(embedding_keys, overwrite_existing)

        # Prepare data
        points = self._build_points(documents, embedding_keys, vectors_config, only_code)

        # Upsert in batches of 100 with progress bar
        if points:
            for i in tqdm(range(0, len(points), self.upsert_batch_size), desc="Upserting to Qdrant"):
                batch = points[i:i + self.upsert_batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

        # Create indexes
        if create_payload_indexes:
            self._create_payload_indexes()

    # ------------- private helpers -------------

    def _validate_documents(self, documents: Sequence[Dict[str, Any]], only_code: bool) -> set[str]:
        if not documents:
            raise ValueError("No documents provided")

        embedding_keys = {
            key
            for doc in documents
            for key in doc.keys()
            if key.endswith("_to_embedded")
        }

        if only_code:
            embedding_keys = {k for k in embedding_keys if "code" in k}

        if not embedding_keys:
            raise KeyError("No '*_to_embedded' fields found in documents")

        return embedding_keys

    def _ensure_collection(self, embedding_keys: set[str], overwrite_existing: bool) -> Dict[str, VectorParams]:
        if overwrite_existing and self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)

        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    key: self._vector_params_for_key(key)
                    for key in embedding_keys
                },
                on_disk_payload=True
            )

        # always rebuild config dict (for use in embedding step)
        return {key: self._vector_params_for_key(key) for key in embedding_keys}

    def _vector_params_for_key(self, key: str) -> VectorParams:
        if key.startswith("code"):
            dim = self.embedding_model.get_dim_size_code()
        else:
            dim = self.embedding_model.get_dim_size_text()
        return VectorParams(size=dim, distance=Distance.COSINE)

    def _build_points(
            self,
            documents: Sequence[Dict[str, Any]],
            embedding_keys: set[str],
            vectors_config: Dict[str, VectorParams],
            only_code: bool,
    ) -> list[PointStruct]:

        points: list[PointStruct] = []
        n = len(documents)

        zero_vecs = {k: [0.0] * vectors_config[k].size for k in embedding_keys}
        vectors_by_key: Dict[str, list] = {k: [None] * n for k in embedding_keys}

        def process_key(key: str, is_code: bool):  # TODo jsut return normaly ar result and put  vectors_by_key as param
            items: list[tuple[int, str]] = []
            for i, doc in enumerate(documents):
                v = doc.get(key)
                if v is None or str(v).strip() == "":
                    vectors_by_key[key][i] = zero_vecs[key]
                else:
                    items.append((i, str(v)))

            if not items:
                return

            # pick batch function; expect it to exist
            if is_code:
                batch_fn = self.embedding_model.code_embed
            else:
                batch_fn = self.embedding_model.text_embed

            for start in range(0, len(items), self.embedding_batch_size):
                chunk = items[start: start + self.embedding_batch_size]
                indices = [t[0] for t in chunk]
                texts = [t[1] for t in chunk]

                embeddings = batch_fn(texts)
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Embedding batch size mismatch for key='{key}': "
                        f"expected {len(texts)} got {len(embeddings)}"
                    )

                for idx, emb in zip(indices, embeddings):
                    vectors_by_key[key][idx] = emb

        process_key("code_to_embedded", is_code=True)
        if not only_code:
            process_key("doc_to_embedded", is_code=False)

        for i, doc in enumerate(documents):
            metadata = doc.get("metadata", {})
            vecs = {k: (vectors_by_key[k][i] if vectors_by_key[k][i] is not None else zero_vecs[k])
                    for k in embedding_keys}
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vecs,
                    payload={
                        "project": metadata['repo'],
                        "path": metadata['path'],
                        "file_ext": metadata['file_ext'],
                        "language": metadata['language'],
                        "doc_kind": metadata["doc_kind"],
                        "metadata": metadata  # full 20-30 fields
                    },
                )
            )

        return points

    def _create_payload_indexes(self) -> None:
        fields_to_index = {
            "project": "keyword",
            "path": "keyword",
            "file_ext": "keyword",
            "language": "keyword",
            "symbol_kind": "keyword",
            "doc_kind": "keyword",
        }
        for field_name, schema in fields_to_index.items():
            self.qdrant_client.create_payload_index(
                self.collection_name,
                field_name=field_name,
                field_schema=schema,
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
        resp = self.qdrant_client.get_collections()
        names = [c.name for c in getattr(resp, "collections", [])]
        for name in names:
            try:
                self.qdrant_client.delete_collection(name)
            except Exception as e:
                print(f"failed to delete {name}: {e}")
        # verify
        remaining = getattr(self.qdrant_client.get_collections(), "collections", [])
        if remaining:
            raise RuntimeError(f"Collections remain: {[c.name for c in remaining]}")
