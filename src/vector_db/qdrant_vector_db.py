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
    MatchAny
)

from src.embedding_module.emmbeding_builder import EmbeddingBuilder
from src.utils.profiler import execution_profiler, time_it
from src.utils.helper import load_yaml
from src.utils.logger import logger


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
        logger.info("Initializing QdrantVectorDB")
        connection_cfg = cfg["connection"]
        collection_cfg = cfg["collection_settings"]

        logger.debug("Connection config: %s", {k: v for k, v in connection_cfg.items() if k != "api_key_path"})
        logger.debug("Collection config: %s", collection_cfg)

        if connection_cfg["type"] == "local":
            logger.info("Using local Qdrant client at %s", connection_cfg["url"])
            self.qdrant_client = QdrantClient(
                url=connection_cfg["url"],
                timeout=10
            )
        elif connection_cfg["type"] == "cloud":
            logger.info("Using cloud Qdrant client at %s", connection_cfg["url"])
            self.qdrant_client = QdrantClient(
                url=connection_cfg["url"],
                api_key=load_yaml(connection_cfg["api_key_path"])["key"],
                timeout=30,
                prefer_grpc=True
            )

        # Embedding model
        self.embedding_model = embedding_model
        logger.debug("Embedding model attached: %s", getattr(embedding_model, "model_name", "<unknown>"))

        # Distance metric
        self.distance_metric = self._DISTANCE_MAP[collection_cfg["distance_metric"]]
        logger.info("Distance metric set to %s", collection_cfg["distance_metric"])

        self.upsert_batch_size = collection_cfg["upsert_batch_size"]
        self.embedding_batch_size = collection_cfg["embedding_batch_size"]
        logger.debug("Batch sizes - upsert: %d, embedding: %d", self.upsert_batch_size, self.embedding_batch_size)

    # ------------- public API -------------
    @execution_profiler
    def search_collection(
            self,
            collection_name: str,
            query_code_vector: np.ndarray,
            query_doc_vector: np.ndarray,
            top_k: int = 5,
            filter_conditions: Optional[Dict[str, Union[str, int, float, bool, list]]] = None,
            include_payload: bool = True,
    ):
        logger.debug("search_collection called collection=%s top_k=%d include_payload=%s", collection_name, top_k, include_payload)
        query_filter = (
            self._build_eq_filter(filter_conditions) if filter_conditions else None
        )
        logger.debug("Using query_filter=%s", bool(query_filter))

        # ✅ return top-k per field
        code_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=("code_to_embedded", query_code_vector),
            limit=top_k,
            with_payload=include_payload,
            query_filter=query_filter,
        )
        logger.debug("Code search returned %d hits", len(code_results) if code_results is not None else 0)

        if query_doc_vector:
            doc_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=("doc_to_embedded", query_doc_vector),
                limit=top_k,
                with_payload=include_payload,
                query_filter=query_filter,
            )
            logger.debug("Doc search returned %d hits", len(doc_results) if doc_results is not None else 0)
        else:
            doc_results = None
            logger.debug("Doc vector not provided; skipping doc search")

        return {"code": code_results, "doc": doc_results}

    def set_collection_name(self,
                            collection_name):  # TODO remove self.collection_name , make it fully dynamic per project
        logger.debug("Setting collection_name to %s", collection_name)
        self.collection_name = collection_name

    @time_it
    @execution_profiler
    def create_collection_with_data(
            self,
            documents: Sequence[Dict[str, Any]],
            only_code: bool,
            overwrite_existing: bool = False,
            create_payload_indexes: bool = True,
    ) -> None:
        logger.info("create_collection_with_data called for collection=%s docs=%d only_code=%s overwrite=%s",
                    getattr(self, "collection_name", "<unset>"), len(documents), only_code, overwrite_existing)
        # Validate
        embedding_keys = self._validate_documents(documents, only_code)
        logger.debug("Embedding keys validated: %s", embedding_keys)

        # Create/recreate collection
        vectors_config = self._ensure_collection(embedding_keys, overwrite_existing)
        logger.debug("Vectors config: %s", {k: v.size for k, v in vectors_config.items()})

        # Prepare data
        points = self._build_points(documents, embedding_keys, vectors_config, only_code)
        logger.info("Built %d points for upsert", len(points))

        # Upsert in batches of 100 with progress bar
        if points:
            for i in tqdm(range(0, len(points), self.upsert_batch_size), desc="Upserting to Qdrant"):
                batch = points[i:i + self.upsert_batch_size]
                logger.debug("Upserting batch %d - size %d", i // self.upsert_batch_size, len(batch))
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
            logger.info("Upsert finished for collection %s", self.collection_name)

        # Create indexes
        if create_payload_indexes:
            logger.info("Creating payload indexes for collection %s", self.collection_name)
            self._create_payload_indexes()
            logger.info("Payload indexes created")

    # ------------- private helpers -------------

    def _validate_documents(self, documents: Sequence[Dict[str, Any]], only_code: bool) -> set[str]:
        logger.debug("_validate_documents called with %d documents only_code=%s", len(documents), only_code)
        if not documents:
            logger.error("No documents provided to _validate_documents")
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
            logger.error("No '*_to_embedded' fields found in documents")
            raise KeyError("No '*_to_embedded' fields found in documents")

        logger.debug("Found embedding keys: %s", embedding_keys)
        return embedding_keys

    def _ensure_collection(self, embedding_keys: set[str], overwrite_existing: bool) -> Dict[str, VectorParams]:
        logger.debug("_ensure_collection called embedding_keys=%s overwrite=%s", embedding_keys, overwrite_existing)
        if overwrite_existing and self.qdrant_client.collection_exists(self.collection_name):
            logger.info("Overwrite requested. Deleting existing collection %s", self.collection_name)
            self.qdrant_client.delete_collection(self.collection_name)

        if not self.qdrant_client.collection_exists(self.collection_name):
            logger.info("Creating collection %s with keys %s", self.collection_name, embedding_keys)
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    key: self._vector_params_for_key(key)
                    for key in embedding_keys
                },
                on_disk_payload=True
            )
        else:
            logger.debug("Collection %s already exists", self.collection_name)

        # always rebuild config dict (for use in embedding step)
        cfg = {key: self._vector_params_for_key(key) for key in embedding_keys}
        logger.debug("Rebuilt vectors config for keys: %s", cfg.keys())
        return cfg

    def _vector_params_for_key(self, key: str) -> VectorParams:
        if key.startswith("code"):
            dim = self.embedding_model.get_dim_size_code()
        else:
            dim = self.embedding_model.get_dim_size_text()
        logger.debug("_vector_params_for_key key=%s dim=%d", key, dim)
        return VectorParams(size=dim, distance=Distance.COSINE)

    def _build_points(
            self,
            documents: Sequence[Dict[str, Any]],
            embedding_keys: set[str],
            vectors_config: Dict[str, VectorParams],
            only_code: bool,
    ) -> list[PointStruct]:

        logger.debug("_build_points called documents=%d keys=%s only_code=%s", len(documents), embedding_keys, only_code)
        points: list[PointStruct] = []
        n = len(documents)

        zero_vecs = {k: [0.0] * vectors_config[k].size for k in embedding_keys}
        vectors_by_key: Dict[str, list] = {k: [None] * n for k in embedding_keys}

        def process_key(key: str, is_code: bool):  # TODo jsut return normaly ar result and put  vectors_by_key as param
            logger.debug("process_key start key=%s is_code=%s", key, is_code)
            items: list[tuple[int, str]] = []
            for i, doc in enumerate(documents):
                v = doc.get(key)
                if v is None or str(v).strip() == "":
                    vectors_by_key[key][i] = zero_vecs[key]
                else:
                    items.append((i, str(v)))

            if not items:
                logger.debug("No items to embed for key=%s", key)
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

                logger.debug("Embedding batch for key=%s start=%d size=%d", key, start, len(texts))
                embeddings = batch_fn(texts)
                if len(embeddings) != len(texts):
                    logger.error("Embedding batch size mismatch for key=%s expected=%d got=%d", key, len(texts), len(embeddings))
                    raise RuntimeError(
                        f"Embedding batch size mismatch for key='{key}': "
                        f"expected {len(texts)} got {len(embeddings)}"
                    )

                for idx, emb in zip(indices, embeddings):
                    vectors_by_key[key][idx] = emb
            logger.debug("process_key finished key=%s", key)

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

        logger.debug("Built %d PointStruct points", len(points))
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
            logger.debug("Creating payload index field=%s schema=%s", field_name, schema)
            self.qdrant_client.create_payload_index(
                self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    @staticmethod
    def _build_eq_filter(filters: Dict[str, Union[str, int, float, bool, list]]) -> Filter:
        logger.debug("_build_eq_filter called with filters=%s", filters)
        conds = []
        for k, v in filters.items():
            if isinstance(v, (list, tuple)):
                conds.append(FieldCondition(key=k, match=MatchAny(any=list(v))))
                logger.debug("Built MatchAny for key=%s values=%s", k, v)
            else:
                conds.append(FieldCondition(key=k, match=MatchValue(value=v)))
                logger.debug("Built MatchValue for key=%s value=%s", k, v)
        logger.debug("_build_eq_filter built %d conditions", len(conds))
        return Filter(must=conds)

    def erase_database(self) -> None:
        """
        Permanently delete ALL collections from the connected Qdrant instance.

        DANGER: This cannot be undone. It removes every collection, not just `self.collection_name`.
        """
        logger.warning("erase_database called. This will delete ALL collections on the Qdrant instance.")
        resp = self.qdrant_client.get_collections()
        names = [c.name for c in getattr(resp, "collections", [])]
        logger.info("Found %d collections to delete", len(names))
        for name in names:
            try:
                logger.debug("Deleting collection %s", name)
                self.qdrant_client.delete_collection(name)
            except Exception as e:
                logger.exception("Failed to delete collection %s: %s", name, e)
        # verify
        remaining = getattr(self.qdrant_client.get_collections(), "collections", [])
        if remaining:
            names_rem = [c.name for c in remaining]
            logger.error("Collections remain after erase_database: %s", names_rem)
            raise RuntimeError(f"Collections remain: {names_rem}")
        logger.info("erase_database completed. No collections remain.")
