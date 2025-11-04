import glob
import os
from src.ast.metadata_extractor_manager import MetadataExtractorManager
from src.vector_db.helpers_vector_db import assemble_function_docs_generic
from src.embedding_module.emmbeding_builder import EmbeddingBuilder
from src.reranker_module.voyager_reranker import VoyagerReranker
from src.vector_db.qdrant_vector_db import QdrantVectorDB
from src.utils.profiler import execution_profiler, time_it
from src.utils.logger import logger
from qdrant_client.http import models as rest
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


class ManagerQdrantVectorDb:
    def __init__(self, config: Dict[str, Any], embedding_config: dict, repo_metadata_manager_config: dict,
                 reranker_config: dict, ignore_patterns_config: dict):
        logger.info("Initializing ManagerQdrantVectorDb")
        self.config = config
        self.embedding_model = EmbeddingBuilder(embedding_config)
        self.reranker_model = VoyagerReranker(reranker_config)

        self.metadata_extractor_manager = MetadataExtractorManager(repo_metadata_manager_config, ignore_patterns_config)

        # Parse connection settings
        self.host_url: str = config["connection"]["url"]
        self.container_name: str = config["connection"].get("container_name", "qdrant")
        self.build_only_code: bool = config["collection_settings"]["build_only_code"]
        logger.info("Connection parsed. host_url=%s container_name=%s build_only_code=%s",
                    self.host_url, self.container_name, self.build_only_code)

        # Parse search settings
        self.use_reranker: bool = config["search_settings"]["use_reranker"]
        self.top_k_multiply: float = config["search_settings"]["top_k_multiply"]
        logger.debug("Search settings loaded: use_reranker=%s top_k_multiply=%.2f",
                     self.use_reranker, self.top_k_multiply)

        self._qdrant_vector_db = QdrantVectorDB(config, self.embedding_model)

    @time_it
    @execution_profiler
    def create_vector_db_from_dir(self, root_repos_dir):
        repos_dir = [d for d in glob.glob(os.path.join(root_repos_dir, "*/")) if os.path.isdir(d)]
        logger.info("Scanning root_repos_dir=%s found %d repos", root_repos_dir, len(repos_dir))
        # cnt = 0
        for repo_path in tqdm(map(Path, repos_dir), desc="Processing repos"):
            # cnt +=1
            # if cnt < 134:
            #     continue
            repo_root = Path(repo_path).resolve()
            logger.info("Processing repo: %s", repo_root)
            metadata_map = self.metadata_extractor_manager.process_repo(repo_root)
            logger.debug("Metadata extracted for %s: %d entries", repo_root.name, len(metadata_map))
            docs = assemble_function_docs_generic(metadata_map, repo_root=repo_root)
            logger.debug("Assembled docs for %s: %d docs", repo_root.name, len(docs))
            self._qdrant_vector_db.collection_name = repo_root.name  # TODO
            if not docs:
                logger.warning("No docs assembled for repo %s. Skipping.", repo_root)
                continue

            logger.info("Creating collection for repo %s (only_code=%s)", repo_root.name, self.build_only_code)
            self._qdrant_vector_db.create_collection_with_data(docs,
                                                               only_code=self.build_only_code,
                                                               overwrite_existing=False)

            logger.info("Collection created for repo %s", repo_root.name)

    @time_it
    @execution_profiler
    def search(self, query: str, top_k: int = 3, per_field: bool = False,
               positive_filter_conditions: dict = None,
               negative_filter_conditions: dict = None,
               diversity: bool = False):  # TODO change to number like give at least 3 diversity repo
        logger.info("Search called. query=%s top_k=%d per_field=%s diversity=%s",
                    query, top_k, per_field, diversity)

        # 1) Get all collections
        collections = self._qdrant_vector_db.qdrant_client.get_collections().collections
        logger.debug("Retrieved %d collections from Qdrant", len(collections) if collections is not None else 0)
        if not collections:
            logger.warning("No collections found in Qdrant.")
            return []

        filtered_collections = self._filter_collections(collections,
                                                        positive_filter_conditions=positive_filter_conditions,
                                                        negative_filter_conditions=negative_filter_conditions)
        logger.debug("Filtered collections count: %d", len(filtered_collections))

        initial_top_k = top_k
        if self.use_reranker:
            logger.debug("Using reranker: initial_top_k=%d multiply=%.2f", top_k, self.top_k_multiply)
            top_k = int(top_k * self.top_k_multiply)
            logger.debug("Adjusted top_k from %d -> %d", initial_top_k, top_k)

        query_code_vector = self.embedding_model.code_embed(query)
        logger.debug("Computed code embedding for query (length=%d)",
                     len(query_code_vector) if hasattr(query_code_vector, "__len__") else -1)
        if per_field:
            query_doc_vector = self.embedding_model.text_embed(query)
            logger.debug("Computed text embedding for query (per_field=True)")
        else:
            query_doc_vector = None

        all_hits = []
        # 3) Loop through collections
        for collection in filtered_collections:
            cname = collection.name
            logger.debug("Searching collection: %s", cname)
            hits = self._qdrant_vector_db.search_collection(
                collection_name=cname,
                query_code_vector=query_code_vector,
                query_doc_vector=query_doc_vector,
                top_k=top_k,
            )

            # 4) Collect results with metadata
            for field, results in hits.items():
                if results:
                    logger.debug("Collection %s field %s returned %d results", cname, field, len(results))
                    for hit in results:
                        score = getattr(hit, "score", None)
                        all_hits.append({
                            "collection": cname,
                            "field": field,
                            "value": hit.payload["metadata"].get(field, "<missing>"),
                            "score": score,
                            "metadata": hit.payload
                        })

        logger.info("Collected total hits across collections: %d", len(all_hits))

        # 5) Keep only global top_k (with optional diversity)
        def _score(h):
            return h["score"] if (h["score"] is not None) else float("-inf")

        sorted_hits = sorted(all_hits, key=_score, reverse=True)
        logger.debug("Sorted hits. total=%d", len(sorted_hits))

        if not diversity:
            top_results = sorted_hits[:top_k]
        else:
            top_results = []
            used_collections = set()
            # 1) pick one top hit per collection
            for h in sorted_hits:
                if h["collection"] not in used_collections:
                    top_results.append(h)
                    used_collections.add(h["collection"])
                if len(top_results) == top_k:
                    break
            # 2) if not enough distinct collections, fill with highest remaining hits
            if len(top_results) < top_k:
                for h in sorted_hits:
                    if h in top_results:
                        continue
                    top_results.append(h)
                    if len(top_results) == top_k:
                        break

        minimalized_results = self._minimalize_rag_results(top_results)
        if self.use_reranker:
            minimalized_results = self.reranker_model.rerank_hits(query=query,
                                                                  hits=minimalized_results,
                                                                  top_k=initial_top_k)

        summary = [
            {
                "rank": rank,
                "project": hit["project"],
                "path": hit["path_to_file"],
                "score": hit["score"],
            }
            for rank, hit in enumerate(minimalized_results, start=1)
        ]
        logger.debug("Top %d search results: %s", initial_top_k, summary)

        logger.info("Top results prepared. returning %d items", len(top_results))
        return minimalized_results

    @time_it
    @execution_profiler
    def search_project_readme(self, query: str, top_k: int = 3):
        # TODO merge code with search() to avoid duplciate code. Low priority
        logger.info("search_project_readme called. query=%s top_k=%d", query, top_k)

        # 1) Get all collections
        collections = self._qdrant_vector_db.qdrant_client.get_collections().collections
        logger.debug("Retrieved %d collections from Qdrant for readme search",
                     len(collections) if collections is not None else 0)
        if not collections:
            logger.warning("No collections found in Qdrant (readme search).")
            return []

        query_code_vector = self.embedding_model.code_embed(query)
        logger.debug("Computed code embedding for query in readme search (length=%d)",
                     len(query_code_vector) if hasattr(query_code_vector, "__len__") else -1)

        initial_top_k = top_k
        if self.use_reranker:
            logger.debug("Using reranker: initial_top_k=%d multiply=%.2f", top_k, self.top_k_multiply)
            top_k = int(top_k * self.top_k_multiply)
            logger.debug("Adjusted top_k from %d -> %d", initial_top_k, top_k)

        all_hits = []
        # 3) Loop through collections
        for collection in collections:
            cname = collection.name
            logger.debug("Searching README in collection: %s", cname)
            hits = self._qdrant_vector_db.search_collection(
                collection_name=cname,
                query_code_vector=query_code_vector,
                query_doc_vector=None,
                top_k=1,
                filter_conditions={"path": ["README.md", "Readme.md", "readme.md"]}  # we search only on top repo readme
            )

            # 4) Collect results with metadata
            for field, results in hits.items():
                if results:
                    logger.debug("Collection %s field %s returned %d readme results", cname, field, len(results))
                    for hit in results:
                        score = getattr(hit, "score", None)
                        all_hits.append({
                            "collection": cname,
                            "field": field,
                            "value": hit.payload["metadata"].get(field, "<missing>"),
                            "score": score,
                            "metadata": hit.payload
                        })

        # 5) Keep only global top_k (with optional diversity)
        def _score(h):
            return h["score"] if (h["score"] is not None) else float("-inf")

        sorted_hits = sorted(all_hits, key=_score, reverse=True)
        logger.debug("Sorted readme hits. total=%d", len(sorted_hits))

        top_results = []
        used_collections = set()
        # 1) pick one top hit per collection
        for h in sorted_hits:
            if h["collection"] not in used_collections:
                top_results.append(h)
                used_collections.add(h["collection"])
            if len(top_results) == top_k:
                break
        # 2) if not enough distinct collections, fill with highest remaining hits
        if len(top_results) < top_k:
            for h in sorted_hits:
                if h in top_results:
                    continue
                top_results.append(h)
                if len(top_results) == top_k:
                    break

        minimalized_results = self._minimalize_rag_results(top_results)
        if self.use_reranker:
            minimalized_results = self.reranker_model.rerank_hits(query=query,
                                                                  hits=minimalized_results,
                                                                  top_k=initial_top_k)

        summary = [
            {
                "rank": rank,
                "project": hit["project"],
                "path": hit["path_to_file"],
                "score": hit["score"],
            }
            for rank, hit in enumerate(minimalized_results, start=1)
        ]
        logger.debug("Top %d README results: %s", initial_top_k, summary)

        logger.info("Top README results prepared. returning %d items", len(top_results))
        return minimalized_results

    @staticmethod
    def _minimalize_rag_results(res, full_mode: bool = False):  # TODO move it, fix that double emtadata
        formated_results = []
        for r in res:
            formated_dcit = {}
            formated_dcit['project'] = r['metadata']['project']
            formated_dcit['path_to_file'] = r['metadata']['project'] + "/" + r['metadata']['path']
            # if full_mode:
            #     formated_dcit['value'] = fetch_file_from_patch(formated_dcit['path_to_file'])
            # else:
            formated_dcit['value'] = r['value']
            if r['field'] == "doc":
                formated_dcit['start_line'] = r['metadata']['metadata']['start_line_documentation']
                formated_dcit['end_line'] = r['metadata']['metadata']['end_line_documentation']
            elif r['field'] == "code":
                formated_dcit['start_line'] = r['metadata']['metadata']['start_line_code']
                formated_dcit['end_line'] = r['metadata']['metadata']['end_line_code']

            formated_dcit['score'] = round(r['score'], 3)

            formated_results.append(formated_dcit)
        logger.debug("Minimalized %d results", len(formated_results))
        return formated_results

    @staticmethod
    def _build_filter_conditions(conditions: dict, positive: bool = True):
        """
        Convert a simple dict {key: value} into Qdrant conditions.
        Supports scalars and lists.
        If positive=True → 'must'
        If positive=False → 'must_not'
        """
        if not conditions:
            logger.debug("_build_filter_conditions called with empty conditions")
            return None

        built = []
        for k, v in conditions.items():
            if isinstance(v, (list, tuple, set)):
                cond = rest.FieldCondition(
                    key=f"metadata.{k}",
                    match=rest.MatchAny(any=list(v))
                )
                logger.debug("Built MatchAny condition for key=%s values=%s", k, v)
            else:
                cond = rest.FieldCondition(
                    key=f"metadata.{k}",
                    match=rest.MatchValue(value=v)
                )
                logger.debug("Built MatchValue condition for key=%s value=%s", k, v)
            built.append(cond)

        logger.debug("_build_filter_conditions built %d conditions positive=%s", len(built), positive)
        return {"must": built} if positive else {"must_not": built}

    def delete_db(self):
        logger.info("Erasing Qdrant database via delete_db()")
        self._qdrant_vector_db.erase_database()

    @staticmethod
    def _filter_collections(collections, positive_filter_conditions=None,
                            negative_filter_conditions=None):  # TODO make it better
        """
        Filter Qdrant collections by project name.
        - conditions must look like {"project": "ecodash"} or {"project": ["ecodash", "acbc"]}
        """

        # positive has priority
        if positive_filter_conditions and "project" in positive_filter_conditions:
            v = positive_filter_conditions["project"]
            allowed = {v} if isinstance(v, str) else set(v)
            logger.debug("Applying positive project filter: %s", allowed)
            filtered = [c for c in collections if c.name in allowed]
            logger.info("Positive filter matched %d collections", len(filtered))
            return filtered

        # else negative
        if negative_filter_conditions and "project" in negative_filter_conditions:
            v = negative_filter_conditions["project"]
            blocked = {v} if isinstance(v, str) else set(v)
            logger.debug("Applying negative project filter: %s", blocked)
            filtered = [c for c in collections if c.name not in blocked]
            logger.info("Negative filter reduced to %d collections", len(filtered))
            return filtered

        # no filters
        logger.debug("No collection filters applied. Returning all %d collections", len(collections))
        return collections
