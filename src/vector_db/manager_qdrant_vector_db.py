import subprocess
import shlex
import time
import requests
import glob
import os
from heapq import nlargest
from src.ast.repo_metadata_manager import MetadataExtractorManager
from src.vector_db.helpers_vector_db import *
from src.embedding_module.emmbeding_builder import EmbeddingBuilder
from src.vector_db.qdrant_vector_db import QdrantVectorDB
from src.utils.profiler import execution_profiler


class ManagerQdrantVectorDb:
    def __init__(self, config: Dict[str, Any], embedding_config: dict):
        self.config = config
        self.embedding_model = EmbeddingBuilder(embedding_config)

        self.metadata_extractor_manager = MetadataExtractorManager()

        # Parse connection settings
        self.host_url: str = config["connection"]["host_url"]
        self.container_name: str = config["connection"].get("container_name", "qdrant")

        # Derive port from host_url (assume format http://host:port)
        self.port: int = int(self.host_url.split(":")[-1])

        # Run Qdrant if needed
        # self._run_docker()
        # self._wait_for_ready()

        self._qdrant_vector_db = QdrantVectorDB(config, self.embedding_model)

    def _run_docker(self):
        """Force remove any existing Qdrant container and restart clean."""
        print(f"[Qdrant] Nuking old containers and restarting '{self.container_name}' on port {self.port}...")

        # 1. Stop + remove any container with the same name
        subprocess.run(shlex.split(f"docker stop {self.container_name}"), check=False)
        subprocess.run(shlex.split(f"docker rm -f {self.container_name}"), check=False)

        # 2. Kill anything else hogging the port (e.g. crashed container or zombie)
        try:
            subprocess.run(shlex.split(f"fuser -k {self.port}/tcp"), check=False)
        except Exception:
            pass  # ignore if fuser not available

        # 3. Run new one
        docker_cmd = (
            f"docker run -d --rm "
            f"-p {self.port}:6333 "
            f"--name {self.container_name} "
            f"qdrant/qdrant"
        )
        print(f"[Qdrant] Running: {docker_cmd}")
        subprocess.run(shlex.split(docker_cmd), check=True)
        print(f"[Qdrant] Container '{self.container_name}' started fresh on port {self.port}")

    def _wait_for_ready(self, timeout: int = 20):
        """Wait until Qdrant is responding on HTTP API."""
        url = f"http://localhost:{self.port}/healthz"
        print(f"[Qdrant] Waiting for Qdrant to be ready at {url} ...")

        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    print("[Qdrant] Ready ✅")
                    return
            except Exception:
                pass
            time.sleep(1)

        raise TimeoutError("Qdrant did not become ready in time.")

    def stop(self):
        """Stop the Qdrant container."""
        print(f"[Qdrant] Stopping container {self.container_name} ...")
        subprocess.run(shlex.split(f"docker stop {self.container_name}"), check=False)

    @execution_profiler
    def create_vector_db_from_dir(self, root_repos_dir):
        repos_dir = [d for d in glob.glob(os.path.join(root_repos_dir, "*/")) if os.path.isdir(d)]
        for repo_path in repos_dir:
            repo_root = Path(repo_path).resolve()
            metadata_map = self.metadata_extractor_manager.process_repo(repo_root)
            docs = assemble_function_docs_generic(metadata_map, repo_root=repo_root)
            self._qdrant_vector_db.collection_name = repo_root.name
            self._qdrant_vector_db.create_collection_with_data(docs,
                                                               overwrite_existing=True)  # TODO overwrite_existing?

    @execution_profiler
    def search(self, query: str, top_k: int = 3, per_field: bool = True, filter_conditions: dict = None):
        print(f"\n🔍 Searching for: {query!r}")

        # 1) Get all collections
        collections = self._qdrant_vector_db.qdrant_client.get_collections().collections
        if not collections:
            print("⚠️ No collections found in Qdrant.")
            return []

        all_hits = []

        # 2) Loop through collections
        for collection in collections:
            cname = collection.name
            hits = self._qdrant_vector_db.search_collection(
                collection_name=cname,
                query_text=query,
                top_k=top_k,  # local top_k per collection
                per_field=per_field,
                filter_conditions=filter_conditions,
            )

            # 3) Collect results with metadata
            for field, results in hits.items():
                for hit in results:
                    score = getattr(hit, "score", None)
                    all_hits.append({
                        "collection": cname,
                        "field": field,
                        "value": hit.payload["metadata"].get(field, "<missing>"),
                        "score": score,
                        "metadata": hit.payload
                    })

        # 4) Keep only global top_k
        top_results = nlargest(top_k, all_hits, key=lambda x: x["score"] or -1e9)

        # 5) Print nicely
        print(f"\n🏆 Global Top {top_k} Results:")
        for rank, hit in enumerate(top_results, start=1):
            print(f"  {rank}. [{hit['collection']}] {hit['field']} = {hit['value']}  (score={hit['score']:.4f})")

        return top_results

    def delete_db(self):
        self._qdrant_vector_db.erase_database()
