from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.utils.helper import load_yaml
import time

# Load configs
qdrant_config = load_yaml('configs/qdrant_config.yaml')
embedding_config = load_yaml('configs/embedding_config.yaml')
reranker_config = load_yaml('configs/reranker_config.yaml')
repo_metadata_manager_config = load_yaml('configs/json_schema/ast/metadata_schema.json')
ignore_patterns_config = load_yaml('configs/ignore_patterns_config.yaml')

# Initialize vector DB manager
manager = ManagerQdrantVectorDb(
    config=qdrant_config,
    embedding_config=embedding_config,
    reranker_config=reranker_config,
    repo_metadata_manager_config=repo_metadata_manager_config,
    ignore_patterns_config=ignore_patterns_config,
)

# Test search with Filecoin+ filter to specific repo
query = "DataCap"
print(f"Testing search with query: {query}")

# Try searching just the filplus-registry collection
results = manager.search(
    query,
    top_k=3,
    positive_filter_conditions={"project": ["filplus-registry", "filplus-utils", "filecoin-plus-leaderboard-data"]}
)
print(f"Search results: {len(results) if results else 0} found")

if results:
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.get('project', 'unknown')} - {result.get('value', 'no content')[:150]}...")
else:
    print("No results found")
