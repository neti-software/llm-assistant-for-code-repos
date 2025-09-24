from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.utils.helper import load_yaml

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

# Check what collections are available
collections = manager._qdrant_vector_db.qdrant_client.get_collections().collections
print(f"Found {len(collections)} collections:")
for collection in collections:
    print(f"  - {collection.name}")

# Test search with a simple query
query = "Filecoin+"
print(f"\nTesting search with query: {query}")
results = manager.search(query, top_k=3)
print(f"Search results: {len(results) if results else 0} found")
