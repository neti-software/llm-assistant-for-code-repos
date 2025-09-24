from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.utils.helper import load_yaml
import os

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

# Test search with Filecoin+ related query
query = 'How to obtain DataCap in Filecoin+ program'
print('Testing search with query:', query)
results = manager.search(query, top_k=5)
print('Search results:', len(results) if results else 0, 'found')

if results:
    for i, result in enumerate(results):
        print(f'Result {i+1}:')
        print(f'  Project: {result.get("project", "unknown")}')
        print(f'  Path: {result.get("path_to_file", "unknown")}')
        print(f'  Content: {result.get("value", "unknown")[:200]}...')
        print()
else:
    print('No results found - this explains the Evidence from None issue')
