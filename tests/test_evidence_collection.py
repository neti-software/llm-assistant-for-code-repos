from src.langgraph.agents.repo_intelligence import RepoIntelligenceAgent
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

# Test the evidence collection with a mock state
from src.langgraph.state_models import ConversationState

# Create a mock state
state = ConversationState()

# Test the repo intelligence agent directly
from src.langgraph.tool_nodes.base import ToolNodeAdapter

# Create a tool node for rag_search
tool_node = ToolNodeAdapter(manager, "rag_search")

# Test the evidence collection
query = "How to obtain DataCap in Filecoin+ program"
print(f"Testing evidence collection with query: {query}")

# Call the tool directly
tool_result = tool_node.invoke(query=query, top_k=3)

print("Tool result keys:", list(tool_result.keys()))
print("Tool result data length:", len(tool_result.get("data", [])) if tool_result.get("data") else 0)

# Test the conversion process
if tool_result.get("data"):
    agent = RepoIntelligenceAgent([tool_node])
    state, evidence = agent.run(state, query=query)

    print(f"Evidence items collected: {len(evidence)}")
    for i, item in enumerate(evidence):
        print(f"Evidence {i+1}:")
        print(f"  Source: {item.source_path}")
        print(f"  Snippet: {item.snippet[:100]}..." if item.snippet else f"  Snippet: {item.snippet}")
        print(f"  Metadata: {item.metadata}")
        print()
else:
    print("No data in tool result")
