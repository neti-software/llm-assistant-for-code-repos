from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb

from src.utils.helper import load_yaml
from src.utils.profiler import execution_profiler

# ----------- run -----------

if __name__ == "__main__":
    embedding_config = load_yaml("configs/embedding_config.yaml")
    llm_config = load_yaml("configs/llm_config.yaml")
    qdrant_config = load_yaml("configs/qdrant_config.yaml")

    manager_qdrant_vector_db = ManagerQdrantVectorDb(config=qdrant_config, embedding_config=embedding_config)
    # manager_qdrant_vector_db.create_vector_db_from_dir("DATA_TO_TEST")

    # filter_conditions = {"repo": "Project1"}
    # search_functions("Which function calculate most important points to get later the longest diagonal?", top_k=5, filter_conditions=filter_conditions)
    manager_qdrant_vector_db.search("Which function calculate most important points to get later the longest diagonal?",
                                    top_k=3)

    #
    # search_functions("Which function predict rock", top_k=3)
    execution_profiler.print_info()
