from src.utils.helper import load_yaml
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb


def main():
    print("🔧 Loading configuration files...")

    embedding_config = load_yaml("configs/embedding_config.yaml")
    print("✅ Loaded embedding_config.yaml")

    qdrant_config = load_yaml("configs/qdrant_config.yaml")
    print("✅ Loaded qdrant_config.yaml")

    repos_config = load_yaml("configs/repos_config.yaml")
    print("✅ Loaded repos_config.yaml")

    repo_metadata_manager_config = load_yaml("configs/json_schema/ast/metadata_schema.json")
    print("✅ Loaded metadata_schema.json")

    ignore_patterns_config = load_yaml("configs/ignore_patterns_config.yaml")
    print("✅ Loaded ignore_patterns_config.yaml")

    print("\n📦 Initializing Qdrant Vector DB Manager...")
    manager_qdrant_vector_db = ManagerQdrantVectorDb(
        config=qdrant_config,
        embedding_config=embedding_config,
        repo_metadata_manager_config=repo_metadata_manager_config,
        ignore_patterns_config=ignore_patterns_config,
    )
    print("✅ Qdrant Vector DB Manager initialized")

    print("Are you sure to delete db? Press Y")
    user_input = input().strip()
    if user_input == "Y":
        print("\n🗑️ Deleting existing Qdrant DB (if any)...")
        manager_qdrant_vector_db.delete_db()  # uncomment to actually delete
        print("✅ Database deleted")

        repo_dir = repos_config["path_to_repos"]
        print(f"\n📂 Creating vector DB from directory: {repo_dir}")
        manager_qdrant_vector_db.create_vector_db_from_dir(repo_dir)
        print("✅ Vector DB created successfully")
    else:
        print("❌ Aborted. Database not deleted.")

if __name__ == "__main__":
    main()
