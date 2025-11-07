#!/usr/bin/env python3
"""
Quick script to delete specific collections from Qdrant by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helper import load_yaml
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb


def delete_collections_by_name(collection_names: list[str], confirm: bool = False) -> None:
    """Delete specified collections from Qdrant."""
    
    print(f"\n{'=' * 50}")
    print(f"Collections to delete ({len(collection_names)}):")
    for name in collection_names:
        print(f"  - {name}")
    print(f"{'=' * 50}")
    
    # Confirm deletion
    if not confirm:
        response = input("\n⚠️  Are you sure you want to DELETE all these collections? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("❌ Aborted. No collections were deleted.")
            return
    
    # Load manager
    print("\n🔧 Loading Qdrant manager...")
    embedding_config = load_yaml("configs/embedding_config.yaml")
    qdrant_config = load_yaml("configs/qdrant_config.yaml")
    metadata_schema = load_yaml("configs/json_schema/ast/metadata_schema.json")
    reranker_config = load_yaml("configs/reranker_config.yaml")
    ignore_patterns = load_yaml("configs/ignore_patterns_config.yaml")

    manager = ManagerQdrantVectorDb(
        config=qdrant_config,
        embedding_config=embedding_config,
        repo_metadata_manager_config=metadata_schema,
        reranker_config=reranker_config,
        ignore_patterns_config=ignore_patterns,
    )
    
    qdrant_db = manager._qdrant_vector_db
    client = qdrant_db.qdrant_client
    
    print("\n🗑️  Deleting collections...")
    deleted_count = 0
    not_found_count = 0
    
    for collection_name in collection_names:
        if client.collection_exists(collection_name):
            try:
                client.delete_collection(collection_name)
                print(f"  ✅ Deleted: {collection_name}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete {collection_name}: {e}")
        else:
            print(f"  ⚠️  Not found: {collection_name}")
            not_found_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  - Successfully deleted: {deleted_count}")
    print(f"  - Not found: {not_found_count}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    # Collections to delete (from your listing)
    collections = [
        ".github",
        "arkiv-sdk-js",
        "arkiv-sdk-python",
        "arkiv-sdk-rust",
        "blockscout",
        "blockscout-be",
        "blockscout-compose-files",
        "blockscout-fe",
        "blockscout-rs",
        "blockscout-rs-neti",
    ]
    
    # Use --confirm to skip confirmation prompt
    confirm = "--confirm" in sys.argv
    delete_collections_by_name(collections, confirm=confirm)

