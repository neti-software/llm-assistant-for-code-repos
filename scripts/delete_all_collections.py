#!/usr/bin/env python3
"""
Delete ALL collections from Qdrant - nuclear option to start fresh.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helper import load_yaml  # noqa: E402
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb  # noqa: E402
from src.utils.logger import logger  # noqa: E402


def delete_all_collections(confirm: bool = False) -> None:
    """Delete ALL collections from Qdrant."""
    
    logger.info("🔧 Loading Qdrant manager...")
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
    
    # Get all collections
    response = client.get_collections()
    collections = [c.name for c in getattr(response, "collections", [])]
    
    if not collections:
        logger.info("✅ No collections found in Qdrant. Already clean!")
        return
    
    print("\n" + "="*60)
    print("ALL COLLECTIONS TO DELETE:")
    print("="*60)
    for i, name in enumerate(collections, 1):
        print(f"  {i}. {name}")
    print("="*60)
    
    # Confirm deletion
    if not confirm:
        response = input(f"\n⚠️  ARE YOU SURE? This will delete ALL {len(collections)} collections!\nType 'DELETE ALL' to confirm: ").strip()
        if response != "DELETE ALL":
            logger.info("❌ Aborted. No collections were deleted.")
            return
    
    print(f"\n🗑️  Deleting {len(collections)} collections...\n")
    
    deleted_count = 0
    failed_count = 0
    
    for name in collections:
        try:
            client.delete_collection(name)
            print(f"  ✅ Deleted: {name}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ Failed to delete {name}: {e}")
            failed_count += 1
    
    print("\n" + "="*60)
    print(f"Summary:")
    print(f"  - Successfully deleted: {deleted_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Total: {len(collections)}")
    print("="*60 + "\n")
    
    if failed_count == 0:
        logger.info("✅ All collections deleted successfully!")
    else:
        logger.warning(f"⚠️  {failed_count} collections failed to delete")


if __name__ == "__main__":
    # Use --confirm to skip confirmation prompt
    confirm = "--confirm" in sys.argv
    delete_all_collections(confirm=confirm)

