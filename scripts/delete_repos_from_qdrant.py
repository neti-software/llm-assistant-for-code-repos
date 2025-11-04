#!/usr/bin/env python3
"""
Delete collections from Qdrant for repositories in specified directories.

This script mirrors the logic from ``add_new_repos_to_qdrant.py`` but performs
deletion instead of ingestion.

Safety features:
* Lists all collections that will be deleted before proceeding
* Requires explicit confirmation (--confirm flag or interactive prompt)
* Reports which collections were successfully deleted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helper import load_yaml  # noqa: E402
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb  # noqa: E402


def _load_manager() -> ManagerQdrantVectorDb:
    """Instantiate the Qdrant manager with the standard project configs."""

    embedding_config = load_yaml("configs/embedding_config.yaml")
    qdrant_config = load_yaml("configs/qdrant_config.yaml")
    metadata_schema = load_yaml("configs/json_schema/ast/metadata_schema.json")
    reranker_config = load_yaml("configs/reranker_config.yaml")
    ignore_patterns = load_yaml("configs/ignore_patterns_config.yaml")

    return ManagerQdrantVectorDb(
        config=qdrant_config,
        embedding_config=embedding_config,
        repo_metadata_manager_config=metadata_schema,
        reranker_config=reranker_config,
        ignore_patterns_config=ignore_patterns,
    )


def _iter_repo_roots(root: Path) -> Iterable[Path]:
    """Yield repository directories underneath ``root``."""
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def _collect_collections_to_delete(
    directories: Iterable[Path],
    use_org_prefix: bool = False,
) -> list[str]:
    """
    Collect all collection names that would be deleted based on repo directories.
    Returns a list of collection names (repo names).
    """
    collections_to_delete = []
    
    for org_dir in directories:
        if not org_dir.exists():
            print(f"[skip] {org_dir} does not exist")
            continue
        if not org_dir.is_dir():
            print(f"[skip] {org_dir} is not a directory")
            continue

        org_prefix = org_dir.name if use_org_prefix else None
        
        if use_org_prefix:
            print(f"\n=== Scanning {org_dir} (prefix: {org_prefix}) ===")
        else:
            print(f"\n=== Scanning {org_dir} ===")
        
        for repo_root in _iter_repo_roots(org_dir):
            repo_name = repo_root.name
            if org_prefix:
                collection_name = f"{org_prefix}-{repo_name}"
            else:
                collection_name = repo_name
            
            collections_to_delete.append(collection_name)
            print(f"  - Found: {collection_name}")
    
    return collections_to_delete


def _delete_collections(
    manager: ManagerQdrantVectorDb,
    collection_names: list[str],
    dry_run: bool = False,
) -> None:
    """
    Delete the specified collections from Qdrant.
    
    Args:
        manager: The Qdrant manager instance
        collection_names: List of collection names to delete
        dry_run: If True, only show what would be deleted without actually deleting
    """
    qdrant_db = manager._qdrant_vector_db  # type: ignore[attr-defined]
    client = qdrant_db.qdrant_client
    
    deleted_count = 0
    not_found_count = 0
    
    for collection_name in collection_names:
        if client.collection_exists(collection_name):
            if dry_run:
                print(f"[dry-run] Would delete: {collection_name}")
            else:
                try:
                    client.delete_collection(collection_name)
                    print(f"[deleted] {collection_name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[error] Failed to delete {collection_name}: {e}")
        else:
            print(f"[not-found] {collection_name} (collection does not exist)")
            not_found_count += 1
    
    print(f"\n{'=' * 50}")
    if dry_run:
        print(f"DRY RUN - No collections were actually deleted")
        print(f"Would delete: {deleted_count} collections")
    else:
        print(f"Summary:")
        print(f"  - Successfully deleted: {deleted_count} collections")
        print(f"  - Not found: {not_found_count} collections")


def _list_all_collections(manager: ManagerQdrantVectorDb) -> list[str]:
    """List all collections currently in Qdrant."""
    qdrant_db = manager._qdrant_vector_db  # type: ignore[attr-defined]
    client = qdrant_db.qdrant_client
    
    collections = client.get_collections().collections
    return sorted([c.name for c in collections]) if collections else []


def _verify_remaining_collections(
    manager: ManagerQdrantVectorDb,
    deleted_names: list[str],
) -> None:
    """
    Verify and display which collections remain in Qdrant after deletion.
    Highlights any that were supposed to be deleted but still exist.
    """
    print(f"\n{'=' * 50}")
    print("Verifying remaining collections in Qdrant...")
    print(f"{'=' * 50}")
    
    remaining = _list_all_collections(manager)
    
    if not remaining:
        print("✅ No collections remain in Qdrant.")
        return
    
    print(f"\nFound {len(remaining)} collection(s) still in Qdrant:")
    
    deleted_set = set(deleted_names)
    unexpected_survivors = []
    
    for name in remaining:
        if name in deleted_set:
            # This shouldn't happen - collection was supposed to be deleted
            print(f"  ⚠️  {name} (was supposed to be deleted!)")
            unexpected_survivors.append(name)
        else:
            print(f"  ℹ️  {name}")
    
    if unexpected_survivors:
        print(f"\n⚠️  WARNING: {len(unexpected_survivors)} collection(s) were not deleted successfully!")


def list_all_collections_only() -> None:
    """List all collections in Qdrant without deleting anything."""
    print("🔧 Loading Qdrant manager...")
    manager = _load_manager()
    
    print(f"\n{'=' * 50}")
    print("All collections in Qdrant:")
    print(f"{'=' * 50}")
    
    collections = _list_all_collections(manager)
    
    if not collections:
        print("\n✅ No collections found in Qdrant.")
    else:
        print(f"\nFound {len(collections)} collection(s):")
        for i, name in enumerate(collections, 1):
            print(f"  {i}. {name}")


def delete_repo_collections(
    directories: Iterable[Path],
    confirm: bool = False,
    dry_run: bool = False,
    verify_after: bool = True,
    use_org_prefix: bool = False,
) -> None:
    """
    Delete collections for all repositories in the provided directories.
    
    Args:
        directories: Directories containing repositories
        confirm: If True, skip interactive confirmation
        dry_run: If True, only show what would be deleted
        verify_after: If True, verify remaining collections after deletion
        use_org_prefix: If True, prefix collection names with org folder name
    """
    # Collect collections to delete
    collections_to_delete = _collect_collections_to_delete(directories, use_org_prefix)
    
    if not collections_to_delete:
        print("\n❌ No collections found to delete.")
        return
    
    # Show summary
    print(f"\n{'=' * 50}")
    print(f"Found {len(collections_to_delete)} collection(s) to delete:")
    for name in collections_to_delete:
        print(f"  - {name}")
    print(f"{'=' * 50}")
    
    # Confirm deletion
    if not dry_run and not confirm:
        response = input("\n⚠️  Are you sure you want to delete these collections? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("❌ Aborted. No collections were deleted.")
            return
    
    # Load manager and delete
    print("\n🔧 Loading Qdrant manager...")
    manager = _load_manager()
    
    print("\n🗑️  Deleting collections...")
    _delete_collections(manager, collections_to_delete, dry_run=dry_run)
    
    if not dry_run:
        print("\n✅ Deletion complete!")
        
        # Verify remaining collections
        if verify_after:
            _verify_remaining_collections(manager, collections_to_delete)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete collections from Qdrant for repositories located inside the "
            "provided directories, or list all existing collections."
        ),
        epilog=(
            "Examples:\n"
            "  # List all collections in Qdrant\n"
            "  python scripts/delete_repos_from_qdrant.py --list-all\n\n"
            "  # Delete with org prefix (matches add_new_repos_to_qdrant.py --use-org-prefix)\n"
            "  python scripts/delete_repos_from_qdrant.py --use-org-prefix "
            "DATA_TO_TEST/new/Arkiv-Network DATA_TO_TEST/new/Golem-Base\n\n"
            "  # Dry run to see what would be deleted\n"
            "  python scripts/delete_repos_from_qdrant.py DATA_TO_TEST/new/Arkiv-Network --dry-run\n\n"
            "  # Delete with interactive confirmation\n"
            "  python scripts/delete_repos_from_qdrant.py DATA_TO_TEST/new/Arkiv-Network DATA_TO_TEST/new/Golem-Base\n\n"
            "  # Delete without confirmation prompt\n"
            "  python scripts/delete_repos_from_qdrant.py DATA_TO_TEST/new/Arkiv-Network --confirm\n\n"
            "  # Delete without verification after\n"
            "  python scripts/delete_repos_from_qdrant.py DATA_TO_TEST/new/Arkiv-Network --no-verify\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "directories",
        nargs="*",
        help="Directories whose immediate children are repositories to delete from Qdrant.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip interactive confirmation prompt and proceed with deletion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of remaining collections after deletion.",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all collections currently in Qdrant and exit (no deletion).",
    )
    parser.add_argument(
        "--use-org-prefix",
        action="store_true",
        help="Prefix collection names with the organization folder name (e.g., 'Arkiv-Network-blockscout').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # List mode - just show all collections and exit
    if args.list_all:
        list_all_collections_only()
        return
    
    # Deletion mode - require directories
    if not args.directories:
        print("❌ Error: You must provide directories to delete, or use --list-all to view collections.")
        print("Run with --help for usage information.")
        sys.exit(1)
    
    repo_dirs = [Path(d).expanduser().resolve() for d in args.directories]
    delete_repo_collections(
        repo_dirs,
        confirm=args.confirm,
        dry_run=args.dry_run,
        verify_after=not args.no_verify,
        use_org_prefix=args.use_org_prefix,
    )


if __name__ == "__main__":
    main()

