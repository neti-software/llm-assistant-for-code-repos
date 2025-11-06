#!/usr/bin/env python3
"""
Incremental Qdrant ingestion for freshly downloaded repositories.

This helper mirrors the config-loading logic from ``main_build_db.py`` but adds
several safety switches:

* ``--skip-existing``    – leave collections that already exist in Qdrant.
* ``--overwrite-existing`` – drop and rebuild existing collections.

By default the script behaves like the original manager: it appends new points
to existing collections if they share the same name.
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
from src.vector_db.helpers_vector_db import assemble_function_docs_generic  # noqa: E402
from src.utils.profiler import set_progress_context, clear_progress_context  # noqa: E402


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


def _process_repo(
    manager: ManagerQdrantVectorDb,
    repo_root: Path,
    skip_existing: bool,
    overwrite_existing: bool,
    org_prefix: str | None = None,
) -> None:
    """Ingest a single repository directory into Qdrant."""

    # Build collection name with optional org prefix
    repo_name = repo_root.name
    if org_prefix:
        collection_name = f"{org_prefix}-{repo_name}"
    else:
        collection_name = repo_name
    
    qdrant_db = manager._qdrant_vector_db  # type: ignore[attr-defined]
    client = qdrant_db.qdrant_client

    collection_exists = client.collection_exists(collection_name)
    if collection_exists and skip_existing:
        print(f"[skip] {collection_name} already exists in Qdrant")
        return

    metadata_map = manager.metadata_extractor_manager.process_repo(repo_root)
    docs = assemble_function_docs_generic(metadata_map, repo_root=repo_root)

    if not docs:
        print(f"[warn] {collection_name}: no documents built, skipping")
        return

    qdrant_db.collection_name = collection_name
    qdrant_db.create_collection_with_data(
        docs,
        only_code=manager.build_only_code,
        overwrite_existing=overwrite_existing,
    )

    print(f"[done] {collection_name} indexed ({len(docs)} docs)")


def ingest_directories(
    directories: Iterable[Path],
    skip_existing: bool,
    overwrite_existing: bool,
    use_org_prefix: bool = False,
) -> None:
    """Run ingestion for every provided directory."""
    manager = _load_manager()

    for org_dir in directories:
        if not org_dir.exists():
            print(f"[skip] {org_dir} does not exist")
            continue
        if not org_dir.is_dir():
            print(f"[skip] {org_dir} is not a directory")
            continue

        # Use the org folder name as prefix if requested
        org_prefix = org_dir.name if use_org_prefix else None
        
        # Count total repos upfront
        repo_roots = list(_iter_repo_roots(org_dir))
        total_repos = len(repo_roots)
        
        if use_org_prefix:
            print(f"\n=== Processing {org_dir} (prefix: {org_prefix}) - {total_repos} repos ===")
        else:
            print(f"\n=== Processing {org_dir} - {total_repos} repos ===")
        
        for repo_idx, repo_root in enumerate(repo_roots, start=1):
            collection_name = f"{org_prefix}-{repo_root.name}" if org_prefix else repo_root.name
            set_progress_context(repo_root.name, repo_idx, total_repos)
            _process_repo(manager, repo_root, skip_existing, overwrite_existing, org_prefix)
        
        clear_progress_context()
        print(f"--- Finished {org_dir} ---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upsert embeddings for repositories located inside the provided "
            "directories without wiping the whole Qdrant instance."
        ),
        epilog=(
            "Examples:\n"
            "  # Add with org prefix (recommended to avoid name conflicts)\n"
            "  python scripts/add_new_repos_to_qdrant.py --use-org-prefix "
            "DATA_TO_TEST/new/Arkiv-Network DATA_TO_TEST/new/Golem-Base\n\n"
            "  # Add without prefix (original behavior)\n"
            "  python scripts/add_new_repos_to_qdrant.py "
            "DATA_TO_TEST/new/Arkiv-Network\n\n"
            "  # Skip existing collections\n"
            "  python scripts/add_new_repos_to_qdrant.py --skip-existing "
            "DATA_TO_TEST/new/Arkiv-Network"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories whose immediate children are repositories to ingest.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip repositories whose collections already exist in Qdrant.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Delete and rebuild collections that already exist in Qdrant.",
    )
    parser.add_argument(
        "--use-org-prefix",
        action="store_true",
        help="Prefix collection names with the organization folder name (e.g., 'Arkiv-Network-blockscout').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_existing and args.overwrite_existing:
        raise SystemExit("Choose either --skip-existing or --overwrite-existing, not both.")

    repo_dirs = [Path(d).expanduser().resolve() for d in args.directories]
    ingest_directories(
        repo_dirs,
        args.skip_existing,
        args.overwrite_existing,
        args.use_org_prefix,
    )


if __name__ == "__main__":
    main()
