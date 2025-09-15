#!/usr/bin/env python3
from pathlib import Path
from typing import List, Optional


def fetch_project_structure(repo_name: str, root: Path, ignore_patterns: List[str]) -> List[str]:
    # basic safety: repo_name must be a simple single-folder name
    root = root / repo_name

    if Path(repo_name).anchor or ".." in Path(repo_name).parts or "/" in repo_name or "\\" in repo_name:
        raise ValueError("repo_name must be a single folder name (no slashes or ..).")

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"repo not found: {root}")

    files: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()

        # skip if matches any ignore pattern
        if any(p.match(pattern) or Path(rel).match(pattern) for pattern in ignore_patterns):
            continue

        files.append(rel)

    return sorted(files)
