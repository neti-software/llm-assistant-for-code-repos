#!/usr/bin/env python3
from pathlib import Path
from typing import List, Optional

# set to directory that contains many repos (edit)
_ROOT = "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/DATA_TO_TEST" # TODO

def fetch_project_structure(repo_name: str) -> List[str]:
    # basic safety: repo_name must be a simple single-folder name
    root = Path(_ROOT + "/" + repo_name)

    if Path(repo_name).anchor or ".." in Path(repo_name).parts or "/" in repo_name or "\\" in repo_name:
        raise ValueError("repo_name must be a single folder name (no slashes or ..).")

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"repo not found: {root}")
    files: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        files.append(rel)
    return sorted(files)

