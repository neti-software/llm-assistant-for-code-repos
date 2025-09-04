from pathlib import Path
from typing import Optional, Dict, Union


def fetch_file_from_patch(
        file_path: str,
        root: Path,
        start: Optional[int] = None,
        end: Optional[int] = None,
) -> Optional[Union[Dict[str, str], str]]:
    path = root / file_path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return {"error": f"Could not read file {path}: {e}"}

    # slice like numpy: start=None → from 0, end=None → to end
    sliced = lines[start:end]
    return "\n".join(sliced)
