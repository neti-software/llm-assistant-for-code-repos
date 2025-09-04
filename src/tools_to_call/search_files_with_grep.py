import subprocess
from pathlib import Path
from typing import List, Optional


def search_files_with_grep(pattern: str, root: Path, sub_path: Optional[str] = None) -> List[str]:
    base = root.expanduser().resolve()

    if sub_path in (None, "", "."):
        target = base
    else:
        target = (base / sub_path).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError(f"Invalid sub_path: {sub_path}")

    result = subprocess.run(
        ["rg", "-l", "--hidden", "--no-ignore-vcs", pattern, str(target)],
        capture_output=True, text=True, check=False  # <-- changed
    )

    if result.returncode == 1:
        # no matches
        return []
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ripgrep failed with code {result.returncode}: {result.stderr}")

    files = []
    for f in result.stdout.splitlines():
        try:
            files.append(str(Path(f).resolve().relative_to(base)))
        except Exception:
            files.append(f)
    return sorted(set(files))
