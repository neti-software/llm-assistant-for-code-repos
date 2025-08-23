from pathlib import Path
from typing import Optional, Dict, Union

_ROOT = "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/DATA_TO_TEST" # TODO

def fetch_file_from_patch(file_path: str) -> Optional[Union[Dict[str, str], str]]:

    path = Path(_ROOT) / file_path
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"Could not read file {path}: {e}"}

    return text

