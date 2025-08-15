import yaml
from pathlib import Path
from typing import Union, Dict


# --------- tiny helper ----------
def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML and return a plain dict (raises if file/keys are bad)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping (dict).")
    return data
