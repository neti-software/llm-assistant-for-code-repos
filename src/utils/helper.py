import yaml
from pathlib import Path
from typing import Union, Dict, Any
import logging
import traceback
from abc import ABCMeta

logger = logging.getLogger(__name__)


# --------- tiny helper ----------
def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML and return a plain dict (raises if file/keys are bad)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping (dict).")
    return data


class SingletonMeta(type):
    """Metaclass that enforces singleton and warns on duplicate instantiation."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls in cls._instances:
            # log a warning with traceback of where it was called
            stack = "".join(traceback.format_stack(limit=5))  # last few frames
            logger.warning(
                f"[Singleton] Attempt to re-instantiate {cls.__name__}. "
                f"Returning existing instance instead.\nCall trace:\n{stack}"
            )
            return cls._instances[cls]
        instance = super().__call__(*args, **kwargs)
        cls._instances[cls] = instance
        return instance


class SingletonABCMeta(ABCMeta):
    """Metaclass that enforces singleton + ABC compatibility."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls in cls._instances:
            stack = "".join(traceback.format_stack(limit=5))
            logger.warning(
                f"[SingletonABC] Attempt to re-instantiate {cls.__name__}. "
                f"Returning existing instance instead.\nCall trace:\n{stack}"
            )
            return cls._instances[cls]
        instance = super().__call__(*args, **kwargs)
        cls._instances[cls] = instance
        return instance