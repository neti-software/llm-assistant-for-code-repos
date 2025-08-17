from typing import Dict, Any
from src.llm_module.local_llm import LocalLLM
from src.llm_module.cloud_llm import CloudLLM


def build_llm(config: Dict[str, Any]):
    llm_type = config["type"]
    if llm_type == "local":
        return LocalLLM(config)
    elif llm_type == "cloud":
        return CloudLLM(config)
    else:
        raise ValueError(f"Unknown llm type: {llm_type}")
