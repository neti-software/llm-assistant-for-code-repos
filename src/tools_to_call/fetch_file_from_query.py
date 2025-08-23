from typing import Optional, Dict, Union
from src.tools_to_call.fetch_metadata_from_query import fetch_metadata_from_query
from src.tools_to_call.fetch_file_from_patch import fetch_file_from_patch

def fetch_file_from_query(rag_response, query_number: int) -> Optional[Union[Dict[str, str], str]]:
    metadata = fetch_metadata_from_query(rag_response, query_number)
    repo = metadata.get("repo")
    file_path = metadata.get("path")
    if not (repo and file_path):
        return {"error": "Missing repo/path in metadata"}
    return fetch_file_from_patch(f"{repo}/{file_path}")

