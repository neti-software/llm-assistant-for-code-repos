from typing import Dict, Any


def pack_metadata(raw_meta: Dict[str, Any]) -> Dict[str, Any]: # TODO
    """
    Reduce raw metadata to only the fields useful for the LLM.
    """
    return {
        "repo": raw_meta.get("repo"),
        "path": raw_meta.get("repo") + "/" + raw_meta.get("path"),
        "symbol_name": raw_meta.get("symbol_name"),
        "symbol_kind": raw_meta.get("symbol_kind"),
        "signature": raw_meta.get("signature"),
        "docstring": raw_meta.get("docstring"),
        "code": raw_meta.get("code"),
        "exports": raw_meta.get("exports", []),
        "enclosing_class": raw_meta.get("enclosing_class"),
    }



def fetch_metadata_from_query(rag_response, query_number: int):
    full_metadata = rag_response[query_number - 1].get("metadata", {})
    return pack_metadata(full_metadata)