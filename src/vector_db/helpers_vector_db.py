from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# -----------------------------
# File read cache (per repo/file)
# -----------------------------
_FILE_CACHE: Dict[Tuple[str, str], List[str]] = {}


def _read_lines(repo_root: Path, rel_path: Union[str, Path]) -> List[str]:
    """
    Read file lines with a small cache. Returns [] if file not found or unreadable.
    """
    rr = str(Path(repo_root).resolve())
    rp = str(rel_path)
    key = (rr, rp)

    if key in _FILE_CACHE:
        return _FILE_CACHE[key]

    try:
        lines = (Path(rr) / rp).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        lines = []

    _FILE_CACHE[key] = lines
    return lines


def build_function_snippet(
    repo_root: Path,
    rel_path: Union[str, Path],
    func_meta: Dict[str, Any],
) -> str:
    """
    Return function code (def+body) for [start_line_code, end_line_code] inclusive.
    If a docstring range is present, remove those lines from the snippet.
    """
    lines = _read_lines(repo_root, rel_path)
    if not lines:
        return ""

    s = int(func_meta.get("start_line_code") or 0)
    e = int(func_meta.get("end_line_code") or 0)
    if s <= 0 or e <= 0 or e < s:
        return ""

    # clamp to file bounds (1-based → 0-based slice)
    s0 = max(1, s)
    e0 = min(len(lines), e)
    if e0 < s0:
        return ""

    block = lines[s0 - 1 : e0]  # slice already copies

    # remove docstring if its span (global line numbers) overlaps this block
    ds = func_meta.get("start_line_doc")
    de = func_meta.get("end_line_doc")
    if isinstance(ds, int) and isinstance(de, int) and ds > 0 and de > 0 and de >= ds:
        rel_s = ds - s0  # relative to block start
        rel_e = de - s0
        # Only delete if strictly within the block
        if 0 <= rel_s <= rel_e < len(block):
            del block[rel_s : rel_e + 1]

    # Preserve original indentation; do not strip trailing spaces (keeps fidelity)
    return "\n".join(block)


# ----------- generic class metadata attach -----------

def _find_class_meta_generic(
    file_meta: Dict[str, Any],
    func_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    If function has 'enclosing_class' and file has 'classes', return the matching class dict.
    Matches by 'symbol_name' or 'name'. Returns {} if none.
    """
    enc = func_meta.get("enclosing_class")
    classes = file_meta.get("classes")
    if not enc or not isinstance(classes, list):
        return {}
    for cls in classes:
        if cls.get("symbol_name") == enc or cls.get("name") == enc:
            return cls
    return {}


# ----------- assemble docs (generic) -----------

def assemble_function_docs_generic(
    metadata_map: Dict[str, Dict[str, Any]],
    repo_root: Union[str, Path],
) -> List[Dict[str, Any]]:
    """
    Build Qdrant-ready documents (functions + classes) in a generic way.
    - For functions: use start/end code lines and strip docstring segment if present.
    - For classes: synthesize a class with indented method snippets and merged docstrings.
    """
    root = Path(repo_root).resolve()
    docs: List[Dict[str, Any]] = []

    for rel_path, file_meta in metadata_map.items():
        if not isinstance(file_meta, dict):
            continue

        global_meta = _extract_global_meta(file_meta)

        # --- functions ---
        funcs = file_meta.get("functions", [])
        if isinstance(funcs, list) and funcs:
            docs.extend(
                _build_function_docs(root, rel_path, funcs, global_meta, file_meta)
            )

        # --- classes ---
        classes = file_meta.get("classes", [])
        if isinstance(classes, list) and classes:
            docs.extend(
                _build_class_docs(root, rel_path, classes, global_meta, file_meta)
            )

    return docs


# ---------------- Helpers ----------------

def _extract_global_meta(file_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract file-level metadata excluding 'functions' and 'classes'.
    Keep neutral names ('docstring' is kept for class/doc merge).
    """
    return {
        k: v
        for k, v in file_meta.items()
        if k not in {"functions", "classes"}
    }


def _build_function_docs(
    repo_root: Path,
    rel_path: Union[str, Path],
    funcs: List[Dict[str, Any]],
    global_meta: Dict[str, Any],
    file_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build documents for all functions in a file."""
    docs: List[Dict[str, Any]] = []
    for func_meta in funcs:
        if not isinstance(func_meta, dict):
            continue

        code_snippet = build_function_snippet(repo_root, rel_path, func_meta)
        if not code_snippet.strip():
            continue

        class_meta = _find_class_meta_generic(file_meta, func_meta)
        metadata = _merge_function_metadata(global_meta, class_meta, func_meta, code_snippet)

        doc = _make_function_doc(func_meta, code_snippet, metadata)
        docs.append(doc)

    return docs


def _merge_function_metadata(
    global_meta: Dict[str, Any],
    class_meta: Dict[str, Any],
    func_meta: Dict[str, Any],
    code_snippet: str,
) -> Dict[str, Any]:
    """
    Merge global, class, and function metadata into one dict.
    Leaves original keys intact; adds normalized 'code' and 'doc'.
    """
    return {
        **global_meta,
        **class_meta,
        **func_meta,
        "code": code_snippet,
        "doc": func_meta.get("docstring", "") or "",
    }


def _make_function_doc(
    func_meta: Dict[str, Any],
    code_snippet: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the final function document for Qdrant."""
    return {
        "code_to_embedded": code_snippet,
        "doc_to_embedded": func_meta.get("docstring", "") or "",
        "metadata": metadata,
    }


def _build_class_docs(
    repo_root: Path,
    rel_path: Union[str, Path],
    classes: List[Dict[str, Any]],
    global_meta: Dict[str, Any],
    file_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build documents for all classes in a file.

    For each class:
      - Collect method code (via build_function_snippet) and indent once (4 spaces).
      - Merge docstrings: file + class + methods.
      - Merge global + class metadata.
      - Produce 'code_to_embedded' and 'doc_to_embedded'.
    """
    docs: List[Dict[str, Any]] = []
    file_docstring = (file_meta.get("docstring") or "").strip()

    for class_meta in classes:
        if not isinstance(class_meta, dict):
            continue

        class_name = class_meta.get("symbol_name") or class_meta.get("name") or "UnknownClass"

        # --- collect code snippets from all methods ---
        method_snippets: List[str] = []
        method_docstrings: List[str] = []

        for method_meta in class_meta.get("methods", []) or []:
            if not isinstance(method_meta, dict):
                continue

            code_snippet = build_function_snippet(repo_root, rel_path, method_meta)
            if code_snippet.strip():
                # indent each line once to form a class block
                indented = "\n".join(("    " + line) if line.strip() else line
                                     for line in code_snippet.splitlines())
                method_snippets.append(indented)

            ds = (method_meta.get("docstring") or "").strip()
            if ds:
                method_docstrings.append(ds)

        # --- construct class-like code ---
        class_header = f"class {class_name}:\n"
        merged_methods = "\n\n".join(method_snippets).rstrip()
        merged_code = class_header + (merged_methods if merged_methods else "    pass")

        # --- merge docstrings ---
        class_docstring = (class_meta.get("docstring") or "").strip()
        merged_doc = "\n\n".join(
            [s for s in [file_docstring, class_docstring] + method_docstrings if s]
        ).strip()

        # --- metadata ---
        metadata = {
            **global_meta,
            **class_meta,
            "code": merged_code,
            "doc": merged_doc,
        }

        # --- final doc ---
        docs.append({
            "code_to_embedded": merged_code,
            "doc_to_embedded": merged_doc,
            "metadata": metadata,
        })

    return docs
