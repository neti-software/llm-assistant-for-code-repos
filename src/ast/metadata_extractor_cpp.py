import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tree_sitter_languages import get_parser


class MetadataExtractorCAndCpp:
    """
    Extract metadata for C and C++ source files.
    Auto-detect language from file extension (.c/.h -> C, .cpp/.cxx/.cc/.hpp -> C++).
    """

    CPP_EXTS = {".cpp", ".cxx", ".cc", ".c++", ".hpp", ".hh", ".hxx"}
    C_EXTS = {".c", ".h"}

    def __init__(self):
        self.parser = None
        self._current_language = None

    def _init_parser_for(self, language: str):
        if self._current_language == language and self.parser:
            return
        self.parser = get_parser(language)
        self._current_language = language

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src = p.read_bytes()
        src_text = src.decode("utf-8", errors="ignore")
        ext = p.suffix.lower()
        language = "cpp" if ext in self.CPP_EXTS else ("c" if ext in self.C_EXTS else "cpp")
        self._init_parser_for(language)

        tree = self.parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)

        # Collect explicit classes/structs found in the file
        classes: List[Dict] = []
        class_kinds = {"class_specifier", "struct_specifier"} if language == "cpp" else {"struct_specifier"}
        for node in self._find_nodes(root, class_kinds):
            name = self._node_name(src, node) or ""
            doc, sdoc, edoc = self._leading_comment_above(src, node, comments)
            # ensure integers for start/end; fallback to 0 when unknown
            start_code = (node.start_point[0] + 1) if getattr(node, "start_point", None) else 0
            end_code = (node.end_point[0] + 1) if getattr(node, "end_point", None) else 0
            classes.append({
                "path": rel_path,
                "symbol_name": name,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": start_code,
                "end_line_code": end_code,
                "methods": []
            })

        processed_node_ids = set()
        functions: List[Dict] = []

        # First pass: detect functions and qualified methods (Class::method)
        for node in self._find_nodes(root, {"function_definition", "function_declaration"}):
            node_text = self._decode(src, node)
            name_raw = self._node_name(src, node) or ""
            sig = self._build_signature(src, node)
            doc, sdoc, edoc = self._leading_comment_above(src, node, comments)

            # try ancestry
            enclosing = self._find_ancestor(node, {"class_specifier", "struct_specifier"})
            enclosing_name = self._node_name(src, enclosing) if enclosing else None

            # if ancestry not present, try qualified name in node text or signature
            method_name = ""
            if not enclosing_name:
                m = re.search(r'([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)', node_text)
                if not m:
                    m = re.search(r'([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)', sig)
                if m:
                    enclosing_name = m.group(1)
                    method_name = m.group(2)
            # final fallback for method/function name
            if not method_name:
                extracted = self._extract_fn_name_from_sig(sig)
                method_name = name_raw if name_raw else extracted

            # if we discovered an enclosing class name treat as method
            if enclosing_name:
                method_entry = {
                    "path": rel_path,
                    "symbol_name": method_name,
                    "enclosing_class": enclosing_name,
                    "signature": sig,
                    "docstring": doc,
                    "start_line_documentation": sdoc,
                    "end_line_documentation": edoc,
                    "start_line_code": node.start_point[0] + 1,
                    "end_line_code": node.end_point[0] + 1
                }
                attached = False
                for cls in classes:
                    if cls["symbol_name"] == enclosing_name:
                        cls["methods"].append(method_entry)
                        attached = True
                        break
                if not attached and enclosing_name:
                    cls_start = enclosing.start_point[0] + 1 if enclosing and getattr(enclosing, "start_point", None) else 0
                    cls_end = enclosing.end_point[0] + 1 if enclosing and getattr(enclosing, "end_point", None) else 0
                    classes.append({
                        "path": rel_path,
                        "symbol_name": enclosing_name,
                        "docstring": "",
                        "start_line_documentation": None,
                        "end_line_documentation": None,
                        "start_line_code": cls_start,
                        "end_line_code": cls_end,
                        "methods": [method_entry]
                    })
                processed_node_ids.add(node.id)
                continue

            # free/top-level function
            functions.append({
                "path": rel_path,
                "symbol_name": method_name,
                "enclosing_class": None,
                "signature": sig,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1
            })

        # Second pass: attach remaining methods by ancestry (safe)
        for node in self._find_nodes(root, {"function_definition", "function_declaration"}):
            if node.id in processed_node_ids:
                continue
            cls_node = self._find_ancestor(node, {"class_specifier", "struct_specifier"})
            if not cls_node:
                continue
            name = self._node_name(src, node) or ""
            sig = self._build_signature(src, node)
            doc, sdoc, edoc = self._leading_comment_above(src, node, comments)
            cls_name = self._node_name(src, cls_node) or ""
            method_entry = {
                "path": rel_path,
                "symbol_name": name or self._extract_fn_name_from_sig(sig),
                "enclosing_class": cls_name,
                "signature": sig,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1
            }
            attached = False
            for cls in classes:
                if cls["symbol_name"] == cls_name:
                    cls["methods"].append(method_entry)
                    attached = True
                    break
            if not attached and cls_name:
                classes.append({
                    "path": rel_path,
                    "symbol_name": cls_name,
                    "docstring": "",
                    "start_line_documentation": None,
                    "end_line_documentation": None,
                    "start_line_code": cls_node.start_point[0] + 1 if getattr(cls_node, "start_point", None) else 0,
                    "end_line_code": cls_node.end_point[0] + 1 if getattr(cls_node, "end_point", None) else 0,
                    "methods": [method_entry]
                })

        # --- C-specific grouping: attach functions like prefix_name(...) to struct 'prefix' ---
        # compute types once (best-effort)
        found_types: List[str] = []
        for n in self._find_nodes(root, {"typedef_specifier", "enum_specifier", "type_definition"}):
            name = self._node_name(src, n)
            if name:
                found_types.append(name)
        for m in re.finditer(r'(?m)^\s*typedef\s+.+\s+([A-Za-z_]\w*)\s*;', src_text):
            found_types.append(m.group(1))
        found_types = list(dict.fromkeys([t for t in found_types if t])) if found_types else None
        types = found_types

        if language == "c":
            prefix_map = {}
            # build prefix map from function names
            for fn in functions:
                fn_name = fn.get("symbol_name") or self._extract_fn_name_from_sig(fn.get("signature", ""))
                if not fn_name:
                    continue
                m = re.match(r'^([A-Za-z_]\w*)_([A-Za-z_]\w*)$', fn_name)
                if m:
                    prefix = m.group(1)
                    prefix_map.setdefault(prefix, []).append(fn)

            # move functions into classes where appropriate
            new_functions: List[Dict] = []
            for prefix, fns in prefix_map.items():
                match_type = any(t.lower() == prefix.lower() for t in (types or []))
                if len(fns) >= 2 or match_type:
                    cls_name = None
                    if types:
                        for t in types:
                            if t.lower() == prefix.lower():
                                cls_name = t
                                break
                    if not cls_name:
                        cls_name = prefix[0].upper() + prefix[1:] if prefix else prefix

                    # find or create class entry
                    target_cls = None
                    for cls in classes:
                        if cls["symbol_name"] == cls_name:
                            target_cls = cls
                            break
                    if not target_cls:
                        target_cls = {
                            "path": rel_path,
                            "symbol_name": cls_name,
                            "docstring": "",
                            "start_line_documentation": None,
                            "end_line_documentation": None,
                            "start_line_code": 0,
                            "end_line_code": 0,
                            "methods": []
                        }
                        classes.append(target_cls)

                    # move each function into target class as a method
                    for fn in fns:
                        sig = fn.get("signature", "")
                        real_fn_name = fn.get("symbol_name") or self._extract_fn_name_from_sig(sig) or ""
                        m2 = re.match(r'^([A-Za-z_]\w*)_([A-Za-z_]\w*)$', real_fn_name)
                        method_nm = m2.group(2) if m2 else real_fn_name
                        method_entry = {
                            "path": fn.get("path"),
                            "symbol_name": method_nm,
                            "enclosing_class": target_cls["symbol_name"],
                            "signature": fn.get("signature"),
                            "docstring": fn.get("docstring"),
                            "start_line_documentation": fn.get("start_line_documentation"),
                            "end_line_documentation": fn.get("end_line_documentation"),
                            "start_line_code": fn.get("start_line_code"),
                            "end_line_code": fn.get("end_line_code")
                        }
                        target_cls["methods"].append(method_entry)

            # rebuild functions list excluding moved ones
            for fn in functions:
                fn_name = fn.get("symbol_name") or self._extract_fn_name_from_sig(fn.get("signature", ""))
                if fn_name and re.match(r'^([A-Za-z_]\w*)_([A-Za-z_]\w*)$', fn_name):
                    prefix = re.match(r'^([A-Za-z_]\w*)_([A-Za-z_]\w*)$', fn_name).group(1)
                    match_type = any(t.lower() == prefix.lower() for t in (types or []))
                    occurrences = len(prefix_map.get(prefix, []))
                    if occurrences >= 2 or match_type:
                        continue
                new_functions.append(fn)
            functions = new_functions

            # update synthetic class start/end line bounds from attached methods
            for cls in classes:
                if cls.get("methods"):
                    starts = [m.get("start_line_code") for m in cls["methods"] if m.get("start_line_code")]
                    ends = [m.get("end_line_code") for m in cls["methods"] if m.get("end_line_code")]
                    if starts:
                        cls["start_line_code"] = min(starts)
                    else:
                        cls["start_line_code"] = cls.get("start_line_code", 0) or 0
                    if ends:
                        cls["end_line_code"] = max(ends)
                    else:
                        cls["end_line_code"] = cls.get("end_line_code", 0) or 0

        # includes
        includes = []
        for m in re.finditer(r'(?m)^\s*#\s*include\s+[<"]([^">]+)[">]', src_text):
            inc = m.group(1)
            if inc:
                includes.append(inc.strip())
        includes = list(dict.fromkeys(includes)) if includes else None

        module_doc_end = self._module_comment_block_end(src_text)

        file_ext = Path(file_path).suffix or (".cpp" if language == "cpp" else ".c")

        namespaces = self._collect_namespaces(src, root) if language == "cpp" else None
        namespace_field = namespaces[0] if namespaces and len(namespaces) else ""

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": file_ext,
            "language": language,
            "namespace": namespace_field,
            "doc_kind": "code",
            "module_docstring_end": module_doc_end,
            "imports": includes,
            # removed macros and types per request
            "classes": classes if classes else None,
            "functions": functions if functions else None
        }
        return global_meta

    # ---------- helpers ----------
    @staticmethod
    def _decode(src: bytes, node) -> str:
        try:
            return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore") if node else ""
        except Exception:
            return ""

    @staticmethod
    def _find_nodes(root, kinds: set):
        cursor = root.walk()
        seen = set()
        while True:
            n = cursor.node
            if n.id not in seen:
                seen.add(n.id)
                if n.type in kinds:
                    yield n
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return

    def _collect_comment_nodes(self, root):
        nodes = []
        cursor = root.walk()
        seen = set()
        while True:
            n = cursor.node
            if n.id not in seen:
                seen.add(n.id)
                if n.type == "comment":
                    nodes.append(n)
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    break
            else:
                continue
            break
        return nodes

    def _leading_comment_above(self, src: bytes, node, comments: List) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        t_start_line = node.start_point[0]
        close = [c for c in comments if c.end_point[0] >= t_start_line - 3 and c.end_point[0] < t_start_line]
        if not close:
            return None, None, None
        close = sorted([c for c in close if c.end_byte <= node.start_byte], key=lambda n: n.start_byte)
        block = []
        last_line = None
        for c in reversed(close):
            if last_line is None or c.end_point[0] >= last_line - 1:
                block.append(c)
                last_line = c.start_point[0]
            else:
                break
        if not block:
            return None, None, None
        block = list(reversed(block))
        raw = "\n".join(self._decode(src, c) for c in block)
        clean = self._clean_comment_text(raw)
        start_line = block[0].start_point[0] + 1
        end_line = block[-1].end_point[0] + 1
        return clean, start_line, end_line

    @staticmethod
    def _clean_comment_text(s: str) -> str:
        s = re.sub(r"(?m)^\s*//\s?", "", s)
        s = re.sub(r"(?m)^\s*/\*\*?\s?", "", s)
        s = re.sub(r"(?m)\*/\s*$", "", s)
        s = re.sub(r"(?m)^\s*\*\s?", "", s)
        return s.strip()

    def _node_name(self, src: bytes, node) -> str:
        try:
            n = node.child_by_field_name("name")
            if n:
                return self._decode(src, n)
        except Exception:
            pass
        for ch in node.children:
            if ch.type in {"identifier", "type_identifier", "field_identifier", "scoped_identifier", "name"}:
                txt = self._decode(src, ch)
                if txt:
                    return txt
        txt = self._decode(src, node)
        m = re.search(r'\b(class|struct|enum|typedef|namespace)\s+([A-Za-z_]\w*)', txt)
        if m:
            return m.group(2)
        m2 = re.search(r'([A-Za-z_]\w*)\s*\(', txt)
        if m2:
            return m2.group(1)
        return ""

    @staticmethod
    def _has_ancestor_of_type(node, types) -> bool:
        if isinstance(types, (list, set)):
            target = set(types)
        else:
            target = {types}
        cur = node.parent
        while cur is not None:
            if cur.type in target:
                return True
            cur = cur.parent
        return False

    def _find_ancestor(self, node, kinds: set):
        cur = node.parent
        while cur is not None:
            if cur.type in kinds:
                return cur
            cur = cur.parent
        return None

    @staticmethod
    def _build_signature(src: bytes, node) -> str:
        try:
            txt = src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        except Exception:
            return ""
        m = re.search(r'^[\s\S]*?(?=\{|\;)', txt)
        sig = m.group(0).strip() if m else txt.strip().splitlines()[0].strip()
        sig = re.sub(r'\s+', ' ', sig)
        return sig

    @staticmethod
    def _module_comment_block_end(src_text: str) -> Optional[int]:
        lines = src_text.splitlines()
        end = None
        started = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*(//|/\*|\*)", line):
                started = True
                end = i + 1
            elif started and line.strip() == "":
                end = i + 1
                continue
            elif started:
                break
            elif not started and line.strip() == "":
                continue
            else:
                break
        return end

    def _collect_namespaces(self, src: bytes, root) -> Optional[List[str]]:
        names = []
        for n in self._find_nodes(root, {"namespace_definition"}):
            nm = self._node_name(src, n)
            if nm:
                names.append(nm)
        return list(dict.fromkeys(names)) if names else None

    @staticmethod
    def _extract_fn_name_from_sig(sig: str) -> str:
        """Return the function identifier from a C/C++ signature string. Fallback empty."""
        if not sig:
            return ""
        matches = re.findall(r'([A-Za-z_]\w*)\s*\(', sig)
        return matches[-1] if matches else ""
