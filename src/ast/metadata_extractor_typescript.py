import re
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_languages import get_parser


class MetadataExtractorTS:
    """
    Metadata extractor for TypeScript (.ts, .tsx).
    Preserves type annotations in signatures and captures interfaces/type aliases/enums.
    """

    def __init__(self, allow_tsx: bool = True):
        # parser selected per-file in extract; keep default parser instances cached
        self.ts_parser = get_parser("typescript")
        self.tsx_parser = get_parser("tsx") if allow_tsx else None

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src = p.read_bytes()
        src_text = src.decode("utf-8", errors="ignore")

        ext = p.suffix.lower()
        parser = self.tsx_parser if ext == ".tsx" and self.tsx_parser else self.ts_parser

        tree = parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)

        # classes
        classes: List[Dict] = []
        for node in self._find_nodes(root, {"class_declaration"}):
            name = self._decode(src, node.child_by_field_name("name"))
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            methods: List[Dict] = []
            body = node.child_by_field_name("body")
            if body:
                for method in self._find_nodes(body, {"method_definition", "public_field_definition", "property_signature"}):
                    mname = self._decode(src, method.child_by_field_name("name")) or self._node_name_from_text(src, method)
                    params = self._decode(src, method.child_by_field_name("parameters")) or ""
                    sig = self._build_signature_from_text(self._decode(src, method)) or f"{mname}{params}"
                    mdoc, ms, me = self._leading_docstring(src, comments, method)
                    methods.append({
                        "path": rel_path,
                        "symbol_name": mname,
                        "enclosing_class": name,
                        "signature": sig.strip(),
                        "docstring": mdoc,
                        "start_line_documentation": ms,
                        "end_line_documentation": me,
                        "start_line_code": method.start_point[0] + 1,
                        "end_line_code": method.end_point[0] + 1
                    })
            classes.append({
                "path": rel_path,
                "symbol_name": name,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1,
                "methods": methods
            })

        # interfaces / type aliases / enums
        types: List[Dict] = []
        for k in ("interface_declaration", "type_alias_declaration", "enum_declaration"):
            for node in self._find_nodes(root, {k}):
                name = self._node_name_from_text(src, node) or self._decode(src, node.child_by_field_name("name"))
                doc, sdoc, edoc = self._leading_docstring(src, comments, node)
                types.append({
                    "path": rel_path,
                    "symbol_name": name,
                    "kind": k,
                    "docstring": doc,
                    "start_line_documentation": sdoc,
                    "end_line_documentation": edoc,
                    "start_line_code": node.start_point[0] + 1,
                    "end_line_code": node.end_point[0] + 1
                })

        # free functions (function_declaration)
        functions: List[Dict] = []
        for node in self._find_nodes(root, {"function_declaration"}):
            name = self._decode(src, node.child_by_field_name("name")) or self._node_name_from_text(src, node)
            sig = self._build_signature_from_node(src, node)
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            functions.append({
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": None,
                "signature": sig,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1
            })

        # variable declarations with arrow functions or typed function expressions (const/let)
        for node in self._find_nodes(root, {"lexical_declaration", "variable_declaration"}):
            for decl in self._find_nodes(node, {"variable_declarator"}):
                id_node = decl.child_by_field_name("name") or decl.child_by_field_name("identifier")
                init = decl.child_by_field_name("value") or decl.child_by_field_name("initializer")
                name = self._decode(src, id_node) or self._node_name_from_text(src, decl)
                if init and init.type in {"arrow_function", "function", "function_expression"}:
                    params = self._decode(src, init.child_by_field_name("parameters")) or ""
                    sig = self._build_signature_from_text(self._decode(src, decl)) or f"{name}{params}"
                    doc, sdoc, edoc = self._leading_docstring(src, comments, init)
                    functions.append({
                        "path": rel_path,
                        "symbol_name": name,
                        "enclosing_class": None,
                        "signature": sig,
                        "docstring": doc,
                        "start_line_documentation": sdoc,
                        "end_line_documentation": edoc,
                        "start_line_code": decl.start_point[0] + 1,
                        "end_line_code": decl.end_point[0] + 1
                    })
                else:
                    # non-function constant/typed variable
                    if id_node:
                        text = self._decode(src, decl).strip()
                        first_line = text.splitlines()[0] if text else "const"
                        doc, sdoc, edoc = self._leading_docstring(src, comments, decl)
                        functions.append({
                            "path": rel_path,
                            "symbol_name": name or f"const@line{decl.start_point[0] + 1}",
                            "enclosing_class": None,
                            "signature": first_line,
                            "docstring": doc,
                            "start_line_documentation": sdoc,
                            "end_line_documentation": edoc,
                            "start_line_code": decl.start_point[0] + 1,
                            "end_line_code": decl.end_point[0] + 1,
                            "is_constant": True
                        })

        # imports (ESM and import type) and requires
        imports = []
        for m in re.finditer(r'(?m)^\s*import\s+(?:.+?\s+from\s+)?[\'"]([^\'"]+)[\'"]', src_text):
            imports.append(m.group(1))
        for m in re.finditer(r"require\((?:'|\")([^'\"]+)(?:'|\")\)", src_text):
            imports.append(m.group(1))
        imports = list(dict.fromkeys(imports)) if imports else None

        module_doc_end = self._module_docstring_end(src_text)
        file_ext = ".tsx" if ext == ".tsx" else ".ts"

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": file_ext,
            "language": "typescript",
            "namespace": "",
            "doc_kind": "code",
            "module_docstring_end": module_doc_end,
            "imports": imports,
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
                if n.type in {"comment", "line_comment", "block_comment", "comment_block"}:
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

    def _leading_docstring(self, src: bytes, comments: List, target_node) -> (Optional[str], Optional[int], Optional[int]):
        if target_node is None:
            return None, None, None
        t_start_line = target_node.start_point[0]
        close = [c for c in comments if c.end_point[0] >= t_start_line - 1 and c.end_point[0] < t_start_line]
        if not close:
            close = [c for c in comments if t_start_line - 3 <= c.end_point[0] < t_start_line]
            if not close:
                return None, None, None
        close = sorted([c for c in close if c.end_byte <= target_node.start_byte], key=lambda n: n.start_byte)
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
        s = re.sub(r"(?m)^\s*/\*\*\s?", "", s)
        s = re.sub(r"(?m)^\s*//\s?", "", s)
        s = re.sub(r"(?m)^\s*\*\s?", "", s)
        s = re.sub(r"/\*+", "", s)
        s = re.sub(r"\*+/", "", s)
        s = re.sub(r"(?m)^\s*/\s*$", "", s)
        s = s.rstrip(" \t/\n\r")
        return s.strip()

    @staticmethod
    def _module_docstring_end(src_text: str) -> Optional[int]:
        lines = src_text.splitlines()
        end = None
        started = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*(//|\/*\*?)", line):
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

    def _node_name_from_text(self, src: bytes, node) -> str:
        txt = self._decode(src, node)
        if not txt:
            return ""
        m = re.search(r'\b([A-Za-z_]\w*)\s*(?:[:(<\[])', txt)
        if m:
            return m.group(1)
        m2 = re.search(r'\b([A-Za-z_]\w*)\s*\(', txt)
        return m2.group(1) if m2 else ""

    @staticmethod
    def _build_signature_from_text(txt: str) -> str:
        if not txt:
            return ""
        # keep everything up to '{' or ';' and collapse whitespace to preserve types
        txt = txt.split('{')[0].split(';')[0].strip()
        return re.sub(r'\s+', ' ', txt).strip()

    def _build_signature_from_node(self, src: bytes, node) -> str:
        try:
            txt = src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        except Exception:
            return ""
        return self._build_signature_from_text(txt)

