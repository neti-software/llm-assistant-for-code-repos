import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tree_sitter_languages import get_parser


class MetadataExtractorHeaders:
    """
    Extract metadata from C/C++ header files (.h, .hpp, .hh, .hxx).
    Focuses on declarations: classes/structs, method declarations, free function declarations,
    typedefs/enums, includes and top comment block.
    """

    HEADER_CPP_EXTS = {".hpp", ".hh", ".hxx"}
    HEADER_C_EXTS = {".h"}

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
        # heuristics: explicit cpp header exts => cpp, plain .h try to detect via tokens
        language = "cpp" if ext in self.HEADER_CPP_EXTS else "c"
        # for .h attempt to detect C++ if file contains "class" or "namespace"
        if ext in self.HEADER_C_EXTS:
            if re.search(r'\b(class|namespace|template|operator|constexpr)\b', src_text):
                language = "cpp"
        self._init_parser_for(language)

        tree = self.parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)

        classes: List[Dict] = []
        # collect class/struct declarations (with inline method declarations)
        class_kinds = {"class_specifier", "struct_specifier"} if language == "cpp" else {"struct_specifier"}
        for node in self._find_nodes(root, class_kinds):
            name = self._node_name(src, node) or ""
            doc, sdoc, edoc = self._leading_comment_above(src, node, comments)
            start_code = (node.start_point[0] + 1) if getattr(node, "start_point", None) else 0
            end_code = (node.end_point[0] + 1) if getattr(node, "end_point", None) else 0
            cls_entry = {
                "path": rel_path,
                "symbol_name": name,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": start_code,
                "end_line_code": end_code,
                "methods": []
            }

            # find method-like declarations inside class body
            for child in node.children:
                for fn in self._find_nodes(child, {"function_declarator", "field_declaration", "function_declaration", "declaration"}):
                    # decode candidate text and build signature
                    txt = self._decode(src, fn)
                    sig = self._build_signature_from_text(txt)
                    # try find a name
                    name_fn = self._extract_fn_name_from_sig(sig) or self._node_name(src, fn) or ""
                    # leading doc inside class: try immediate preceding comments relative to fn
                    fdoc, fsdoc, fedoc = self._leading_comment_above(src, fn, comments)
                    # append method
                    cls_entry["methods"].append({
                        "path": rel_path,
                        "symbol_name": name_fn,
                        "enclosing_class": name,
                        "signature": sig,
                        "docstring": fdoc,
                        "start_line_documentation": fsdoc,
                        "end_line_documentation": fedoc,
                        "start_line_code": fn.start_point[0] + 1 if getattr(fn, "start_point", None) else 0,
                        "end_line_code": fn.end_point[0] + 1 if getattr(fn, "end_point", None) else 0
                    })

            classes.append(cls_entry)

        # free (top-level) function declarations
        functions: List[Dict] = []
        for node in self._find_nodes(root, {"function_declaration", "declaration"}):
            # skip declarations that are inside classes (we already captured)
            if self._has_ancestor_of_type(node, class_kinds):
                continue
            sig = self._build_signature_from_node(src, node)
            if not sig:
                continue
            name = self._extract_fn_name_from_sig(sig) or self._node_name(src, node) or ""
            doc, sdoc, edoc = self._leading_comment_above(src, node, comments)
            functions.append({
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": None,
                "signature": sig,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1 if getattr(node, "start_point", None) else 0,
                "end_line_code": node.end_point[0] + 1 if getattr(node, "end_point", None) else 0
            })

        # includes
        includes = []
        for m in re.finditer(r'(?m)^\s*#\s*include\s+[<"]([^">]+)[">]', src_text):
            includes.append(m.group(1).strip())
        includes = list(dict.fromkeys(includes)) if includes else None

        module_doc_end = self._module_comment_block_end(src_text)

        file_ext = Path(file_path).suffix or (".hpp" if language == "cpp" else ".h")
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
    def _build_signature_from_node(src: bytes, node) -> str:
        try:
            txt = src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        except Exception:
            return ""
        # strip trailing semicolon and whitespace
        txt = txt.split(';')[0].strip()
        # collapse whitespace
        return re.sub(r'\s+', ' ', txt).strip()

    @staticmethod
    def _build_signature_from_text(txt: str) -> str:
        if not txt:
            return ""
        txt = txt.split(';')[0].strip()
        return re.sub(r'\s+', ' ', txt).strip()

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
        if not sig:
            return ""
        matches = re.findall(r'([A-Za-z_]\w*)\s*\(', sig)
        return matches[-1] if matches else ""
