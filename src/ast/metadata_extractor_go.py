import re
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_languages import get_parser


class MetadataExtractorGo:
    def __init__(self):
        self.parser = get_parser("go")

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src = p.read_bytes()
        src_text = src.decode("utf-8", errors="ignore")
        tree = self.parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)

        # --- classes: structs + interfaces ---
        classes: List[Dict] = []
        for decl in self._find_nodes(root, {"type_declaration"}):
            specs = [c for c in decl.children if c.type == "type_spec"]
            for spec in specs:
                type_node = spec.child_by_field_name("type")
                if type_node and type_node.type in ("struct_type", "interface_type"):
                    name = self._decode(src, spec.child_by_field_name("name"))
                    doc, sdoc, edoc = self._leading_docstring(src, comments, spec)
                    if spec.start_point and spec.end_point:  # only if tree-sitter gave positions
                        classes.append({
                            "path": rel_path,
                            "symbol_name": name,
                            "docstring": doc,
                            "start_line_documentation": sdoc,
                            "end_line_documentation": edoc,
                            "start_line_code": spec.start_point[0] + 1,
                            "end_line_code": spec.end_point[0] + 1,
                            "methods": []
                        })

        # --- free functions ---
        functions: List[Dict] = []
        for node in self._find_nodes(root, {"function_declaration"}):
            name = self._decode(src, node.child_by_field_name("name"))
            params = self._decode(src, node.child_by_field_name("parameters"))
            result = self._decode(src, node.child_by_field_name("result"))
            signature = self._build_func_signature(name, params, result)
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            functions.append({
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": None,
                "signature": signature,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1
            })

        # --- anonymous functions (func_literal) ---
        for node in self._find_nodes(root, {"func_literal"}):
            name = f"anonymous@line{node.start_point[0] + 1}"
            params = self._decode(src, node.child_by_field_name("parameters"))
            result = self._decode(src, node.child_by_field_name("result"))
            signature = self._build_func_signature(name, params, result)
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            functions.append({
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": None,
                "signature": signature,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1,
                "is_anonymous": True
            })

        # --- constants / variables ---
        for node in self._find_nodes(root, {"const_declaration", "var_declaration"}):
            text = self._decode(src, node).strip()
            first_line = text.splitlines()[0] if text else "const"
            name = f"const@line{node.start_point[0] + 1}"
            functions.append({
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": None,
                "signature": first_line,
                "docstring": None,
                "start_line_documentation": None,
                "end_line_documentation": None,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1,
                "is_constant": True
            })

        # --- methods ---
        for node in self._find_nodes(root, {"method_declaration"}):
            name = self._decode(src, node.child_by_field_name("name"))
            recv = self._decode(src, node.child_by_field_name("receiver"))
            recv_type = self._receiver_type_name(recv)
            params = self._decode(src, node.child_by_field_name("parameters"))
            result = self._decode(src, node.child_by_field_name("result"))
            signature = self._build_method_signature(recv, name, params, result)
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            method_entry = {
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": recv_type,
                "signature": signature,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1
            }
            attached = False
            for cls in classes:
                if cls["symbol_name"] == recv_type:
                    cls["methods"].append(method_entry)
                    attached = True
                    break
            if not attached and recv_type:
                placeholder = {
                    "path": rel_path,
                    "symbol_name": recv_type,
                    "docstring": "",
                    "start_line_documentation": None,
                    "end_line_documentation": None,
                    "start_line_code": node.start_point[0] + 1,
                    "end_line_code": node.end_point[0] + 1,
                    "methods": [method_entry]
                }
                classes.append(placeholder)

        # --- imports ---
        imports: Optional[List[str]] = None
        imported = []
        for m in re.finditer(r'(?m)^\s*import\s+(?:\((?P<group>[\s\S]*?)\)|(?P<single>"[^"]+"|`[^`]+`|[\w\.\/]+))', src_text):
            if m.group("single"):
                val = m.group("single").strip()
                val = val.strip('"').strip('`')
                imported.append(val)
            elif m.group("group"):
                group = m.group("group")
                items = re.findall(r'"([^"]+)"|`([^`]+)`', group)
                for a, b in items:
                    imported.append(a or b)
        if imported:
            seen = set()
            imports = []
            for it in imported:
                if it not in seen:
                    seen.add(it)
                    imports.append(it)

        module_docstring_end = self._module_docstring_end(src_text)

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".go",
            "language": "go",
            "namespace": "",
            "doc_kind": "code",
            "module_docstring_end": module_docstring_end,
            "imports": imports,
            "classes": classes if classes else None,
            "functions": functions if functions else None
        }
        return global_meta



    # ---------------- helpers ----------------
    @staticmethod
    def _decode(src: bytes, node) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore") if node else ""

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

    def _receiver_type_name(self, receiver_src: str) -> Optional[str]:
        if not receiver_src:
            return None
        s = receiver_src.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        parts = s.split()
        if not parts:
            return None
        typ = parts[-1]
        typ = typ.lstrip("*").strip()
        if "." in typ:
            typ = typ.split(".")[-1]
        return typ or None

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

    def _leading_docstring(self, src: bytes, comments: List, target_node) -> (Optional[str], Optional[int], Optional[int]):
        """
        Return (clean_text, start_line, end_line) for the docblock that precedes target_node.
        """
        if target_node is None:
            return None, None, None
        t_start_line = target_node.start_point[0]
        close = [c for c in comments if c.end_point[0] >= t_start_line - 2 and c.end_point[0] < t_start_line]
        if not close:
            close = [c for c in comments if t_start_line - 8 <= c.end_point[0] < t_start_line]
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
        s = re.sub(r"(?m)^\s*//\s?", "", s)
        s = re.sub(r"/\*+", "", s)
        s = re.sub(r"\*+/", "", s)
        s = re.sub(r"(?m)^\s*\*\s?", "", s)
        return s.strip()

    @staticmethod
    def _build_func_signature(name: str, params: str, result: str) -> str:
        name = name or ""
        params = params or "()"
        result = result.strip() if result else ""
        return f"func {name}{params} {result}".strip()

    @staticmethod
    def _build_method_signature(recv: str, name: str, params: str, result: str) -> str:
        recv = recv or "( )"
        name = name or ""
        params = params or "()"
        result = result.strip() if result else ""
        return f"func {recv} {name}{params} {result}".strip()

    @staticmethod
    def _module_docstring_end(src_text: str) -> Optional[int]:
        # contiguous comment block at top of file
        lines = src_text.splitlines()
        end = None
        started = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*//", line) or re.match(r"^\s*/\*", line):
                started = True
                end = i + 1
            elif started and line.strip() == "":
                # allow one blank line inside header and continue
                end = i + 1
                continue
            elif started:
                break
            elif not started and line.strip() == "":
                continue
            else:
                break
        return end
