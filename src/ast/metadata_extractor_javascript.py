import re
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_languages import get_parser


class MetadataExtractorJS:
    def __init__(self):
        self.parser = get_parser("javascript")

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src = p.read_bytes()
        src_text = src.decode("utf-8", errors="ignore")
        tree = self.parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)

        # classes
        classes: List[Dict] = []
        for node in self._find_nodes(root, {"class_declaration"}):
            name = self._decode(src, node.child_by_field_name("name"))
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            # collect methods in class body
            methods: List[Dict] = []
            body = node.child_by_field_name("body")
            if body:
                for method in self._find_nodes(body, {"method_definition"}):
                    mname = self._decode(src, method.child_by_field_name("name"))
                    params = self._decode(src, method.child_by_field_name("parameters"))
                    sig = f"function {mname}{params}".strip()
                    mdoc, ms, me = self._leading_docstring(src, comments, method)
                    methods.append({
                        "path": rel_path,
                        "symbol_name": mname,
                        "enclosing_class": name,
                        "signature": sig,
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

        # free functions (function_declaration)
        functions: List[Dict] = []
        for node in self._find_nodes(root, {"function_declaration"}):
            name = self._decode(src, node.child_by_field_name("name"))
            params = self._decode(src, node.child_by_field_name("parameters"))
            sig = f"function {name}{params}".strip()
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

        # arrow functions assigned to variables: variable_declarator with initializer arrow_function
        for node in self._find_nodes(root, {"variable_declarator"}):
            init = node.child_by_field_name("value") or node.child_by_field_name("initializer")
            if init and init.type == "arrow_function":
                # variable name
                id_node = node.child_by_field_name("name") or node.child_by_field_name("identifier")
                name = self._decode(src, id_node) or self._decode(src, node.child_by_field_name("name"))
                params = self._decode(src, init.child_by_field_name("parameters"))
                sig = f"const {name}{params} =>"
                # use the initializer (arrow function) as the target for doc lookup
                doc, sdoc, edoc = self._leading_docstring(src, comments, init)
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

        # imports: ESM import and require()
        imports = []
        for m in re.finditer(r'(?m)^\s*import\s+(?:.+?\s+from\s+)?[\'"]([^\'"]+)[\'"]', src_text):
            imports.append(m.group(1))
        for m in re.finditer(r"require\((?:'|\")([^'\"]+)(?:'|\")\)", src_text):
            imports.append(m.group(1))
        imports = list(dict.fromkeys(imports)) if imports else None

        module_doc_end = self._module_docstring_end(src_text)
        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".js",
            "language": "javascript",
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
        """
        Return (clean_text, start_line, end_line) for the docblock that immediately
        precedes target_node. Require the doc block to end no more than 1 line
        above target_node (stricter adjacency).
        """
        if target_node is None:
            return None, None, None
        t_start_line = target_node.start_point[0]
        # require comments that end in [t_start_line-1, t_start_line)
        close = [c for c in comments if c.end_point[0] >= t_start_line - 1 and c.end_point[0] < t_start_line]
        if not close:
            # fallback: look back up to 3 lines but still prefer immediate adjacency
            close = [c for c in comments if t_start_line - 3 <= c.end_point[0] < t_start_line]
            if not close:
                return None, None, None
        # keep only comments that are before node by byte and contiguous block
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
        # remove /**, //, leading '*' in JSDoc lines, and closing markers or stray slashes
        s = re.sub(r"(?m)^\s*/\*\*\s?", "", s)  # opening /**
        s = re.sub(r"(?m)^\s*//\s?", "", s)  # single-line //
        s = re.sub(r"(?m)^\s*\*\s?", "", s)  # leading * in JSDoc lines
        s = re.sub(r"/\*+", "", s)
        s = re.sub(r"\*+/", "", s)
        # remove any leftover lines that are just a single slash or trailing slashes
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
