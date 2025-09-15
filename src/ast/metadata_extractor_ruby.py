import re
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_languages import get_parser


class MetadataExtractorRuby:
    """
    Metadata extractor for Ruby source files (.rb).
    Collects classes, methods, module functions, constants, and require/imports.
    """

    def __init__(self):
        self.parser = get_parser("ruby")

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src = p.read_bytes()
        src_text = src.decode("utf-8", errors="ignore")
        tree = self.parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)
        module_doc_end = self._module_docstring_end(src_text)
        module_doc_text = self._module_docstring_text(src_text, module_doc_end) if module_doc_end else ""

        classes: List[Dict] = []
        for node in self._find_nodes(root, {"class"}):
            name = self._decode(src, node.child_by_field_name("name")).strip()
            # skip unnamed/anonymous class nodes
            if not name:
                continue
            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            # trim module doc prefix if it overlaps
            if doc and module_doc_text:
                if doc.startswith(module_doc_text):
                    doc = doc[len(module_doc_text) :].strip()
                    if not doc:
                        doc = None
                        sdoc = None
                        edoc = None

            methods: List[Dict] = []
            for method in self._find_nodes(node, {"method"}):
                mname = self._decode(src, method.child_by_field_name("name")).strip()
                if not mname:
                    mname = ""
                sig = self._decode(src, method).splitlines()[0].strip()
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

        # free functions: Ruby uses `def` at top-level too
        functions: List[Dict] = []
        for node in self._find_nodes(root, {"method"}):
            if self._has_ancestor_of_type(node, {"class", "module"}):
                continue
            name = self._decode(src, node.child_by_field_name("name")).strip()
            if not name:
                continue
            sig = self._decode(src, node).splitlines()[0].strip()
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

        # constants: detect assignment to capitalized identifiers
        constants: List[str] = []
        for m in re.finditer(r'(?m)^([A-Z][A-Za-z0-9_]*)\s*=\s*', src_text):
            constants.append(m.group(1))
        constants = list(dict.fromkeys(constants)) if constants else None

        # imports (require / require_relative)
        imports = []
        for m in re.finditer(r'(?m)^\s*require(?:_relative)?\s+[\'"]([^\'"]+)[\'"]', src_text):
            imports.append(m.group(1))
        imports = list(dict.fromkeys(imports)) if imports else None

        module_doc_end_final = module_doc_end

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".rb",
            "language": "ruby",
            "namespace": "",
            "doc_kind": "code",
            "module_docstring_end": module_doc_end_final,
            "imports": imports,
            "classes": classes if classes else None,
            "functions": functions if functions else None,
            "constants": constants
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

    def _leading_docstring(self, src: bytes, comments: List, target_node) -> (Optional[str], Optional[int], Optional[int]):
        if target_node is None:
            return None, None, None
        t_start_line = target_node.start_point[0]
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
        return re.sub(r"(?m)^\s*#\s?", "", s).strip()

    @staticmethod
    def _has_ancestor_of_type(node, types) -> bool:
        cur = node.parent
        while cur is not None:
            if cur.type in types:
                return True
            cur = cur.parent
        return False

    @staticmethod
    def _module_docstring_end(src_text: str) -> Optional[int]:
        lines = src_text.splitlines()
        end = None
        started = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*#", line):
                started = True
                end = i + 1
            elif started and line.strip() == "":
                end = i + 1
            elif started:
                break
            elif not started and line.strip() == "":
                continue
            else:
                break
        return end

    @staticmethod
    def _module_docstring_text(src_text: str, end_line: Optional[int]) -> str:
        if not end_line or end_line <= 0:
            return ""
        lines = src_text.splitlines()[:end_line]
        raw = "\n".join(lines)
        return MetadataExtractorRuby._clean_comment_text(raw)
