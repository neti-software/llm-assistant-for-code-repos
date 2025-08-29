import re
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_languages import get_parser


class MetadataExtractorRust:
    def __init__(self):
        self.parser = get_parser("rust")

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src = p.read_bytes()
        src_text = src.decode("utf-8", errors="ignore")
        tree = self.parser.parse(src)
        root = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        comments = self._collect_comment_nodes(root)

        classes: List[Dict] = []
        for node in self._find_nodes(root, {"struct_item", "enum_item", "impl_item"}):
            if node.type == "impl_item":
                # get the type being implemented
                type_node = node.child_by_field_name("type")
                name = self._decode(src, type_node).strip() if type_node else None
                if not name:
                    continue
            else:
                # struct or enum
                name = self._decode(src, node.child_by_field_name("name"))
                if not name:
                    continue

            doc, sdoc, edoc = self._leading_docstring(src, comments, node)
            classes.append({
                "path": rel_path,
                "symbol_name": name,
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": node.start_point[0] + 1,
                "end_line_code": node.end_point[0] + 1,
                "methods": []
            })

        functions: List[Dict] = []
        # collect free functions (skip functions that are inside any impl_item ancestor)
        for node in self._find_nodes(root, {"function_item"}):
            if self._has_ancestor_of_type(node, "impl_item"):
                continue
            name = self._decode(src, node.child_by_field_name("name"))
            params = self._decode(src, node.child_by_field_name("parameters"))
            result = self._decode(src, node.child_by_field_name("return_type"))
            sig = self._build_fn_signature(name, params, result)
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

        # def dump_tree(root, src, depth=0):
        #     indent = "  " * depth
        #     print(
        #         f"{indent}{root.type} [{root.start_point}–{root.end_point}] → {src[root.start_byte:root.end_byte].decode('utf-8', 'ignore')[:40]}")
        #     for i in range(root.child_count):
        #         dump_tree(root.children[i], src, depth + 1)
        #
        # # usage
        # tree = self.parser.parse(src)
        # dump_tree(tree.root_node, src)


        # --- foreign (extern "C") functions ---
        for extern in self._find_nodes(root, {"foreign_mod_item"}):
            for node in self._find_nodes(extern, {"function_signature_item"}):
                name = self._decode(src, node.child_by_field_name("name"))
                params = self._decode(src, node.child_by_field_name("parameters"))
                result = self._decode(src, node.child_by_field_name("return_type"))
                sig = self._build_fn_signature(name, params, result)
                doc, sdoc, edoc = self._leading_docstring(src, comments, node)
                functions.append({
                    "path": rel_path,
                    "symbol_name": name or f"extern@line{node.start_point[0] + 1}",
                    "enclosing_class": None,
                    "signature": sig,
                    "docstring": doc,
                    "start_line_documentation": sdoc,
                    "end_line_documentation": edoc,
                    "start_line_code": node.start_point[0] + 1,
                    "end_line_code": node.end_point[0] + 1,
                    "is_extern": True
                })

        # collect methods inside impl_item and attach to classes (do NOT add them to top-level functions)
        for impl in self._find_nodes(root, {"impl_item"}):
            type_node = impl.child_by_field_name("type")
            struct_name = self._decode(src, type_node).strip() if type_node is not None else None
            if struct_name and "::" in struct_name:
                struct_name = struct_name.split("::")[-1].strip()

            impl_header = src[impl.start_byte:impl.end_byte].decode("utf-8", errors="ignore").split("{", 1)[0]
            if " for " in impl_header:
                continue

            for method_node in self._find_nodes(impl, {"function_item"}):
                name = self._decode(src, method_node.child_by_field_name("name"))
                params = self._decode(src, method_node.child_by_field_name("parameters"))
                result = self._decode(src, method_node.child_by_field_name("return_type"))
                sig = self._build_fn_signature(name, params, result)
                doc, sdoc, edoc = self._leading_docstring(src, comments, method_node)
                method_entry = {
                    "path": rel_path,
                    "symbol_name": name,
                    "enclosing_class": struct_name,
                    "signature": sig,
                    "docstring": doc,
                    "start_line_documentation": sdoc,
                    "end_line_documentation": edoc,
                    "start_line_code": method_node.start_point[0] + 1,
                    "end_line_code": method_node.end_point[0] + 1
                }
                attached = False
                for cls in classes:
                    if cls["symbol_name"] == struct_name:
                        cls["methods"].append(method_entry)
                        attached = True
                        break
                if not attached and struct_name:
                    placeholder = {
                        "path": rel_path,
                        "symbol_name": struct_name,
                        "docstring": "",
                        "start_line_documentation": None,
                        "end_line_documentation": None,
                        "start_line_code": None,
                        "end_line_code": None,
                        "methods": [method_entry]
                    }
                    classes.append(placeholder)

        # simple import extraction (lines starting with `use `)
        imports = re.findall(r"(?m)^\s*use\s+([^\n;]+)", src_text)
        imports = imports or None

        # module docstring end: end line of top contiguous comment block at file start (if any)
        module_doc_end = self._module_docstring_end(src_text)

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".rs",
            "language": "rust",
            "namespace": "",
            "doc_kind": "code",
            "module_docstring_end": module_doc_end,
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

    @staticmethod
    def _has_ancestor_of_type(node, type_name: str) -> bool:
        cur = node.parent
        while cur is not None:
            if cur.type == type_name:
                return True
            cur = cur.parent
        return False

    def _collect_comment_nodes(self, root):
        nodes = []
        cursor = root.walk()
        seen = set()
        while True:
            n = cursor.node
            if n.id not in seen:
                seen.add(n.id)
                if n.type in {"line_comment", "block_comment"}:
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
    def _module_docstring_end(src_text: str) -> Optional[int]:
        # contiguous comment block from top of file
        lines = src_text.splitlines()
        end = None
        started = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*(//[/!]?|/\*|\*)", line):
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

    @staticmethod
    def _clean_comment_text(s: str) -> str:
        s = re.sub(r"(?m)^\s*///\s?", "", s)
        s = re.sub(r"(?m)^\s*//!\s?", "", s)
        s = re.sub(r"(?m)^\s*//\s?", "", s)
        s = re.sub(r"/\*+", "", s)
        s = re.sub(r"\*+/", "", s)
        s = re.sub(r"(?m)^\s*\*\s?", "", s)
        return s.strip()

    @staticmethod
    def _build_fn_signature(name: str, params: str, result: str) -> str:
        name = name or ""
        params = params or "()"
        result = result.strip() if result else ""
        if result and not result.startswith("->"):
            result = "-> " + result.lstrip("-> ").strip()
        return f"fn {name}{params} {result}".strip()
