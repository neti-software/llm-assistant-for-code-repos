import re
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_languages import get_parser


class MetadataExtractorSolidity:
    """
    Solidity extractor that maps contract/interface -> classes (no separate 'contracts' key).
    Uses node extent (covers descendants) to compute accurate start/end lines.
    """

    def __init__(self):
        try:
            self.parser = get_parser("solidity")
        except Exception:
            self.parser = None

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src_bytes = p.read_bytes()
        src_text = src_bytes.decode("utf-8", errors="ignore")

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        # fallback to regex implementation if parser unavailable
        if not self.parser:
            return self._extract_with_regex(p, src_text, repo_root)

        tree = self.parser.parse(src_bytes)
        root = tree.root_node

        comments = self._collect_comment_nodes(root)
        module_doc_end = self._module_docstring_end(src_text)

        # imports
        imports = []
        for m in re.finditer(r'(?m)^\s*import\s+[^;]*["\']([^"\']+)["\']\s*;', src_text):
            imports.append(m.group(1))
        imports = list(dict.fromkeys(imports)) if imports else None

        # build classes from contract/interface nodes
        classes: List[Dict] = []
        for node in self._find_nodes(root, {"contract_definition", "interface_definition"}):
            name = self._node_name(src_bytes, node) or ""
            kind = "interface" if node.type == "interface_definition" else "contract"
            doc, sdoc, edoc = self._leading_docstring(src_bytes, comments, node)
            s_code, e_code = self._node_extent_lines(node)
            classes.append({
                "path": rel_path,
                "symbol_name": name,
                "kind": kind,                     # optional extra field, harmless if consumer ignores
                "docstring": doc,
                "start_line_documentation": sdoc,
                "end_line_documentation": edoc,
                "start_line_code": s_code,
                "end_line_code": e_code,
                "methods": [],
                "state_variables": [],
                "events": [],
                "modifiers": []
            })

        # map name -> class dict for quick attach
        class_map = {c["symbol_name"]: c for c in classes if c.get("symbol_name")}

        # find functions and attach to enclosing class if found
        top_level_functions: List[Dict] = []
        for node in self._find_nodes(root, {"function_definition"}):
            fname = self._node_name(src_bytes, node) or self._extract_fn_name_from_text(src_bytes, node)
            sig = self._build_signature_from_node(src_bytes, node)
            fdoc, fs, fe = self._leading_docstring(src_bytes, comments, node)
            s_code, e_code = self._node_extent_lines(node)

            anc = self._find_ancestor(node, {"contract_definition", "interface_definition"})
            enclosing = self._node_name(src_bytes, anc) if anc else None

            entry = {
                "path": rel_path,
                "symbol_name": fname,
                "enclosing_class": enclosing if enclosing else None,
                "signature": sig,
                "docstring": fdoc,
                "start_line_documentation": fs,
                "end_line_documentation": fe,
                "start_line_code": s_code,
                "end_line_code": e_code
            }

            if enclosing and enclosing in class_map:
                class_map[enclosing]["methods"].append(entry)
            else:
                top_level_functions.append(entry)

        # collect other inside-contract items
        for node in self._find_nodes(root, {"state_variable_declaration", "event_definition", "modifier_definition"}):
            anc = self._find_ancestor(node, {"contract_definition", "interface_definition"})
            enclosing = self._node_name(src_bytes, anc) if anc else None
            txt = self._decode(src_bytes, node).strip()
            doc, ds, de = self._leading_docstring(src_bytes, comments, node)
            s_code, e_code = self._node_extent_lines(node)
            entry = {
                "path": rel_path,
                "symbol_name": (self._node_name(src_bytes, node) or self._extract_fn_name_from_text(src_bytes, node) or "").strip(),
                "enclosing_class": enclosing if enclosing else None,
                "signature": txt.splitlines()[0] if txt else "",
                "docstring": doc,
                "start_line_documentation": ds,
                "end_line_documentation": de,
                "start_line_code": s_code,
                "end_line_code": e_code
            }
            if enclosing and enclosing in class_map:
                if node.type == "state_variable_declaration":
                    class_map[enclosing]["state_variables"].append(entry)
                elif node.type == "event_definition":
                    class_map[enclosing]["events"].append(entry)
                elif node.type == "modifier_definition":
                    class_map[enclosing]["modifiers"].append(entry)

        # ensure arrays remain arrays (empty lists if nothing)
        for c in classes:
            for k in ("methods", "state_variables", "events", "modifiers"):
                if c.get(k) is None:
                    c[k] = []
                else:
                    if k == "methods":
                        for m in c[k]:
                            if "enclosing_class" not in m:
                                m["enclosing_class"] = c.get("symbol_name")

        # namespace extraction (Solidity has no namespace; keep empty string)
        namespaces = self._collect_namespaces(src_bytes, root) if self.parser else None
        namespace_field = namespaces[0] if namespaces and len(namespaces) else ""

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".sol",
            "language": "solidity",
            "namespace": namespace_field,
            "doc_kind": "code",
            "module_docstring_end": module_doc_end,
            "imports": imports,
            "classes": classes if classes else None,
            "functions": top_level_functions if top_level_functions else None
        }
        return global_meta

    # fallback regex extraction (contracts -> classes)
    def _extract_with_regex(self, path: Path, src_text: str, repo_root: Optional[str]) -> Dict:
        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = path.relative_to(repo_root).as_posix() if repo_root else str(path)
        module_doc_end = self._module_docstring_end(src_text)

        imports = []
        for m in re.finditer(r'(?m)^\s*import\s+[^;]*["\']([^"\']+)["\']\s*;', src_text):
            imports.append(m.group(1))
        imports = list(dict.fromkeys(imports)) if imports else None

        classes = []
        for m in re.finditer(r'(?m)^\s*(contract|interface)\s+([A-Za-z_]\w*)', src_text):
            kind, name = m.group(1), m.group(2)
            start_line = src_text[:m.start()].count("\n") + 1
            classes.append({
                "path": rel_path,
                "symbol_name": name,
                "kind": kind,
                "docstring": None,
                "start_line_documentation": None,
                "end_line_documentation": None,
                "start_line_code": start_line,
                "end_line_code": start_line,
                "methods": [],
                "state_variables": [],
                "events": [],
                "modifiers": []
            })

        functions = []
        for m in re.finditer(r'(?m)^\s*function\s+([A-Za-z_]\w*)\s*\(([^)]*)\)', src_text):
            name = m.group(1)
            sig = f"function {name}({m.group(2)})"
            start_line = src_text[:m.start()].count("\n") + 1
            functions.append({
                "path": rel_path,
                "symbol_name": name,
                "enclosing_class": None,
                "signature": sig,
                "docstring": None,
                "start_line_documentation": None,
                "end_line_documentation": None,
                "start_line_code": start_line,
                "end_line_code": start_line
            })

        return {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".sol",
            "language": "solidity",
            "namespace": "",
            "doc_kind": "code",
            "module_docstring_end": module_doc_end,
            "imports": imports,
            "classes": classes if classes else None,
            "functions": functions if functions else None
        }

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
        close = [c for c in comments if c.end_point[0] >= t_start_line - 3 and c.end_point[0] < t_start_line]
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
        s = re.sub(r"(?m)^\s*/\*\*?\s?", "", s)
        s = re.sub(r"(?m)\*/\s*$", "", s)
        s = re.sub(r"(?m)^\s*\*\s?", "", s)
        return s.strip()

    @staticmethod
    def _module_docstring_end(src_text: str) -> Optional[int]:
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

    def _node_name(self, src: bytes, node) -> str:
        try:
            n = node.child_by_field_name("name")
            if n:
                return self._decode(src, n)
        except Exception:
            pass
        txt = self._decode(src, node)
        m = re.search(r'\b([A-Za-z_]\w*)\b', txt)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_fn_name_from_text(src: bytes, node) -> str:
        try:
            txt = src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        except Exception:
            return ""
        m = re.search(r'\b([A-Za-z_]\w*)\s*\(', txt)
        return m.group(1) if m else ""

    def _build_signature_from_node(self, src: bytes, node) -> str:
        try:
            txt = src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        except Exception:
            return ""
        sig = txt.split('{')[0].split(';')[0].strip()
        sig = re.sub(r'\s+', ' ', sig)
        return sig

    def _collect_namespaces(self, src: bytes, root) -> Optional[List[str]]:
        names = []
        for n in self._find_nodes(root, {"namespace_definition"}):
            nm = self._node_name(src, n)
            if nm:
                names.append(nm)
        return list(dict.fromkeys(names)) if names else None

    def _node_extent_lines(self, node) -> (int, int):
        """
        Return (start_line, end_line) 1-based covering the node and all descendants.
        """
        if node is None:
            return 0, 0
        start = node.start_point[0]
        end = node.end_point[0]
        stack = [node]
        while stack:
            n = stack.pop()
            try:
                ep = n.end_point[0]
                if ep > end:
                    end = ep
            except Exception:
                pass
            for ch in getattr(n, "children", []):
                stack.append(ch)
        return start + 1, end + 1
