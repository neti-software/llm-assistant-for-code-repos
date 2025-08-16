from pathlib import Path
from typing import Dict, Any, List, Optional
from tree_sitter_languages import get_parser
from src.utils.profiler import execution_profiler


class RepoMetadataManager:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path).resolve()
        self.extractor = TreeSitterMetadataExtractor()

    @execution_profiler
    def process_repo(self) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        for file_path in self.repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.extractor._SUPPORTED_EXTENSIONS:
                continue

            file_meta = self.extractor.extract(file_path, repo_root=self.repo_path)
            rel_path = file_path.relative_to(self.repo_path).as_posix()
            results[rel_path] = file_meta

        # Map: file → namespace
        namespaces = {
            rel: self._python_module_name(rel)
            for rel, meta in results.items()
            if meta.get("language") == "python" # TODO
        }

        # Fill namespaces
        for rel, meta in results.items():
            if meta.get("language") == "python":  # TODO
                meta["namespace"] = namespaces.get(rel, "")

        # Reverse imports: file → list of files that import it
        reverse_imports: Dict[str, List[str]] = {rel: [] for rel in results}
        for importer_rel, importer_meta in results.items():
            for imp in importer_meta.get("imports", []):
                for target_rel, target_ns in namespaces.items():
                    if target_ns and target_ns in imp:
                        reverse_imports[target_rel].append(importer_rel)

        # Assign exports as “files that import me”
        for rel in results:
            results[rel]["exports"] = reverse_imports[rel]

        return results

    # ---------------- Repo-wide helpers (Python) ----------------
    @execution_profiler
    def _python_module_name(self, rel_path_str: str) -> str:
        """
        Compute module path like `pkg.subpkg.module` using the chain of
        directories that contain __init__.py. If a directory doesn't
        contain __init__.py, it breaks the package chain.
        """
        rel = Path(rel_path_str)
        parts = list(rel.parts)
        # separate dirs + file
        dirs, file_name = parts[:-1], parts[-1]
        stem = Path(file_name).stem

        pkg_parts: List[str] = []
        current = self.repo_path
        for d in dirs:
            current = current / d
            if (current / "__init__.py").exists(): # TODO
                pkg_parts.append(d)
            else:
                # non-package dir: reset chain (pkg roots must be contiguous)
                pkg_parts = []

        if stem == "__init__": # TODO
            # For __init__.py, the module is the package itself
            return ".".join(pkg_parts)
        else:
            return ".".join(pkg_parts + [stem]) if pkg_parts else stem

    @execution_profiler
    def _parse_dunder_all(self, file_path: Path) -> List[str]:
        """
        Parse __all__ = ["a", "b", ...] from a file using tree-sitter.
        Only handles simple list/tuple of string literals (most common case).
        """
        try:
            src = file_path.read_bytes()
        except Exception:
            return []
        parser = get_parser("python")  # TODO
        tree = parser.parse(src)
        root = tree.root_node

        def decode(n):
            return src[n.start_byte:n.end_byte].decode("utf-8")

        names: List[str] = []
        cursor = root.walk()
        seen = set()
        while True:
            n = cursor.node
            if n.id not in seen:
                seen.add(n.id)
                if n.type == "assignment":
                    # left side could be an identifier __all__
                    lhs = None
                    for ch in n.children:
                        if ch.type == "identifier":
                            lhs = decode(ch)
                            break
                    if lhs == "__all__":
                        # right side should be list/tuple of strings ideally
                        rhs = None
                        for ch in n.children[::-1]:
                            if ch.type in ("list", "tuple"):
                                rhs = ch
                                break
                        if rhs:
                            for elem in rhs.children:
                                if elem.type in ("string", "concatenated_string", "f_string"):
                                    text = decode(elem).strip()
                                    # strip quotes (simple)
                                    if len(text) >= 2 and text[0] in "\"'":
                                        text = text.strip("\"'")
                                    names.append(text)
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return names

    @execution_profiler
    def _default_python_exports(self, meta: dict) -> List[str]:
        """
        Fallback when __all__ is not present:
        - public top-level classes (not starting with '_')
        - public top-level functions (enclosing_class is None, not starting with '_')
        """
        exports: List[str] = []
        for cls in meta.get("classes", []):
            name = cls.get("symbol_name") or ""
            if name and not name.startswith("_"):
                exports.append(name)
        for fn in meta.get("functions", []):
            if fn.get("enclosing_class") is None:
                name = fn.get("symbol_name") or ""
                if name and not name.startswith("_"):
                    exports.append(name)
        return exports


class TreeSitterMetadataExtractor:
    _SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "c_sharp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
    }

    def __init__(self):
        pass

    @execution_profiler
    def extract(self, file_path, repo_root=None):
        source_code = Path(file_path).read_bytes()
        language = self._detect_language(file_path)
        parser = get_parser(language)  # dynamic parser selection
        tree = parser.parse(source_code)
        root_node = tree.root_node

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = Path(file_path).relative_to(repo_root).as_posix() if repo_root else str(file_path)

        module_doc, module_doc_start, module_doc_end = self._extract_module_docstring_info(root_node, source_code)

        schema = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": Path(file_path).suffix,
            "language": language,
            "namespace": "",
            "doc_kind": "code",
            "module_docstring": module_doc,
            "module_docstring_start": module_doc_start,
            "module_docstring_end": module_doc_end,
            "exports": [],
            "imports": self._extract_imports(root_node, source_code),
            "variables": self._extract_variables(root_node, source_code),
            "constants": self._extract_constants(root_node, source_code),
            "classes": self._extract_classes(root_node, source_code),
            "functions": self._extract_functions(root_node, source_code)
        }
        return schema

    def _detect_language(self, file_path):
        return self._SUPPORTED_EXTENSIONS.get(Path(file_path).suffix.lower(), "unknown")

    # ---------------- HELPER EXTRACTORS ----------------
    @staticmethod
    def _decode(source_code, node):
        return source_code[node.start_byte:node.end_byte].decode("utf-8") if node else ""

    @staticmethod
    def _is_async(node):
        return any(child.type == "async" for child in node.children)

    @staticmethod
    def _extract_bases(node, source_code):
        bases_node = node.child_by_field_name("superclasses")
        if not bases_node:
            return []
        return [TreeSitterMetadataExtractor._decode(source_code, child)
                for child in bases_node.children if child.type != ","]

    @staticmethod
    def _extract_decorators(node, source_code):
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                decorators.append(TreeSitterMetadataExtractor._decode(source_code, child))
        return decorators

    @staticmethod
    def _extract_parameters(node, source_code):
        params_node = node.child_by_field_name("parameters")
        params = []
        if not params_node:
            return params
        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter", "default_parameter"):
                params.append(TreeSitterMetadataExtractor._decode(source_code, child))
        return params

    @staticmethod
    def _extract_return_annotation(node, source_code):
        ret_node = node.child_by_field_name("return_type")
        return TreeSitterMetadataExtractor._decode(source_code, ret_node) if ret_node else None

    @staticmethod
    def _has_type_annotations(params, return_annotation):
        return bool(return_annotation) or any(":" in p for p in params)

    @staticmethod
    @execution_profiler
    def _extract_calls(node, source_code):
        calls = []
        cursor = node.walk()
        visited = set()
        while True:
            n = cursor.node
            if n.id not in visited:
                visited.add(n.id)
                if n.type == "call":
                    calls.append(
                        TreeSitterMetadataExtractor._decode(source_code, n.child_by_field_name("function")))
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return calls

    @staticmethod
    def _extract_docstring_info(node, source_code):
        body_node = node.child_by_field_name("body")
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement" and first_stmt.children and first_stmt.children[
                0].type == "string":
                text = TreeSitterMetadataExtractor._decode(source_code, first_stmt).strip("\"'")
                return text, first_stmt.start_point[0] + 1, first_stmt.end_point[0] + 1
        return None, None, None

    @staticmethod
    @execution_profiler
    def _extract_raises(node, source_code):
        raises = []
        cursor = node.walk()
        visited = set()
        while True:
            n = cursor.node
            if n.id not in visited:
                visited.add(n.id)
                if n.type == "raise_statement":
                    expr = n.child_by_field_name("exception")
                    if expr is None and n.children:
                        expr = n.children[1]  # usually after the 'raise' keyword
                    if expr:
                        text = TreeSitterMetadataExtractor._decode(source_code, expr)
                        raises.append(text.strip())
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return raises

    @staticmethod
    @execution_profiler
    def _extract_handled_exceptions(node, source_code):
        """
        Return list of exception types handled in try/except blocks within `node`.
        Handles:
          - except ValueError:
          - except (TypeError, ValueError):
          - bare `except:`  -> recorded as "Exception"
        """
        handled = []
        cursor = node.walk()
        seen = set()
        while True:
            n = cursor.node
            if n.id not in seen:
                seen.add(n.id)

                # In tree-sitter-python, try/except is a `try_statement` with `except_clause` children
                if n.type == "except_clause":
                    # preferred: field 'type' holds the exception expression
                    exc = n.child_by_field_name("type")
                    if exc:
                        handled.append(TreeSitterMetadataExtractor._decode(source_code, exc))
                    else:
                        # bare `except:`
                        handled.append("Exception")

            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return handled

    # ---------------- HIGHER-LEVEL EXTRACTIONS ----------------
    @execution_profiler
    def _extract_classes(self, root_node, source_code):
        classes = []
        cursor = root_node.walk()
        visited = set()
        while True:
            n = cursor.node
            if n.id not in visited:
                visited.add(n.id)
                if n.type == "class_definition":
                    name = self._decode(source_code, n.child_by_field_name("name"))
                    doc, doc_start, doc_end = self._extract_docstring_info(n, source_code)
                    bases = self._extract_bases(n, source_code)
                    decorators = self._extract_decorators(n, source_code)

                    # Methods inside
                    methods = self._extract_functions(n, source_code, enclosing_class=name)

                    classes.append({
                        "symbol_name": name,
                        "symbol_kind": "class",
                        "ast_path": f"class[{name}]",
                        "start_line_doc": doc_start,
                        "end_line_doc": doc_end,
                        "start_line_code": n.start_point[0] + 1,
                        "end_line_code": n.end_point[0] + 1,
                        "bases": bases,
                        "decorators": decorators,
                        "docstring": doc,
                        "has_type_annotations": any(m["has_type_annotations"] for m in methods),
                        "methods": methods,
                        "class_variables": self._extract_class_variables(source_code, n)
                    })
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return classes

    @execution_profiler
    def _extract_class_variables(self, source_code, class_node):
        class_vars = []
        body_node = class_node.child_by_field_name("body")

        if not body_node:
            return class_vars

        for child in body_node.children:
            node_type = child.type

            # --- 1. Class-level assignments ---
            if node_type == "expression_statement" and child.child_count > 0:
                child = child.children[0]
                node_type = child.type

            if node_type in ("assignment", "typed_assignment"):
                target_node = (
                        child.child_by_field_name("left")
                        or child.child_by_field_name("name")
                        or child.child_by_field_name("target")
                )
                if target_node:
                    var_name = self._decode(source_code, target_node)
                    class_vars.append(var_name)

            # --- 2. Instance variables in methods ---
            if node_type == "function_definition":
                func_body = child.child_by_field_name("body")
                if func_body:
                    cursor = func_body.walk()
                    visited = set()
                    while True:
                        node = cursor.node
                        if node.id not in visited:
                            visited.add(node.id)
                            if node.type == "attribute":
                                obj_node = node.child_by_field_name("object")
                                if obj_node and self._decode(source_code, obj_node) == "self":
                                    attr_node = node.child_by_field_name("attribute")
                                    if attr_node:
                                        var_name = self._decode(source_code, attr_node)
                                        if var_name not in class_vars:
                                            class_vars.append(var_name)

                        # Standard tree-sitter traversal pattern
                        if cursor.goto_first_child():
                            continue
                        while not cursor.goto_next_sibling():
                            if not cursor.goto_parent():
                                # exit this walk loop
                                break
                        else:
                            continue
                        break  # break outer while True

        return class_vars

    @execution_profiler
    def _extract_functions(self, root_node, source_code, enclosing_class=None):
        funcs = []
        cursor = root_node.walk()
        visited = set()
        while True:
            n = cursor.node
            if n.id not in visited:
                visited.add(n.id)
                if n.type in ("function_definition", "method_definition"):
                    name = self._decode(source_code, n.child_by_field_name("name"))
                    params = self._extract_parameters(n, source_code)
                    ret_ann = self._extract_return_annotation(n, source_code)
                    doc, doc_start, doc_end = self._extract_docstring_info(n, source_code)
                    calls = self._extract_calls(n, source_code)
                    raises = self._extract_raises(n, source_code)
                    decorators = self._extract_decorators(n, source_code)
                    funcs.append({
                        "symbol_name": name,
                        "symbol_kind": "function",
                        "enclosing_class": enclosing_class,
                        "ast_path": (f"class[{enclosing_class}]." if enclosing_class else "") + f"function[{name}]",
                        "signature": f"{name}({', '.join(params)})",
                        "parameters_detail": params,
                        "return_annotation": ret_ann,
                        "decorators": decorators,
                        "is_async": self._is_async(n),
                        "docstring": doc,
                        "start_line_doc": doc_start,
                        "end_line_doc": doc_end,
                        "start_line_code": n.start_point[0] + 1,
                        "end_line_code": n.end_point[0] + 1,
                        "calls": calls,
                        "handles": self._extract_handled_exceptions(n, source_code),
                        "raises": raises,
                        "has_type_annotations": self._has_type_annotations(params, ret_ann)
                    })
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return funcs

    @execution_profiler
    def _extract_imports(self, root_node, source_code):
        # Simplified for Python
        imports = []
        cursor = root_node.walk()
        visited = set()
        while True:
            n = cursor.node
            if n.id not in visited:
                visited.add(n.id)
                if n.type in ("import_statement", "import_from_statement"):
                    imports.append(self._decode(source_code, n))
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return imports

    @staticmethod
    @execution_profiler
    def _extract_module_docstring_info(root_node, source_code):
        """
        Returns (docstring_text, start_line, end_line) for module-level docstring.
        If no docstring at the top of the file, returns (None, None, None).
        """
        # A module docstring is usually the very first statement in the file
        if root_node and root_node.children:
            first_stmt = root_node.children[0]
            if first_stmt.type == "expression_statement" and first_stmt.children and first_stmt.children[
                0].type == "string":
                text = TreeSitterMetadataExtractor._decode(source_code, first_stmt).strip("\"'")
                return text, first_stmt.start_point[0] + 1, first_stmt.end_point[0] + 1
        return None, None, None

    # ----------- utils for previews -----------
    # ---------- robust "is top-level" check (not inside functions/classes) ----------
    @staticmethod
    def _is_module_level(node) -> bool:
        p = node.parent
        while p is not None:
            if p.type in ("function_definition", "class_definition"):
                return False
            p = p.parent
        return True

    # ---------- collect identifiers that belong to the LHS (before RHS start) ----------
    def _collect_identifiers_before(self, node, byte_cutoff: int):
        """Yield identifier nodes under `node` whose end_byte <= byte_cutoff."""
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.type == "identifier" and cur.end_byte <= byte_cutoff:
                yield cur
            stack.extend(cur.children)

    # ---------- iterate module-level assignments, return (name, assign_node, rhs_node) ----------
    @execution_profiler
    def _iter_module_assignments(self, root_node):
        """
        Finds module-level assignments:
          - 'assignment' (x = ...)
          - 'annotated_assignment' (x: T = ... or x: T)
          - (optionally) 'augmented_assignment' if you want, but usually not needed for definitions
        We consider them module-level if they're NOT nested inside a function/class.
        """
        cursor = root_node.walk()
        seen = set()
        while True:
            n = cursor.node
            if n.id not in seen:
                seen.add(n.id)
                if n.type in ("assignment", "annotated_assignment"):  # add 'augmented_assignment' if desired
                    if self._is_module_level(n):
                        # Find RHS node (value), if any
                        rhs = (n.child_by_field_name("right")
                               or n.child_by_field_name("value")
                               or (n.children[-1] if n.children else None))
                        rhs_start = rhs.start_byte if rhs else n.end_byte
                        # Collect LHS identifiers (anything before RHS start)
                        for name_node in self._collect_identifiers_before(n, rhs_start):
                            yield name_node.text.decode("utf-8"), n, rhs
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return

    # ---------- helpers for preview ----------
    def _node_text(self, source_code: bytes, node) -> str:
        return source_code[node.start_byte:node.end_byte].decode("utf-8") if node else ""

    @staticmethod
    def _preview_text(s: str, limit: int = 60) -> str:
        s = " ".join(s.strip().split())
        return s if len(s) <= limit else s[: limit - 1] + "…"

    # ---------- the two extractors (rich dicts) ----------
    @execution_profiler
    def _extract_variables(self, root_node, source_code):
        """
        Module-level variables (non-ALL_CAPS).
        Returns list[{"name","line","value_preview"}] unique in source order.
        """
        out, seen = [], set()
        for name, assign_node, rhs in self._iter_module_assignments(root_node):
            is_const = name.upper() == name and any(c.isalpha() for c in name)
            if is_const or name in seen:
                continue
            seen.add(name)
            preview = self._preview_text(self._node_text(source_code, rhs)) if rhs else None
            out.append({
                "name": name,
                "line": assign_node.start_point[0] + 1,
                "value_preview": preview,
            })
        return out

    @execution_profiler
    def _extract_constants(self, root_node, source_code):
        """
        Module-level constants (ALL_CAPS per Python convention).
        Returns list[{"name","line","value_preview"}] unique in source order.
        """
        out, seen = [], set()
        for name, assign_node, rhs in self._iter_module_assignments(root_node):
            is_const = name.upper() == name and any(c.isalpha() for c in name)
            if not is_const or name in seen:
                continue
            seen.add(name)
            preview = self._preview_text(self._node_text(source_code, rhs)) if rhs else None
            out.append({
                "name": name,
                "line": assign_node.start_point[0] + 1,
                "value_preview": preview,
            })
        return out
