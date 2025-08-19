from pathlib import Path
from typing import Dict, List
from tree_sitter_languages import get_parser
from src.utils.profiler import execution_profiler
from src.ast.metadata_extractor import MetadataExtractor


class MetadataExtractorManager:
    def __init__(self):
        self.extractor = MetadataExtractor()

    @execution_profiler
    def process_repo(self, repo_path) -> Dict[str, dict]:
        repo_path = Path(repo_path).resolve()
        results: Dict[str, dict] = {}
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.extractor._SUPPORTED_EXTENSIONS:
                continue

            file_meta = self.extractor.extract(file_path, repo_root=repo_path)
            rel_path = file_path.relative_to(repo_path).as_posix()
            results[rel_path] = file_meta

        # Map: file → namespace
        namespaces = {
            rel: self._python_module_name(repo_path, rel)
            for rel, meta in results.items()
            if meta.get("language") == "python"  # TODO
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
    def _python_module_name(self, repo_path:Path , rel_path_str: str) -> str:
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
        current = repo_path
        for d in dirs:
            current = current / d
            if (current / "__init__.py").exists():  # TODO
                pkg_parts.append(d)
            else:
                # non-package dir: reset chain (pkg roots must be contiguous)
                pkg_parts = []

        if stem == "__init__":  # TODO
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
