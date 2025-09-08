from pathlib import Path
from typing import Dict
import fnmatch
from src.utils.profiler import execution_profiler
from src.ast.metadata_extractor_python import MetadataExtractorPython
from src.ast.metadata_extractor_go import MetadataExtractorGo
from src.ast.metadata_extractor_javascript import MetadataExtractorJS
from src.ast.metadata_extractor_rust import MetadataExtractorRust
from src.ast.metadata_validator import MetadataValidator
from src.ast.metadata_extractor_python import MetadataExtractorPython
from src.ast.metadata_extractor_ipynb import MetadataExtractorIpynb
from src.ast.metadata_extractor_go import MetadataExtractorGo
from src.ast.metadata_extractor_rust import MetadataExtractorRust
from src.ast.metadata_extractor_javascript import MetadataExtractorJS
from src.ast.metadata_extractor_typescript import MetadataExtractorTS
from src.ast.metadata_extractor_cpp import MetadataExtractorCAndCpp
from src.ast.metadata_extractor_headers import MetadataExtractorHeaders
from src.ast.metadata_extractor_ruby import MetadataExtractorRuby
from src.ast.metadata_extractor_solidity import MetadataExtractorSolidity


class MetadataExtractorManager:
    def __init__(self, metadata_schema, ignore_patterns_config):
        self.metadata_validator = MetadataValidator(metadata_schema)

        # simple extension -> extractor map
        self._extractor_by_ext = {
            ".py": MetadataExtractorPython(),
            ".ipynb": MetadataExtractorIpynb(),
            ".go": MetadataExtractorGo(),
            ".rs": MetadataExtractorRust(),
            ".js": MetadataExtractorJS(),
            ".mjs": MetadataExtractorJS(),
            ".cjs": MetadataExtractorJS(),
            ".ts": MetadataExtractorTS(),
            ".cpp": MetadataExtractorCAndCpp(),
            ".c": MetadataExtractorCAndCpp(),
            ".h": MetadataExtractorHeaders(),
            ".hpp": MetadataExtractorHeaders(),
            ".tsx": MetadataExtractorTS(),
            ".jsx": MetadataExtractorJS(),
            ".rb": MetadataExtractorRuby(),
            ".sol": MetadataExtractorSolidity(),
        }

        self._ignore_patterns = [p for v in ignore_patterns_config.values() for p in v]

    @execution_profiler
    def process_repo(self, repo_path) -> Dict[str, dict]:
        repo_path = Path(repo_path).resolve()
        results: Dict[str, dict] = {}

        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(repo_path).as_posix()

            # --- skip ignored files ---
            if any(fnmatch.fnmatch(rel_path, pat) for pat in self._ignore_patterns):
                # print(f"SKIP: {rel_path} (matched ignore pattern)")
                continue

            suffix = file_path.suffix.lower()
            extractor = self._extractor_by_ext.get(suffix)
            if extractor is None:
                # skip unknown extensions
                continue

            try:
                meta = extractor.extract(str(file_path), repo_root=str(repo_path))
                # normalize expected fields minimally
                if isinstance(meta, dict):
                    meta.setdefault("repo", repo_path.name)
                    meta.setdefault("path", rel_path)
                    meta.setdefault("file_ext", suffix)
                else:
                    meta = {
                        "repo": repo_path.name,
                        "path": rel_path,
                        "file_ext": suffix,
                        "language": "",
                        "classes": None,
                        "functions": None,
                    }
            except Exception as e:
                meta = {
                    "repo": repo_path.name,
                    "path": rel_path,
                    "file_ext": suffix,
                    "language": "",
                    "error": repr(e),
                }

            is_any_errors = self.metadata_validator.validate(meta)
            if is_any_errors:
                print(f"[VALIDATION ERROR] {rel_path}: {is_any_errors}")
                continue

            results[rel_path] = meta

        return results


