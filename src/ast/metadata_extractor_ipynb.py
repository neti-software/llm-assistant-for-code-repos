import os
import tempfile
from typing import Optional, Dict
import nbformat

from src.ast.metadata_extractor_python import MetadataExtractorPython


class MetadataExtractorIpynb:
    """
    Convert .ipynb -> .py in-memory (temp file) and use MetadataExtractorPython to extract metadata.
    Magic/ shell lines starting with %, ! are commented out to avoid syntax errors.
    """

    def __init__(self):
        self.python_extractor = MetadataExtractorPython()


    @staticmethod
    def _notebook_to_py_source(nb) -> str:
        parts = []
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                # convert markdown to commented block
                for line in cell.source.splitlines():
                    parts.append("# " + line)
                parts.append("")  # blank line
            elif cell.cell_type == "code":
                # keep code, but comment out magics/shell lines
                for line in cell.source.splitlines():
                    stripped = line.lstrip()
                    if not stripped:
                        parts.append("")
                    elif stripped.startswith(("%", "!", "?")):
                        parts.append("# " + line)
                    else:
                        parts.append(line)
                parts.append("")  # blank line between cells
        return "\n".join(parts)

    def extract_from_nbobject(self, nb, repo_root: Optional[str] = None) -> Dict:
        """
        Accept a nbformat.NotebookNode (already loaded) and extract metadata.
        """
        py_src = self._notebook_to_py_source(nb)
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
        try:
            tf.write(py_src)
            tf.flush()
            tf.close()
            meta = self.python_extractor.extract(tf.name, repo_root)
        finally:
            try:
                os.unlink(tf.name)
            except Exception:
                pass
        return meta

    def extract(self, ipynb_path: Optional[str] = None, nb: Optional[object] = None,
                repo_root: Optional[str] = None) -> Dict:
        if nb is None:
            if not ipynb_path:
                raise ValueError("provide ipynb_path or nb")
            nb = nbformat.read(str(ipynb_path), as_version=4)
        return self.extract_from_nbobject(nb, repo_root)
