import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional


class MetadataExtractorMarkdown:
    """
    Extract metadata from .md files.

    Output mirrors your Python extractor's shape minimally:
    - "classes": list with one entry representing cleaned human text
    - "functions": list of code-block entries (language, lines, signature, text)
    """

    FENCE_RE = re.compile(r"(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[^\n\r]*)\r?\n(?P<code>.*?)(?P=fence)",
                          re.DOTALL | re.MULTILINE)
    FRONTMATTER_RE = re.compile(r"(?s)^\s*---\s*\n(.*?)\n---\s*\n")

    def __init__(self):
        pass

    def extract(self, file_path: str, repo_root: Optional[str] = None) -> Dict:
        p = Path(file_path)
        src_bytes = p.read_bytes()
        src_text = src_bytes.decode("utf-8", errors="ignore")

        repo_name = Path(repo_root).name if repo_root else ""
        rel_path = p.relative_to(repo_root).as_posix() if repo_root else str(p)

        frontmatter, fm_start, fm_end = self._extract_frontmatter(src_text)
        code_blocks = self._find_code_blocks(src_text)

        # Build functions list from code blocks
        functions: List[Dict] = []
        for i, cb in enumerate(code_blocks, start=1):
            sig = self._first_non_empty_line(cb["code"]) or ""
            sig = sig.strip()
            if len(sig) > 240:
                sig = sig[:237] + "..."
            functions.append({
                "path": rel_path,
                "symbol_name": f"code_block_{i}",
                "enclosing_class": None,
                "signature": sig,
                "docstring": None,
                "start_line_documentation": None,
                "end_line_documentation": None,
                "start_line_code": cb["start_line"],
                "end_line_code": cb["end_line"],
                "language": cb["lang"] or None,
                "code": cb["code"]
            })

        # Clean text: remove frontmatter and code blocks, then strip markdown artifacts
        text_without_fm = self._remove_range(src_text, fm_start, fm_end) if fm_start is not None else src_text
        text_no_code = self._mask_out_code_blocks(text_without_fm, code_blocks)
        cleaned_text = self._clean_markdown_text(text_no_code)

        lines = cleaned_text.splitlines()
        start_doc_line = 1 if lines else None
        end_doc_line = len(lines) if lines else None

        # Determine title: frontmatter title or first H1 or filename
        title = None
        if frontmatter and isinstance(frontmatter, dict):
            title = frontmatter.get("title")
        if not title:
            m = re.search(r"(?m)^\s*#\s+(.+)$", src_text)
            if m:
                title = m.group(1).strip()
        if not title:
            title = p.stem

        classes = [{
            "path": rel_path,
            "symbol_name": title,
            "docstring": cleaned_text,
            "start_line_documentation": start_doc_line,
            "end_line_documentation": end_doc_line,
            "start_line_code": start_doc_line,
            "end_line_code": end_doc_line,
            "methods": None
        }]

        global_meta = {
            "repo": repo_name,
            "path": rel_path,
            "file_ext": ".md",
            "language": "markdown",
            "namespace": "",
            "doc_kind": "docs",
            "frontmatter": frontmatter if frontmatter else None,
            "classes": classes,
            "functions": functions if functions else None
        }
        return global_meta

    # ---------- helpers ----------
    def _extract_frontmatter(self, src_text: str):
        m = self.FRONTMATTER_RE.search(src_text)
        if not m:
            return None, None, None
        raw = m.group(1)
        try:
            parsed = yaml.safe_load(raw) or {}
        except Exception:
            parsed = {}
        return parsed, m.start(), m.end()

    def _find_code_blocks(self, src_text: str) -> List[Dict]:
        blocks = []
        for m in self.FENCE_RE.finditer(src_text):
            lang = m.group("lang").strip() if m.group("lang") else ""
            code = m.group("code")
            start_line = src_text[:m.start()].count("\n") + 1
            end_line = src_text[:m.end()].count("\n") + 1
            blocks.append({
                "lang": lang,
                "code": code,
                "start_line": start_line,
                "end_line": end_line,
                "start_byte": m.start(),
                "end_byte": m.end()
            })
        return blocks

    @staticmethod
    def _remove_range(s: str, start: Optional[int], end: Optional[int]) -> str:
        if start is None or end is None:
            return s
        return s[:start] + s[end:]

    @staticmethod
    def _mask_out_code_blocks(src_text: str, code_blocks: List[Dict]) -> str:
        if not code_blocks:
            return src_text
        pieces = []
        last = 0
        for cb in sorted(code_blocks, key=lambda x: x["start_byte"]):
            pieces.append(src_text[last:cb["start_byte"]])
            # replace code with a single newline placeholder to keep line offsets compact
            nl_count = src_text[cb["start_byte"]:cb["end_byte"]].count("\n")
            pieces.append("\n" * min(nl_count, 2))
            last = cb["end_byte"]
        pieces.append(src_text[last:])
        return "".join(pieces)

    @staticmethod
    def _first_non_empty_line(s: str) -> Optional[str]:
        for line in s.splitlines():
            t = line.strip()
            if t:
                return t
        return None

    @staticmethod
    def _clean_markdown_text(s: str) -> str:
        # remove remaining frontmatter markers if any
        s = re.sub(r"(?s)^---.*?---\s*", "", s)
        # convert images to alt text
        s = re.sub(r"!\[([^\]]*)\]\((?:[^)]+)\)", r"\1", s)
        # convert links to link text
        s = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", s)
        # remove heading markers
        s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
        # remove emphasis and inline code ticks
        s = re.sub(r"(`+)(.*?)\1", r"\2", s)       # inline code
        s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)  # bold
        s = re.sub(r"(\*|_)(.*?)\1", r"\2", s)     # italic
        # remove blockquote markers
        s = re.sub(r"(?m)^\s*>+\s?", "", s)
        # remove markdown table pipes but keep content
        s = re.sub(r"(?m)^\s*\|(.+)\|\s*$", lambda m: "  ".join([c.strip() for c in m.group(1).split("|")]), s)
        # remove html comments
        s = re.sub(r"(?s)<!--.*?-->", "", s)
        # collapse multiple blank lines to max two
        s = re.sub(r"\n{3,}", "\n\n", s)
        # strip leading/trailing whitespace
        return s.strip()


x = MetadataExtractorMarkdown()
file_path= "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/DATA_TO_TEST/filecoin-solidity/README.md"
repo_root= "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/DATA_TO_TEST/filecoin-solidity"

x.extract(file_path=file_path, repo_root=repo_root)