import json
from typing import List, Sequence


class StubLLM:
    """Lightweight LLM stub that returns deterministic responses for tests."""

    def __init__(
        self,
        *,
        coverage_sequence: Sequence[float] | None = None,
        final_answer: str = "Final Answer: Stub response generated from evidence.",
    ) -> None:
        self.coverage_sequence = list(coverage_sequence or [0.85])
        self.coverage_index = 0
        self.final_answer = final_answer

    # Public API -----------------------------------------------------------------
    def generate(self, prompt: str, response_format=None):
        if "Gap-Filling Task Planning Agent" in prompt:
            tasks = self._build_gap_tasks(prompt)
            return False, json.dumps({"tasks": tasks})

        if "Task Planning Agent" in prompt:
            tasks = self._build_initial_plan(prompt)
            return False, json.dumps({"tasks": tasks})

        if "Evidence Analysis and Coverage Assessment Agent" in prompt:
            return False, json.dumps(self._coverage_payload())

        if "expert assistant helping to answer a user's question" in prompt:
            return False, self._build_final_answer(prompt)

        # Default fallback
        return False, self.final_answer

    # Internal helpers -----------------------------------------------------------
    def _build_initial_plan(self, prompt: str) -> List[dict]:
        lowered = prompt.lower()
        if "overview" in lowered:
            return [
                {
                    "type": "repo_research",
                    "description": "Collect repository-level context and summaries.",
                    "priority": 8,
                }
            ]

        return [
            {
                "type": "repo_research",
                "description": "Gather repository context for the question.",
                "priority": 8,
            },
            {
                "type": "code_context",
                "description": "Inspect implementation details relevant to the question.",
                "priority": 7,
            },
        ]

    def _build_gap_tasks(self, prompt: str) -> List[dict]:
        gaps = self._extract_gaps(prompt)
        if not gaps:
            return self._build_initial_plan(prompt)

        tasks: List[dict] = []
        for idx, gap in enumerate(gaps[:3], 1):
            tasks.append(
                {
                    "type": "repo_research" if idx % 2 else "code_context",
                    "description": f"Address gap: {gap}",
                    "priority": max(10 - idx, 5),
                }
            )
        return tasks

    def _extract_gaps(self, prompt: str) -> List[str]:
        marker = "## Identified Information Gaps:"
        if marker not in prompt:
            return []

        tail = prompt.split(marker, 1)[1]
        tail = tail.split("##", 1)[0]

        gaps = []
        for line in tail.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                gaps.append(stripped.lstrip("- ").strip())
        return gaps

    def _coverage_payload(self) -> dict:
        if self.coverage_index < len(self.coverage_sequence):
            coverage = self.coverage_sequence[self.coverage_index]
        else:
            coverage = self.coverage_sequence[-1]
        self.coverage_index += 1

        missing = [] if coverage >= 0.7 else ["Additional evidence required"]
        return {
            "coverage_score": coverage,
            "response_text": "Stub coverage analysis of collected evidence.",
            "missing_items": missing,
            "citations": ["stub:0"],
        }

    def _build_final_answer(self, prompt: str) -> str:
        snippets: List[str] = []
        if "class Foo" in prompt:
            snippets.append("class Foo implementation")
        if "Foo overview" in prompt:
            snippets.append("Foo overview documentation")

        summary = " | ".join(snippets) if snippets else "summarized evidence"
        return f"{self.final_answer} ({summary})"
