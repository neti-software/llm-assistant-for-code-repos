"""Code Inspector agent for targeted file retrieval."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

from ..state_models import ConversationState, EvidenceItem, Task


class CodeInspectorAgent:
    """Fetch precise code snippets using structure and file tool nodes."""

    def __init__(self, *, structure_node: Optional[Any], file_node: Optional[Any]) -> None:
        self.structure_node = structure_node
        self.file_node = file_node

    def run(
        self,
        state: ConversationState,
        *,
        task: Task,
    ) -> Tuple[ConversationState, List[EvidenceItem]]:
        target_paths = task.metadata.get("target_paths", []) if task.metadata else []
        project_root = task.metadata.get("project_root", ".") if task.metadata else "."

        if self.structure_node is not None:
            try:
                self.structure_node.invoke(root=project_root)
            except Exception:
                pass

        if not target_paths:
            return state, []

        evidence_items: List[EvidenceItem] = []
        for path in target_paths:
            if self.file_node is None:
                continue
            try:
                result = self.file_node.invoke(
                    file_path=path,
                    question=task.metadata.get("input_question"),
                )
            except Exception:
                continue
            if result.get("error"):
                continue
            payload = result.get("data")
            if not payload:
                continue
            evidence_items.extend(
                self._convert_results(path, self.file_node.tool_name, payload, result)
            )

        if evidence_items:
            state.evidence_store.extend(evidence_items)

        return state, evidence_items

    def _convert_results(
        self,
        target_path: str,
        tool_name: str,
        payload: Iterable[Any],
        raw_result: dict,
    ) -> List[EvidenceItem]:
        citations = raw_result.get("citations", [])
        confidence = raw_result.get("confidence")
        args_metadata = raw_result.get("metadata", {}).get("args", {})

        evidence: List[EvidenceItem] = []
        for idx, item in enumerate(payload):
            snippet = None
            line_start = None
            line_end = None
            score = None

            if isinstance(item, str):
                snippet = item
            elif isinstance(item, dict):
                snippet = item.get("snippet") or item.get("content")
                line_start = item.get("line_start")
                line_end = item.get("line_end")
                score = item.get("score")
            else:
                continue

            evidence.append(
                EvidenceItem(
                    source_path=target_path,
                    snippet=snippet,
                    citations=[citations[idx]] if idx < len(citations) else [],
                    confidence=confidence,
                    metadata={
                        "tool": tool_name,
                        "line_start": line_start,
                        "line_end": line_end,
                        "score": score,
                        "args": args_metadata,
                    },
                )
            )
        return evidence
