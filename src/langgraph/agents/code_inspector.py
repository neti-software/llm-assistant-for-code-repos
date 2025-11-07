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
            except Exception as e:
                print(f"[ERROR] CodeInspectorAgent: Structure node invocation failed: {e}")
                raise

        # If no target paths provided, try to extract them from existing evidence
        if not target_paths:
            target_paths = self._extract_target_paths_from_evidence(state, task)

        if not target_paths:
            return state, []

        evidence_items: List[EvidenceItem] = []
        for path in target_paths:
            if self.file_node is None:
                continue
            
            result = self.file_node.invoke(
                file_path=path,
                question=task.metadata.get("input_question"),
            )
            
            if result.get("error"):
                error_msg = result.get("error")
                print(f"[ERROR] CodeInspectorAgent: File node returned error for {path}: {error_msg}")
                continue

            payload = result.get("data")
            normalized_items = self._normalize_payload(payload)
            if not normalized_items:
                continue

            evidence_items.extend(
                self._convert_results(path, self.file_node.tool_name, normalized_items, result)
            )

        if evidence_items:
            state.evidence_store.extend(evidence_items)

        return state, evidence_items

    def _normalize_payload(self, payload: Any) -> List[Any]:
        """Ensure tool payload is iterable without breaking strings into characters."""

        if payload is None:
            return []
        if isinstance(payload, (str, bytes)):
            return [payload]
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, Iterable):
            return list(payload)
        return [payload]

    def _convert_results(
        self,
        target_path: str,
        tool_name: str,
        payload_items: List[Any],
        raw_result: dict,
    ) -> List[EvidenceItem]:
        citations = raw_result.get("citations", [])
        confidence = raw_result.get("confidence")
        if confidence is None:
            confidence = 1.0
        args_metadata = raw_result.get("metadata", {}).get("args", {})

        evidence: List[EvidenceItem] = []
        for idx, item in enumerate(payload_items):
            line_start = None
            line_end = None
            score = None
            full_content: str

            if isinstance(item, str):
                full_content = item
            elif isinstance(item, dict):
                # Use full content from result, prefer 'content' or 'full_content', fallback to entire dict as JSON
                extracted = item.get("content") or item.get("full_content")
                if extracted:
                    full_content = str(extracted)
                else:
                    # Use entire dict as JSON string for full context (not snippets)
                    import json
                    full_content = json.dumps(item, indent=2)
                line_start = item.get("line_start")
                line_end = item.get("line_end")
                score = item.get("score")
            else:
                # Convert to string for full context
                full_content = str(item)

            # Skip empty content
            if not full_content.strip():
                continue

            # Create evidence item with full context content
            evidence.append(
                EvidenceItem(
                    source_path=target_path,
                    full_content=full_content,
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

    def _extract_target_paths_from_evidence(self, state: ConversationState, task: Task) -> List[str]:
        """Extract potential target file paths from existing evidence."""
        if not state.evidence_store:
            return []

        source_paths = {
            item.source_path
            for item in state.evidence_store
            if item.source_path and item.source_path != "unknown"
        }

        relevant_paths = list(source_paths)

        if task.metadata:
            task.metadata["target_paths"] = relevant_paths

        return relevant_paths[:5]