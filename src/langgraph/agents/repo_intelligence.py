"""Repo Intelligence agent aggregates retrieval results to EvidenceItems."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

from ..state_models import ConversationState, EvidenceItem


class RepoIntelligenceAgent:
    """Collect repository-level evidence using registered tool nodes."""

    def __init__(self, tool_nodes: Sequence[Any]):
        self.tool_nodes = list(tool_nodes)

    def run(
        self,
        state: ConversationState,
        *,
        query: str,
    ) -> Tuple[ConversationState, List[EvidenceItem]]:
        evidence: List[EvidenceItem] = []

        for node in self.tool_nodes:
            try:
                result = node.invoke(query=query)
            except Exception:
                continue

            if result.get("error"):
                continue
            data = result.get("data")
            if not data:
                continue
            evidence.extend(self._convert_results(node.tool_name, data, result))

        if evidence:
            state.evidence_store.extend(evidence)

        return state, evidence

    def _convert_results(
        self,
        tool_name: str,
        payload: Iterable[Any],
        raw_result: dict,
    ) -> List[EvidenceItem]:
        citations = raw_result.get("citations", [])
        confidence = raw_result.get("confidence")
        metadata_args = raw_result.get("metadata", {}).get("args", {})

        evidence_items: List[EvidenceItem] = []
        for idx, item in enumerate(payload):
            if isinstance(item, str):
                evidence_items.append(
                    EvidenceItem(
                        source_path=item,
                        citations=[citations[idx]] if idx < len(citations) else [],
                        confidence=confidence,
                        metadata={"tool": tool_name, "args": metadata_args},
                    )
                )
                continue

            if not isinstance(item, dict):
                continue

            evidence = EvidenceItem(
                source_path=item.get("path"),
                summary=item.get("summary"),
                snippet=item.get("snippet"),
                citations=[citations[idx]] if idx < len(citations) else [],
                confidence=confidence,
                metadata={
                    "score": item.get("score"),
                    "tool": tool_name,
                    "args": metadata_args,
                },
            )
            evidence_items.append(evidence)

        return evidence_items
