"""Repo Intelligence agent aggregates retrieval results to EvidenceItems."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence, Tuple, Optional

from ..state_models import ConversationState, EvidenceItem
from ..debug_logger import debug_log


class RepoIntelligenceAgent:
    """Collect repository-level evidence using registered tool nodes."""

    def __init__(self, tool_nodes: Sequence[Any], llm=None):
        self.tool_nodes = list(tool_nodes)
        self.llm = llm

    def run(
        self,
        state: ConversationState,
        *,
        query: str,
    ) -> Tuple[ConversationState, List[EvidenceItem]]:
        evidence: List[EvidenceItem] = []
        
        debug_log("RepoIntelligence", f"Starting search with query: '{query}'")

        for node in self.tool_nodes:
            try:
                debug_log("RepoIntelligence", f"Invoking tool: {node.tool_name}")
                result = node.invoke(query=query)
                debug_log("RepoIntelligence", f"Tool {node.tool_name} returned result", result.get("data", [])[:2] if result.get("data") else "None")
            except Exception as e:
                debug_log("RepoIntelligence", f"Tool {node.tool_name} failed: {str(e)}")
                continue

            if result.get("error"):
                debug_log("RepoIntelligence", f"Tool {node.tool_name} returned error: {result.get('error')}")
                continue
            data = result.get("data")
            if not data:
                debug_log("RepoIntelligence", f"Tool {node.tool_name} returned no data")
                continue
            
            new_evidence = self._convert_results(node.tool_name, data, result, query)
            debug_log("RepoIntelligence", f"Tool {node.tool_name} generated {len(new_evidence)} evidence items")
            evidence.extend(new_evidence)

        debug_log("RepoIntelligence", f"Total evidence collected: {len(evidence)} items")
        if evidence:
            state.evidence_store.extend(evidence)

        return state, evidence

    def _convert_results(
        self,
        tool_name: str,
        payload: Iterable[Any],
        raw_result: dict,
        query: str = "",
    ) -> List[EvidenceItem]:
        citations = raw_result.get("citations", [])
        base_confidence = raw_result.get("confidence")
        metadata_args = raw_result.get("metadata", {}).get("args", {})

        evidence_items: List[EvidenceItem] = []
        for idx, item in enumerate(payload):
            item_confidence = base_confidence
            if item_confidence is None:
                item_confidence = self._derive_confidence(item)

            if isinstance(item, str):
                if item_confidence is None:
                    item_confidence = 1.0
                evidence_items.append(
                    EvidenceItem(
                        full_content=item,
                        source_path=item,
                        citations=[citations[idx]] if idx < len(citations) else [],
                        confidence=item_confidence,
                        metadata={"tool": tool_name, "args": metadata_args},
                    )
                )
                continue

            if not isinstance(item, dict):
                continue

            # Handle the actual format from vector database search results
            # The search results have keys like "project", "path_to_file", "value", "score"
            source_path = item.get("path_to_file") or item.get("path", "unknown")
            snippet = item.get("value") or item.get("snippet", "")
            score = item.get("score")
            confidence = item_confidence if item_confidence is not None else self._score_to_confidence(score)

            # Store full content - no automatic processing
            full_content = snippet  # Complete content from the source

            debug_log("RepoIntelligence", f"Creating evidence: source_path='{source_path}', score={score}, confidence={confidence}")
            debug_log("RepoIntelligence", f"Full content preserved (no automatic summarization)")

            evidence = EvidenceItem(
                source_path=source_path,
                full_content=full_content,  # Complete agent answer
                citations=[citations[idx]] if idx < len(citations) else [],
                confidence=confidence if confidence is not None else 1.0,
                metadata={
                    "score": score,
                    "tool": tool_name,
                    "args": metadata_args,
                    "project": item.get("project"),
                    "start_line": item.get("start_line"),
                    "end_line": item.get("end_line"),
                },
            )
            evidence_items.append(evidence)

        return evidence_items

    def _derive_confidence(self, item: Any) -> Optional[float]:
        """Estimate a confidence score for a raw tool payload entry."""

        if isinstance(item, dict):
            return self._score_to_confidence(item.get("score"))
        if isinstance(item, str):
            return 0.5
        return None

    @staticmethod
    def _score_to_confidence(score: Any) -> Optional[float]:
        if score is None:
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return max(0.0, min(1.0, value))

    def _extract_key_information(self, snippet: str, source_path: str, query: str = "") -> str:
        """Return raw snippet content without any LLM processing - let Responder handle synthesis."""
        debug_log("RepoIntelligence", f"_extract_key_information called for: {source_path}")
        debug_log("RepoIntelligence", f"Input snippet length: {len(snippet) if snippet else 0}")

        if not snippet:
            debug_log("RepoIntelligence", "No snippet provided, returning empty")
            return ""

        # Return the raw snippet content directly - no LLM processing
        debug_log("RepoIntelligence", f"Returning raw snippet content ({len(snippet)} characters)")
        return snippet

