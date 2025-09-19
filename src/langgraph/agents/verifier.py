"""Verifier agent responsible for drafting final text and coverage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..state_models import ConversationState, EvidenceItem


@dataclass
class VerifierReport:
    response_text: str
    coverage_score: float
    missing_items: List[str]
    citations: List[str]


class VerifierAgent:
    def __init__(self, coverage_threshold: float = 0.7) -> None:
        self.coverage_threshold = coverage_threshold

    def evaluate(self, state: ConversationState, *, question: str) -> VerifierReport:
        evidence = state.evidence_store
        if not evidence:
            return VerifierReport(
                response_text=f"Insufficient evidence to answer: {question}",
                coverage_score=0.0,
                missing_items=[f"Collect supporting context for: {question}"],
                citations=[],
            )

        summary_lines = ["Summary of gathered evidence:"]
        citations: List[str] = []
        for item in evidence:
            line = "- "
            if item.summary:
                line += item.summary
            elif item.snippet:
                line += item.snippet.splitlines()[0][:120]
            else:
                line += f"Evidence from {item.source_path}"
            summary_lines.append(line)
            citations.extend(item.citations)

        deduped_citations = list(dict.fromkeys(citations))
        coverage_score = self._estimate_coverage(evidence)
        missing_items: List[str] = []
        if coverage_score < self.coverage_threshold:
            missing_items.append("Additional evidence required to meet coverage threshold.")

        response_text = "\n".join(summary_lines)
        return VerifierReport(
            response_text=response_text,
            coverage_score=coverage_score,
            missing_items=missing_items,
            citations=deduped_citations,
        )

    def apply_report(self, state: ConversationState, report: VerifierReport) -> None:
        state.control_flags.last_verifier_report = {
            "coverage_score": report.coverage_score,
            "missing_items": list(report.missing_items),
            "citations": list(report.citations),
        }

    def _estimate_coverage(self, evidence: List[EvidenceItem]) -> float:
        high_confidence_items = [item for item in evidence if (item.confidence or 0) >= 0.7]
        if not evidence:
            return 0.0
        return min(1.0, len(high_confidence_items) / max(1, len(evidence)))
