"""Responder agent that formats the final answer from gathered evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..state_models import ConversationState


@dataclass
class FinalResponse:
    message: str
    citations: List[str]


class ResponderAgent:
    def respond(self, state: ConversationState) -> FinalResponse:
        parts = ["Final Answer:"]
        for item in state.evidence_store:
            if item.summary:
                parts.append(f"- {item.summary}")
            elif item.snippet:
                parts.append(f"- {item.snippet}")
            else:
                parts.append(f"- Evidence from {item.source_path}")

        citations = []
        if state.control_flags.last_verifier_report:
            citations.extend(state.control_flags.last_verifier_report.get("citations", []))
        else:
            for item in state.evidence_store:
                citations.extend(item.citations)

        message = "\n".join(parts)
        return FinalResponse(message=message, citations=list(dict.fromkeys(citations)))

    def persist_response(self, conversation_history, response: FinalResponse) -> None:
        conversation_history.add_model_response(response.message)
