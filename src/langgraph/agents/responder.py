"""Responder agent that formats the final answer from gathered evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..state_models import ConversationState
from ..debug_logger import debug_log


@dataclass
class FinalResponse:
    message: str
    citations: List[str]


class ResponderAgent:
    def respond(self, state: ConversationState) -> FinalResponse:
        # Get the original question to provide context
        question = self._get_latest_question(state)
        

        # If we have evidence, synthesize a focused response
        if state.evidence_store:
            response_text = self._synthesize_response(state, question)
        else:
            response_text = f"No evidence found to answer: {question}"

        # Collect citations
        citations = []
        if state.control_flags.last_verifier_report:
            citations.extend(state.control_flags.last_verifier_report.get("citations", []))
        else:
            for item in state.evidence_store:
                citations.extend(item.citations)

        return FinalResponse(message=response_text, citations=list(dict.fromkeys(citations)))

    def _get_latest_question(self, state: ConversationState) -> str:
        """Extract the latest user question from the state."""
        if state.conversation.history:
            for entry in reversed(state.conversation.history):
                for key, value in entry.items():
                    if key.startswith("user_question"):
                        return value
        if state.conversation.user_questions:
            last_key = sorted(state.conversation.user_questions.keys())[-1]
            return state.conversation.user_questions[last_key]
        return "Unknown question"

    def _synthesize_response(self, state: ConversationState, question: str) -> str:
        """Synthesize a focused response from the collected evidence."""
        if not state.evidence_store:
            return f"Insufficient evidence to answer: {question}"

        # Start with the question context
        parts = [f"Based on the evidence collected, here is the answer to: **{question}**"]
        parts.append("")

        # Group evidence by relevance and synthesize
        high_confidence_items = []
        medium_confidence_items = []
        low_confidence_items = []

        debug_log("ResponderAgent", f"Processing {len(state.evidence_store)} evidence items")
        for i, item in enumerate(state.evidence_store):
            confidence = item.confidence or 0
            debug_log("ResponderAgent", f"Evidence {i}: source='{item.source_path}', confidence={confidence:.3f}")

            # Use full snippet content - prefer snippet over summary if both exist
            content = item.snippet or item.summary or ""
            debug_log("ResponderAgent", f"Evidence {i} content length: {len(content)} characters")

            # Skip evidence items with high confidence but no content (corrupted data)
            if confidence >= 0.8 and not content:
                debug_log("ResponderAgent", f"SKIPPING Evidence {i}: high confidence ({confidence:.3f}) but no content!")
                continue

            if confidence >= 0.8:
                high_confidence_items.append((item, content))
            elif confidence >= 0.6:
                medium_confidence_items.append((item, content))
            elif confidence >= 0.4:  # Lower threshold for low confidence since we have full data
                low_confidence_items.append((item, content))

        # Add high confidence items first
        if high_confidence_items:
            parts.append("**Key findings:**")
            for item, content in high_confidence_items[:3]:  # Limit to top 3
                if content:
                    # Show source path and full content
                    source_info = f"From {item.source_path}:" if item.source_path else ""
                    parts.append(f"- {source_info}")
                    # Format the full content with proper line breaks
                    formatted_content = content.replace('\n', '\n  ')
                    parts.append(f"  {formatted_content}")
            parts.append("")

        # Add medium confidence items if needed
        if medium_confidence_items and len(high_confidence_items) < 2:
            parts.append("**Additional context:**")
            for item, content in medium_confidence_items[:2]:  # Limit to top 2
                if content:
                    # Show source path and full content
                    source_info = f"From {item.source_path}:" if item.source_path else ""
                    parts.append(f"- {source_info}")
                    # Format the full content with proper line breaks
                    formatted_content = content.replace('\n', '\n  ')
                    parts.append(f"  {formatted_content}")
            parts.append("")

        # Fall back to low-confidence observations if nothing stronger is available
        if not (high_confidence_items or medium_confidence_items) and low_confidence_items:
            parts.append("**Preliminary observations (low confidence):**")
            for item, content in low_confidence_items[:2]:
                if content:
                    # Show source path and full content
                    source_info = f"From {item.source_path}:" if item.source_path else ""
                    parts.append(f"- {source_info}")
                    # Format the full content with proper line breaks
                    formatted_content = content.replace('\n', '\n  ')
                    parts.append(f"  {formatted_content}")
            parts.append("")

        # Add source information if we have any evidence references
        sources = set()
        for item in state.evidence_store:
            if item.source_path and item.source_path != "unknown":
                sources.add(item.source_path)
        
        debug_log("ResponderAgent", f"Collected {len(sources)} unique sources")
        for source in sorted(sources):
            debug_log("ResponderAgent", f"Source: {source}")
            
        if sources:
            parts.append("**Sources:**")
            for source in list(sources)[:3]:  # Limit to top 3 sources
                parts.append(f"- {source}")

        return "\n".join(parts)

    def persist_response(self, conversation_history, response: FinalResponse) -> None:
        conversation_history.add_model_response(response.message)
