"""Temporary LangGraph executor stub.

This module allows the CLI feature flag to be exercised before the full
LangGraph graph is implemented. It simply echoes the latest user question
from the shared state and signals that the stub handled the turn. The state is
returned unchanged so existing persistence continues to work.
"""

from __future__ import annotations

from typing import Any, Dict

from .state_models import ConversationState


def execute_turn(llm: Any, tool_manager: Any, state: ConversationState) -> Dict[str, Any]:
    """Return a placeholder response while leaving the state intact."""

    latest_entry = None
    if state.conversation.history:
        latest_entry = state.conversation.history[-1]

    if latest_entry and any(k.startswith("user_question") for k in latest_entry.keys()):
        # Use the latest user question for context in the stub message.
        question_value = next(
            (v for k, v in latest_entry.items() if k.startswith("user_question")),
            None,
        )
    else:
        # Fall back to last recorded question in the dedicated mapping
        question_value = None
        if state.conversation.user_questions:
            last_key = sorted(state.conversation.user_questions.keys())[-1]
            question_value = state.conversation.user_questions[last_key]

    message_parts = ["[LangGraph stub] This feature is under construction."]
    if question_value:
        message_parts.append(f"Latest question: {question_value}")

    response_text = "\n".join(message_parts)

    return {
        "response": response_text,
        "state": state,
    }
