"""State models shared across LangGraph agents.

These lightweight dataclasses avoid additional dependencies while providing
structured accessors for conversation buffers, task queues, and evidence
stores. The design intentionally mirrors the terminology from the migration
plan so later LangGraph integration can reuse these containers directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy


@dataclass
class ConversationBuffer:
    """Snapshot of the running conversation and chronological events."""

    user_questions: Dict[str, str] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def dict(self) -> Dict[str, Any]:
        """Return a deep-copied dictionary representation."""
        return {
            "user_questions": copy.deepcopy(self.user_questions),
            "history": copy.deepcopy(self.history),
        }


@dataclass
class Task:
    """Represent a unit of work emitted by the Task Planner."""

    id: str
    type: str
    description: Optional[str] = None
    priority: int = 0
    status: str = "pending"
    owner: Optional[str] = None
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - helper for future use
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "owner": self.owner,
            "attempts": self.attempts,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class EvidenceItem:
    """Structured evidence captured by specialist agents."""

    source_path: Optional[str] = None
    summary: Optional[str] = None
    snippet: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - helper for future use
        return {
            "source_path": self.source_path,
            "summary": self.summary,
            "snippet": self.snippet,
            "citations": list(self.citations),
            "confidence": self.confidence,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ControlFlags:
    """Orchestrator bookkeeping (iteration counters, budgets, etc.)."""

    iteration: int = 0
    max_iterations: int = 8
    token_budget: int = 0
    tool_budget: int = 0
    last_verifier_report: Optional[Dict[str, Any]] = None

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - helper for future use
        return {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "token_budget": self.token_budget,
            "tool_budget": self.tool_budget,
            "last_verifier_report": copy.deepcopy(self.last_verifier_report),
        }


@dataclass
class ConversationState:
    """Top-level container LangGraph nodes exchange."""

    conversation: ConversationBuffer = field(default_factory=ConversationBuffer)
    tasks: List[Task] = field(default_factory=list)
    evidence_store: List[EvidenceItem] = field(default_factory=list)
    control_flags: ControlFlags = field(default_factory=ControlFlags)

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - helper for future use
        return {
            "conversation": self.conversation.dict(),
            "tasks": [task.dict() for task in self.tasks],
            "evidence_store": [item.dict() for item in self.evidence_store],
            "control_flags": self.control_flags.dict(),
        }


def conversation_from_raw(raw: Dict[str, Any]) -> ConversationBuffer:
    """Construct a ConversationBuffer from the persisted history dict."""
    user_questions = copy.deepcopy(raw.get("user_questions", {}))
    history_items = copy.deepcopy(raw.get("history", []))
    return ConversationBuffer(user_questions=user_questions, history=history_items)
