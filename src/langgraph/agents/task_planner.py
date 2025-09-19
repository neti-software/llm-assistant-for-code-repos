"""Task Planner agent that emits structured tasks for specialist agents."""

from __future__ import annotations

from typing import List, Tuple

from ..state_models import ConversationState, Task


_KEYWORDS_TRIGGERING_CODE_TASK = {
    "class",
    "function",
    "method",
    "def",
    "impl",
    "implementation",
    "file",
    "line",
    "snippet",
    "code",
}


class TaskPlannerAgent:
    """Generate tasks for downstream agents based on the latest conversation."""

    def __init__(self) -> None:
        pass

    def plan(self, state: ConversationState) -> Tuple[ConversationState, List[Task]]:
        """Return the updated state and the list of new tasks."""

        latest_question = self._extract_latest_question(state)
        if not latest_question:
            return state, []

        existing_types = {task.type for task in state.tasks}

        tasks_to_add: List[Task] = []
        next_id = self._next_task_index(state)

        if "repo_research" not in existing_types:
            tasks_to_add.append(
                Task(
                    id=f"task-{next_id}",
                    type="repo_research",
                    description="Gather repository-level context and relevant snippets",
                    owner="repo_intelligence_agent",
                    metadata={
                        "input_question": latest_question,
                        "strategy": "broad_repo_search",
                    },
                )
            )
            next_id += 1

        if self._should_request_code_context(latest_question) and "code_context" not in existing_types:
            tasks_to_add.append(
                Task(
                    id=f"task-{next_id}",
                    type="code_context",
                    description="Collect precise file excerpts supporting the query",
                    owner="code_inspector_agent",
                    metadata={
                        "input_question": latest_question,
                        "strategy": "targeted_file_lookup",
                    },
                )
            )

        if not tasks_to_add:
            return state, []

        state.tasks.extend(tasks_to_add)
        return state, tasks_to_add

    def _extract_latest_question(self, state: ConversationState) -> str | None:
        if state.conversation.history:
            for entry in reversed(state.conversation.history):
                for key, value in entry.items():
                    if key.startswith("user_question"):
                        return value
        if state.conversation.user_questions:
            last_key = sorted(state.conversation.user_questions.keys())[-1]
            return state.conversation.user_questions[last_key]
        return None

    def _should_request_code_context(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in _KEYWORDS_TRIGGERING_CODE_TASK)

    def _next_task_index(self, state: ConversationState) -> int:
        existing_ids = [task.id for task in state.tasks if task.id.startswith("task-")]
        max_existing = 0
        for task_id in existing_ids:
            try:
                _, idx = task_id.split("-", 1)
                max_existing = max(max_existing, int(idx))
            except (ValueError, TypeError):
                continue
        return max_existing + 1 if max_existing >= 0 else 1
