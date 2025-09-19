"""Orchestrator coordinating task planner, agents, and verifier."""

from __future__ import annotations

from typing import Optional

from .state_models import ConversationState, Task


class Orchestrator:
    def __init__(
        self,
        *,
        task_planner,
        repo_agent,
        code_agent,
        verifier,
        max_iterations: int = 5,
    ) -> None:
        self.task_planner = task_planner
        self.repo_agent = repo_agent
        self.code_agent = code_agent
        self.verifier = verifier
        self.max_iterations = max_iterations

    def run(self, state: ConversationState, live_log=None) -> Optional[object]:
        iteration = 0
        final_report = None

        while iteration < self.max_iterations:
            state.control_flags.iteration = iteration

            new_state, new_tasks = self.task_planner.plan(state)
            state = new_state

            for task in list(state.tasks):
                if task.status in {"done", "in_progress"}:
                    continue
                if task.owner == "repo_intelligence_agent":
                    self._log(live_log, "Repo Intelligence agent collecting repo evidence")
                    state, _ = self.repo_agent.run(state, query=self._latest_question(state))
                    task.status = "done"
                elif task.owner == "code_inspector_agent":
                    self._log(live_log, "Code Inspector agent fetching targeted snippets")
                    state, _ = self.code_agent.run(state, task=task)
                    task.status = "done"
                else:
                    task.status = "skipped"

            self._log(live_log, "Verifier agent evaluating coverage")
            report = self.verifier.evaluate(state, question=self._latest_question(state) or "")
            self.verifier.apply_report(state, report)
            final_report = report

            if getattr(report, "missing_items", []):
                iteration += 1
                continue
            break

        state.control_flags.iteration = iteration
        return final_report

    def _latest_question(self, state: ConversationState) -> Optional[str]:
        if state.conversation.history:
            for entry in reversed(state.conversation.history):
                for key, value in entry.items():
                    if key.startswith("user_question"):
                        return value
        if state.conversation.user_questions:
            last_key = sorted(state.conversation.user_questions.keys())[-1]
            return state.conversation.user_questions[last_key]
        return None

    def _log(self, live_log, message: str) -> None:
        if live_log:
            live_log(message)
