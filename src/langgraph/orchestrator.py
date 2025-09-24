"""Orchestrator coordinating task planner, agents, and verifier."""

from __future__ import annotations

import time
from typing import Optional, List, Dict, Any

from .state_models import ConversationState, Task, EvidenceItem
from .agents.responder import FinalResponse


class Orchestrator:
    def __init__(
        self,
        *,
        task_planner,
        repo_agent,
        code_agent,
        verifier,
        responder,
        llm=None,
        max_iterations: int = 5,
    ) -> None:
        self.task_planner = task_planner
        self.repo_agent = repo_agent
        self.code_agent = code_agent
        self.verifier = verifier
        self.responder = responder
        self.llm = llm
        self.max_iterations = max_iterations

        # Ensure all agents have access to LLM if available
        if self.llm:
            if hasattr(self.task_planner, 'llm'):
                self.task_planner.llm = self.llm
            # TaskPlannerAgent uses 'simple_client' attribute for OpenAI client
            if hasattr(self.task_planner, 'simple_client'):
                self.task_planner.simple_client = self.llm

            if hasattr(self.verifier, 'llm'):
                self.verifier.llm = self.llm

            if hasattr(self.repo_agent, 'llm'):
                self.repo_agent.llm = self.llm

            if hasattr(self.code_agent, 'llm'):
                self.code_agent.llm = self.llm

            if hasattr(self.responder, 'llm'):
                self.responder.llm = self.llm

    def run(self, state: ConversationState, live_log=None) -> tuple[object, List[Dict[str, Any]]]:
        iteration = 0
        final_report = None
        start_time = time.perf_counter()
        timeline: List[Dict[str, Any]] = []

        while iteration < self.max_iterations:
            state.control_flags.iteration = iteration

            turn_start = time.perf_counter()
            new_state, new_tasks = self.task_planner.plan(state)
            state = new_state
            planning_duration = time.perf_counter() - turn_start

            iteration_entry: Dict[str, Any] = {
                "iteration": iteration,
                "planning": {
                    "new_tasks": [task.dict() for task in new_tasks],
                    "total_tasks": len(state.tasks),
                    "duration_sec": round(planning_duration, 3),
                },
                "repo_agent": None,
                "code_agent": None,
                "verifier": None,
            }

            self._log(
                live_log,
                f"Task Planner generated {len(new_tasks)} task(s) in {planning_duration:.2f}s",
            )

            repo_new_items = []
            code_new_items = []

            for task in list(state.tasks):
                if task.status in {"done", "in_progress"}:
                    continue
                if task.owner == "repo_intelligence_agent":
                    self._log(live_log, "Repo Intelligence agent collecting repo evidence")
                    start = time.perf_counter()
                    state, evidence = self.repo_agent.run(state, query=self._latest_question(state))
                    elapsed = time.perf_counter() - start
                    repo_new_items = evidence
                    iteration_entry["repo_agent"] = {
                        "items_collected": len(evidence),
                        "duration_sec": round(elapsed, 3),
                        "total_content_length": sum(len(item.full_content) for item in evidence),
                    }
                    self._log(
                        live_log,
                        f"Repo Intelligence agent returned {len(evidence)} item(s) in {elapsed:.2f}s",
                    )
                    task.status = "done"
                elif task.owner == "code_inspector_agent":
                    self._log(live_log, "Code Inspector agent fetching targeted snippets")
                    start = time.perf_counter()
                    state, evidence = self.code_agent.run(state, task=task)
                    elapsed = time.perf_counter() - start
                    code_new_items = evidence
                    iteration_entry["code_agent"] = {
                        "items_collected": len(evidence),
                        "duration_sec": round(elapsed, 3),
                        "total_content_length": sum(len(item.full_content) for item in evidence),
                    }
                    self._log(
                        live_log,
                        f"Code Inspector agent returned {len(evidence)} item(s) in {elapsed:.2f}s",
                    )
                    task.status = "done"
                else:
                    task.status = "skipped"

            total_added = len(repo_new_items) + len(code_new_items)
            if total_added:
                self._log(
                    live_log,
                    f"Evidence collection: {total_added} item(s) gathered this iteration",
                )

            self._log(live_log, "Verifier agent evaluating coverage")
            start = time.perf_counter()
            report = self.verifier.evaluate(state, question=self._latest_question(state) or "")
            self.verifier.apply_report(state, report)
            elapsed = time.perf_counter() - start
            iteration_entry["verifier"] = {
                "coverage_score": getattr(report, "coverage_score", 0),
                "missing_items": getattr(report, "missing_items", []),
                "duration_sec": round(elapsed, 3),
            }
            self._log(
                live_log,
                f"Verifier coverage={getattr(report, 'coverage_score', 0):.2f} evaluated in {elapsed:.2f}s",
            )

            # Generate final answer using responder
            self._log(live_log, "Responder agent generating final answer")
            start = time.perf_counter()
            final_response = self.responder.respond(state)
            elapsed = time.perf_counter() - start
            iteration_entry["responder"] = {
                "duration_sec": round(elapsed, 3),
                "response_length": len(final_response.message),
            }
            self._log(
                live_log,
                f"Final answer generated in {elapsed:.2f}s ({len(final_response.message)} chars)",
            )
            final_report = final_response

            iteration_entry["total_evidence_items"] = len(state.evidence_store)
            timeline.append(iteration_entry)

            if getattr(report, "missing_items", []):
                iteration += 1
                continue
            break

        state.control_flags.iteration = iteration

        total_iterations = iteration + 1
        total_time = time.perf_counter() - start_time
        timeline_summary = (
            f"LangGraph execution completed: {total_iterations} iterations, {total_time:.2f}s total time"
        )
        self._log(live_log, timeline_summary)

        return final_report, timeline

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
