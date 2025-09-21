"""Task Planner agent that emits structured tasks for specialist agents."""

from __future__ import annotations

import json
import time
from typing import List, Tuple, Optional, Dict, Any

from ..state_models import ConversationState, Task
from ...llm_module.llm_builder import build_llm


class TaskPlannerAgent:
    """Generate tasks for downstream agents based on the latest conversation using LLM analysis."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None) -> None:
        self.llm_config = llm_config
        self.llm = None
        self.simple_client = None
        if llm_config:
            try:
                # Try to build LLM normally first
                self.llm = build_llm(llm_config)
                print(f"[DEBUG] TaskPlannerAgent: Successfully initialized with regular LLM")
            except Exception as e:
                print(f"[DEBUG] TaskPlannerAgent: Regular LLM failed: {e}")
                # If normal LLM fails, try to create a simple client for task planning
                try:
                    api_key = self._load_api_key(llm_config.get("path_to_api_key") or "")
                    base_url = llm_config.get("base_url", "https://api.openai.com/v1")
                    model = llm_config.get("model", "openai/gpt-4o-mini")

                    print(f"[DEBUG] TaskPlannerAgent: Trying simple client with API key length: {len(api_key)}")
                    from openai import OpenAI
                    self.simple_client = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    self.model = model
                    print(f"[DEBUG] TaskPlannerAgent: Successfully initialized with simple client")
                except Exception as e2:
                    print(f"[DEBUG] TaskPlannerAgent: Simple client also failed: {e2}")
                    # Fallback to rule-based if everything fails
                    self.llm = None
                    self.simple_client = None

    def plan(self, state: ConversationState) -> Tuple[ConversationState, List[Task]]:
        """Return the updated state and the list of new tasks."""

        start_time = time.perf_counter()
        latest_question = self._extract_latest_question(state)

        if not latest_question:
            return state, []

        # Check if we already have tasks for this question
        existing_tasks_for_question = [
            task for task in state.tasks
            if task.metadata.get("input_question") == latest_question
        ]

        if existing_tasks_for_question:
            return state, []

        # Use LLM to intelligently plan tasks
        print(f"[DEBUG] TaskPlannerAgent: llm={self.llm is not None}, simple_client={self.simple_client is not None}")
        if self.llm or self.simple_client:
            print(f"[DEBUG] TaskPlannerAgent: Using LLM for planning")
            tasks_to_add = self._plan_with_llm(latest_question, state)
        else:
            print(f"[DEBUG] TaskPlannerAgent: No LLM available, using rule-based planning fallback")
            # Fallback to rule-based planning when no LLM is available (for testing)
            tasks_to_add = self._plan_with_rules(latest_question, state)

        if not tasks_to_add:
            return state, []

        state.tasks.extend(tasks_to_add)
        planning_time = time.perf_counter() - start_time

        # Add planning metadata to the first task for debugging
        if tasks_to_add:
            tasks_to_add[0].metadata["planning_time"] = planning_time
            tasks_to_add[0].metadata["planning_method"] = "llm" if self.llm else ("simple_client" if self.simple_client else "rules")

        return state, tasks_to_add

    def _load_api_key(self, key_path: str) -> str:
        """Load API key from YAML file."""
        try:
            import yaml
            with open(key_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("key", "")
        except Exception:
            return ""

    def _plan_with_llm(self, question: str, state: ConversationState) -> List[Task]:
        """Use LLM to intelligently plan tasks based on the question."""
        try:
            prompt = f"""Analyze this user question and determine what types of research tasks are needed to provide a comprehensive answer.

User Question: {question}

Available Task Types:
1. repo_research - Gather repository-level context, documentation, and relevant code snippets
2. code_context - Collect specific code implementations, functions, or classes
3. documentation_search - Find README files, API docs, or help documentation
4. configuration_analysis - Analyze configuration files, settings, or deployment files

Consider:
- What specific information does the user need?
- What types of evidence would be most helpful?
- Are they asking about implementation details, configuration, or general concepts?
- Do they need code examples or just conceptual explanations?

Respond with a JSON array of tasks. Each task should have:
- type: one of the available task types
- description: clear description of what to research
- priority: integer (1-10, higher = more important)

Example response:
[
  {{
    "type": "repo_research",
    "description": "Search for existing implementations and usage patterns",
    "priority": 8
  }}
]
"""

            # Call LLM to get task plan
            if self.llm:
                want_tool, response = self.llm.generate(prompt)
                if want_tool:
                    # This shouldn't happen for task planning, but handle gracefully
                    return []
                
                # Extract content from response dictionary
                if isinstance(response, dict) and "content" in response:
                    response_text = response["content"] or ""
                else:
                    response_text = str(response) if response is not None else ""
                
                print(f"[DEBUG] LLM response length: {len(response_text)}")
                print(f"[DEBUG] LLM response preview: {response_text[:300]}...")
            elif self.simple_client:
                try:
                    response_obj = self.simple_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a task planning assistant. Analyze user questions and create research tasks."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    response_text = response_obj.choices[0].message.content or ""
                    print(f"[DEBUG] Simple client response length: {len(response_text)}")
                    print(f"[DEBUG] Simple client response preview: {response_text[:300]}...")
                except Exception as e:
                    print(f"[DEBUG] Simple client failed: {e}")
                    return []
            else:
                return []

            # Parse LLM response
            try:
                # Try to extract JSON from response
                response_clean = response_text.strip()
                print(f"[DEBUG] Response clean length: {len(response_clean)}")

                if "```json" in response_clean:
                    response_clean = response_clean.split("```json")[1].split("```")[0]
                    print("[DEBUG] Extracted JSON from ```json markers")
                elif "```" in response_clean:
                    response_clean = response_clean.split("```")[1].split("```")[0]
                    print("[DEBUG] Extracted JSON from ``` markers")

                print(f"[DEBUG] Final clean response: {response_clean[:200]}...")
                tasks_data = json.loads(response_clean)
                if not isinstance(tasks_data, list):
                    tasks_data = []

                print(f"[DEBUG] Successfully parsed {len(tasks_data)} tasks")

            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON parsing failed: {e}")
                print(f"[DEBUG] Failed response: {response_clean[:200]}...")
                # Fallback: parse simple text response
                tasks_data = self._parse_text_response(response_text)
                print(f"[DEBUG] Fallback text parsing returned: {len(tasks_data)} tasks")

            return self._create_tasks_from_llm_plan(tasks_data, question)

        except Exception as e:
            print(f"[DEBUG] TaskPlannerAgent: LLM planning failed with exception: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to rule-based planning so execution can continue without LLM
            return self._plan_with_rules(question, state)

    def _parse_text_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse a simple text response into task structures."""
        tasks_data = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        for line in lines:
            if 'research' in line.lower() or 'search' in line.lower():
                tasks_data.append({
                    "type": "repo_research",
                    "description": line,
                    "priority": 8
                })
            elif 'code' in line.lower() or 'implementation' in line.lower():
                tasks_data.append({
                    "type": "code_context",
                    "description": line,
                    "priority": 7
                })
            elif 'doc' in line.lower() or 'documentation' in line.lower():
                tasks_data.append({
                    "type": "documentation_search",
                    "description": line,
                    "priority": 6
                })

        if not tasks_data:
            # Default fallback
            tasks_data = [{
                "type": "repo_research",
                "description": "Gather relevant information and context",
                "priority": 8
            }]

        return tasks_data

    def _create_tasks_from_llm_plan(self, tasks_data: List[Dict[str, Any]], question: str) -> List[Task]:
        """Convert LLM task plan into Task objects."""
        print(f"[DEBUG] _create_tasks_from_llm_plan called with {len(tasks_data)} tasks")
        tasks = []
        next_id = 1  # Start with task ID 1

        task_type_to_owner = {
            "repo_research": "repo_intelligence_agent",
            "code_context": "code_inspector_agent",
            "documentation_search": "repo_intelligence_agent",
            "configuration_analysis": "code_inspector_agent",
        }

        for i, task_data in enumerate(tasks_data[:3]):  # Limit to 3 tasks max
            print(f"[DEBUG] Creating task {i+1}: {task_data}")
            task_type = task_data.get("type", "repo_research")
            description = task_data.get("description", f"Research {task_type}")

            tasks.append(Task(
                id=f"task-{next_id + i}",
                type=task_type,
                description=description,
                owner=task_type_to_owner.get(task_type, "repo_intelligence_agent"),
                metadata={
                    "input_question": question,
                    "llm_planned": True,
                    "priority": task_data.get("priority", 5),
                },
            ))

        print(f"[DEBUG] Created {len(tasks)} tasks from LLM plan")
        return tasks

    def _plan_with_rules(self, question: str, state: ConversationState) -> List[Task]:
        """Fallback rule-based task planning (original implementation)."""
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
                        "input_question": question,
                        "strategy": "broad_repo_search",
                    },
                )
            )
            next_id += 1

        # Use the updated keyword detection
        if self._should_request_code_context(question) and "code_context" not in existing_types:
            tasks_to_add.append(
                Task(
                    id=f"task-{next_id}",
                    type="code_context",
                    description="Collect precise file excerpts supporting the query",
                    owner="code_inspector_agent",
                    metadata={
                        "input_question": question,
                        "strategy": "targeted_file_lookup",
                        "target_paths": [],
                    },
                )
            )

        return tasks_to_add

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
        """Check if question contains keywords that suggest need for code analysis."""
        lowered = question.lower()
        keywords = {
            "class", "function", "method", "def", "impl", "implementation",
            "file", "line", "snippet", "code", "how to", "example", "usage"
        }
        return any(keyword in lowered for keyword in keywords)

    def _next_task_index(self, state: ConversationState) -> int:
        """Generate next task index based on existing tasks."""
        existing_ids = [task.id for task in state.tasks if task.id.startswith("task-")]
        max_existing = 0
        for task_id in existing_ids:
            try:
                _, idx = task_id.split("-", 1)
                max_existing = max(max_existing, int(idx))
            except (ValueError, TypeError):
                continue
        return max_existing + 1 if max_existing >= 0 else 1
