"""Task Planner agent that emits structured tasks for specialist agents."""

from __future__ import annotations

import json
import time
from typing import List, Tuple, Optional, Dict, Any

from ..state_models import ConversationState, Task
from src.llm_module.llm_builder import build_llm


class TaskPlannerAgent:
    """Generate tasks for downstream agents based on the latest conversation using LLM analysis."""

    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.llm_config = llm_config
        self.llm = llm
        if self.llm is None and llm_config:
            self.llm = build_llm(llm_config)
            print(f"[DEBUG] TaskPlannerAgent: Successfully initialized with regular LLM")

    def plan(self, state: ConversationState, identified_gaps: Optional[List[str]] = None) -> Tuple[ConversationState, List[Task]]:
        """Return the updated state and the list of new tasks.
        
        Args:
            state: Current conversation state
            identified_gaps: Optional list of information gaps identified by the Verifier
        """

        start_time = time.perf_counter()
        latest_question = self._extract_latest_question(state)

        if not latest_question:
            return state, []

        # Check if we already have tasks for this question
        existing_tasks_for_question = [
            task for task in state.tasks
            if task.metadata.get("input_question") == latest_question
        ]

        # If we have existing tasks but no gaps to address, skip planning
        if existing_tasks_for_question and not identified_gaps:
            return state, []

        if not self.llm:
            raise RuntimeError("TaskPlannerAgent requires an LLM configuration. None was provided.")

        # Determine planning mode
        if identified_gaps:
            print(f"[DEBUG] TaskPlannerAgent: Using LLM for gap-based planning ({len(identified_gaps)} gaps)")
            tasks_to_add = self._plan_for_gaps(latest_question, identified_gaps, state)
        else:
            print(f"[DEBUG] TaskPlannerAgent: Using LLM for initial planning")
            tasks_to_add = self._plan_with_llm(latest_question, state)

        if not tasks_to_add:
            return state, []

        state.tasks.extend(tasks_to_add)
        planning_time = time.perf_counter() - start_time

        # Add planning metadata to the first task for debugging
        if tasks_to_add:
            tasks_to_add[0].metadata["planning_time"] = planning_time
            tasks_to_add[0].metadata["planning_method"] = "gap_filling" if identified_gaps else "initial"
            if identified_gaps:
                tasks_to_add[0].metadata["addressing_gaps"] = identified_gaps

        return state, tasks_to_add

    def _plan_with_llm(self, question: str, state: ConversationState) -> List[Task]:
        """Use LLM to intelligently plan tasks based on the question."""
        try:
            prompt = f"""You are an expert Task Planning Agent for a comprehensive code repository analysis system. Your role is to break down complex user questions into specific, actionable research tasks that will gather all necessary information to provide a complete answer.

## Your Goals:
- Analyze the user's question to understand their exact information needs
- Create a strategic plan using multiple specialized agents
- Prioritize tasks that will provide the most valuable evidence
- Ensure comprehensive coverage of all aspects mentioned in the question
- Balance between breadth (overview) and depth (specific details) as needed

## Available Specialized Agents:
1. **repo_research** - Repository Intelligence Agent
   - Searches repository contents, documentation, and codebase
   - Finds relevant files, functions, and implementation patterns
   - Extracts contextual information from multiple sources
   - Best for: understanding overall structure, finding examples, documentation

2. **code_context** - Code Inspector Agent
   - Analyzes specific code files and functions
   - Provides detailed code implementations
   - Extracts specific algorithms or patterns
   - Best for: deep code analysis, implementation details, API usage

3. **documentation_search** - Documentation Specialist Agent
   - Focuses on README files, API docs, guides
   - Extracts tutorial information and setup instructions
   - Best for: learning paths, configuration guides, conceptual explanations

4. **configuration_analysis** - Configuration Expert Agent
   - Analyzes config files, deployment settings, environment setup
   - Understands build processes and deployment configurations
   - Best for: setup instructions, environment requirements, deployment guides

## Analysis Guidelines:
- **Technical Questions**: Usually need both repo_research AND code_context
- **Implementation Questions**: Prioritize code_context with supporting repo_research
- **Setup/Configuration Questions**: Focus on documentation_search and configuration_analysis
- **Conceptual Questions**: Start with repo_research for context, then code examples
- **Multi-part Questions**: Create separate tasks for each distinct aspect

## Task Creation Rules:
- **Be Specific**: Each task should have a clear, focused objective
- **Set Priorities**: Use 1-10 scale (10 = critical, 1 = nice-to-have)
- **Limit Quantity**: Maximum 3-4 tasks per question to avoid overwhelming the system
- **Sequence Properly**: Higher priority tasks should come first
- **Consider Dependencies**: Some tasks may need results from others

## Question Analysis:
User Question: {question}

## Task Planning Strategy:
1. Identify the main topic and any sub-topics mentioned
2. Determine what type of information is most critical
3. Consider what agents are best suited for each information need
4. Prioritize tasks based on importance and logical sequence
5. Ensure tasks cover all aspects of the question comprehensively

Respond with a JSON array of 2-4 strategic tasks that will provide comprehensive coverage of the user's question.
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
            raise RuntimeError("Task planning failed due to upstream LLM error") from e

    def _plan_for_gaps(self, question: str, gaps: List[str], state: ConversationState) -> List[Task]:
        """Generate targeted tasks to address specific information gaps identified by the Verifier."""
        try:
            # Format existing evidence for context
            existing_evidence_summary = ""
            if state.evidence_store:
                existing_evidence_summary = f"\n## Evidence Already Collected ({len(state.evidence_store)} items):\n"
                for i, item in enumerate(state.evidence_store[:10], 1):  # Show first 10
                    existing_evidence_summary += f"{i}. {item.source_path or 'Unknown source'} (confidence: {item.confidence:.2f})\n"
                if len(state.evidence_store) > 10:
                    existing_evidence_summary += f"... and {len(state.evidence_store) - 10} more items\n"

            # Format identified gaps
            gaps_text = "\n".join(f"- {gap}" for gap in gaps)

            prompt = f"""You are a Gap-Filling Task Planning Agent. The Verifier has identified specific information gaps in our current evidence. Your role is to create targeted research tasks that will fill these gaps.

## Original Question:
{question}

{existing_evidence_summary}

## Identified Information Gaps:
{gaps_text}

## Your Task:
Create specific, targeted research tasks that will gather information to address ONLY the identified gaps above. Do not recreate tasks that would collect information we already have.

## Available Agents:
1. **repo_research** - Search repository, documentation, find files/patterns
2. **code_context** - Analyze specific code implementations and details
3. **documentation_search** - Find setup guides, tutorials, conceptual docs
4. **configuration_analysis** - Examine config files, deployment settings

## Requirements:
- Create 1-3 highly targeted tasks
- Each task should address one or more specific gaps
- Be specific about what information to look for
- Prioritize tasks that will have the most impact on coverage
- Avoid duplicating information we already have

## Response Format (JSON):
{{
    "tasks": [
        {{
            "type": "repo_research | code_context | documentation_search | configuration_analysis",
            "description": "Specific, actionable task description targeting the gap",
            "priority": 1-10,
            "targets_gaps": ["gap1", "gap2"]
        }}
    ]
}}

Generate gap-filling tasks now:"""

            # Use structured JSON response
            json_schema = {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "priority": {"type": "integer"},
                                "targets_gaps": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["type", "description", "priority", "targets_gaps"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["tasks"],
                "additionalProperties": False
            }

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "gap_filling_plan",
                    "schema": json_schema,
                    "strict": True
                }
            }

            want_tool, raw_response = self.llm.generate(prompt, response_format=response_format)

            if want_tool:
                raise RuntimeError("TaskPlannerAgent received unexpected tool request during gap planning")

            # Extract response content
            if isinstance(raw_response, dict):
                if "content" in raw_response:
                    response_content = raw_response["content"]
                elif "response" in raw_response:
                    response_content = raw_response["response"]
                else:
                    message = raw_response.get("message", {})
                    if isinstance(message, dict) and "content" in message:
                        response_content = message["content"]
                    else:
                        response_content = str(raw_response)
            else:
                response_content = str(raw_response)

            # Parse JSON response
            import json
            plan_data = json.loads(response_content)
            tasks_data = plan_data.get("tasks", [])

            if not tasks_data:
                print(f"[DEBUG] TaskPlannerAgent: No gap-filling tasks generated")
                return []

            print(f"[DEBUG] TaskPlannerAgent: Generated {len(tasks_data)} gap-filling tasks")
            return self._create_tasks_from_llm_plan(tasks_data, question)

        except Exception as e:
            print(f"[DEBUG] TaskPlannerAgent: Gap-based planning failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return empty list to avoid breaking the flow
            return []

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
