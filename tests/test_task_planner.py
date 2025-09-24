#!/usr/bin/env python3
"""Test script to verify TaskPlannerAgent timing and LLM usage."""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.langgraph.agents.task_planner import TaskPlannerAgent
from src.langgraph.state_models import ConversationState, ConversationBuffer

def test_task_planner_timing():
    """Test that TaskPlannerAgent takes reasonable time when using LLM."""

    # Create a test state with a question
    state = ConversationState()
    state.conversation = ConversationBuffer(
        user_questions={"user_question1": "What is IPNI in the Filecoin ecosystem and is it centralized?"},
        history=[
            {"user_question1": "What is IPNI in the Filecoin ecosystem and is it centralized?"}
        ]
    )

    # Test without LLM config (should fail)
    print("Testing no-LLM task planner...")
    start_time = time.perf_counter()

    planner_rules = TaskPlannerAgent(llm_config=None)
    try:
        new_state, tasks = planner_rules.plan(state)
        rule_time = time.perf_counter() - start_time
        print(f"No-LLM planning took: {rule_time:.4f}s")
        print(f"Generated {len(tasks)} tasks")
    except Exception as e:
        rule_time = time.perf_counter() - start_time
        print(f"No-LLM planning failed after {rule_time:.4f}s with error: {e}")

    # Test with LLM config (should use LLM planning)
    print("\nTesting LLM-based task planner...")
    llm_config = {
        "type": "cloud",
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "context_length": 2000000,
        "max_completion_tokens": 4000,
        "tool_choice": "auto",
        "path_to_api_key": "configs/openrouter_key.yaml",
        "path_to_prompt_config": "configs/prompt_config.yaml",
        "base_url": "https://openrouter.ai/api/v1"
    }

    start_time = time.perf_counter()

    planner_llm = TaskPlannerAgent(llm_config=llm_config)
    try:
        new_state, tasks = planner_llm.plan(state)
        llm_time = time.perf_counter() - start_time
        print(f"LLM-based planning took: {llm_time:.4f}s")
        print(f"Generated {len(tasks)} tasks")

        # Verify timing difference
        if llm_time > 0.1:  # Should take at least 100ms for LLM call
            print("✅ LLM planning took reasonable time (> 0.1s)")
        else:
            print("❌ LLM planning was too fast - may not be calling LLM")
    except Exception as e:
        llm_time = time.perf_counter() - start_time
        print(f"LLM-based planning failed after {llm_time:.4f}s with error: {e}")
        import traceback
        traceback.print_exc()

    # Show task details
    if 'tasks' in locals():
        print("\nTasks generated:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task.type}: {task.description}")
            print(f"     Owner: {task.owner}")
            print(f"     Metadata: {task.metadata}")
    else:
        print("\nNo tasks generated due to exception")

if __name__ == "__main__":
    test_task_planner_timing()
