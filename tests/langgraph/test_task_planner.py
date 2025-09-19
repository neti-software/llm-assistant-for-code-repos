import pytest

from src.langgraph.state_models import ConversationState, Task, ConversationBuffer
from src.langgraph.agents.task_planner import TaskPlannerAgent


@pytest.fixture
def base_state():
    buffer = ConversationBuffer(
        user_questions={"user_question1": "How do I locate the Foo class implementation?"},
        history=[{"iteration": 0, "user_question1": "How do I locate the Foo class implementation?"}],
    )
    state = ConversationState(conversation=buffer)
    return state


def test_task_planner_generates_repo_and_code_tasks(base_state):
    planner = TaskPlannerAgent()
    updated_state, new_tasks = planner.plan(base_state)

    assert len(new_tasks) == 2
    repo_task = next(t for t in new_tasks if t.type == "repo_research")
    code_task = next(t for t in new_tasks if t.type == "code_context")

    assert repo_task.owner == "repo_intelligence_agent"
    assert code_task.owner == "code_inspector_agent"

    assert repo_task.metadata["input_question"].startswith("How do I locate")
    assert code_task.metadata["input_question"].startswith("How do I locate")

    assert any(t.id == repo_task.id for t in updated_state.tasks)
    assert any(t.id == code_task.id for t in updated_state.tasks)


def test_task_planner_skips_duplicate_pending_tasks(base_state):
    planner = TaskPlannerAgent()

    existing_task = Task(
        id="task-1",
        type="repo_research",
        owner="repo_intelligence_agent",
        status="pending",
    )
    base_state.tasks.append(existing_task)

    updated_state, new_tasks = planner.plan(base_state)

    assert len(new_tasks) == 1
    assert new_tasks[0].type == "code_context"
    task_types = [task.type for task in updated_state.tasks]
    assert task_types.count("repo_research") == 1
    assert task_types.count("code_context") == 1


def test_task_planner_handles_non_specific_question():
    buffer = ConversationBuffer(
        user_questions={"user_question1": "Give me a high-level overview"},
        history=[{"iteration": 0, "user_question1": "Give me a high-level overview"}],
    )
    state = ConversationState(conversation=buffer)

    planner = TaskPlannerAgent()
    updated_state, new_tasks = planner.plan(state)

    assert len(new_tasks) == 1
    assert new_tasks[0].type == "repo_research"
    assert new_tasks[0].owner == "repo_intelligence_agent"
