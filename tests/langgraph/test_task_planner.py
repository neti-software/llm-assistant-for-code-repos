import pytest

from src.langgraph.state_models import ConversationState, Task, ConversationBuffer
from src.langgraph.agents.task_planner import TaskPlannerAgent
from tests.stubs.simple_llm import StubLLM


@pytest.fixture
def planner():
    return TaskPlannerAgent(llm=StubLLM())


@pytest.fixture
def base_state():
    buffer = ConversationBuffer(
        user_questions={"user_question1": "How do I locate the Foo class implementation?"},
        history=[{"iteration": 0, "user_question1": "How do I locate the Foo class implementation?"}],
    )
    state = ConversationState(conversation=buffer)
    return state


def test_task_planner_generates_repo_and_code_tasks(base_state, planner):
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


def test_task_planner_skips_duplicate_pending_tasks(base_state, planner):
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

    planner = TaskPlannerAgent(llm=StubLLM())
    updated_state, new_tasks = planner.plan(state)

    assert len(new_tasks) == 1
    assert new_tasks[0].type == "repo_research"
    assert new_tasks[0].owner == "repo_intelligence_agent"


def test_task_planner_creates_new_tasks_when_previous_tasks_done():
    """Test Fix #2: Task planner should create new tasks when all previous tasks are done."""
    question = "How do I locate the Foo class implementation?"
    buffer = ConversationBuffer(
        user_questions={"user_question1": question},
        history=[{"iteration": 0, "user_question1": question}],
    )
    state = ConversationState(conversation=buffer)
    
    # Add a task that's already done
    existing_task = Task(
        id="task-1",
        type="repo_research",
        owner="repo_intelligence_agent",
        status="done",  # This task is completed
        metadata={"input_question": question}
    )
    state.tasks.append(existing_task)
    
    # Set verifier report with missing items
    state.control_flags.last_verifier_report = {
        "coverage_score": 0.5,
        "missing_items": ["Need more implementation details"],
    }

    planner = TaskPlannerAgent(llm=StubLLM())
    updated_state, new_tasks = planner.plan(state)

    # Should create new tasks because existing task is done
    assert len(new_tasks) >= 1
    # New tasks should be for the same question
    for task in new_tasks:
        assert task.metadata.get("input_question") == question
        assert task.status == "pending"


def test_task_planner_creates_new_tasks_when_previous_tasks_skipped():
    """Test Fix #2: Task planner should create new tasks when all previous tasks are skipped."""
    question = "How do I locate the Foo class implementation?"
    buffer = ConversationBuffer(
        user_questions={"user_question1": question},
        history=[{"iteration": 0, "user_question1": question}],
    )
    state = ConversationState(conversation=buffer)
    
    # Add tasks that are skipped
    state.tasks.append(Task(
        id="task-1",
        type="repo_research",
        owner="repo_intelligence_agent",
        status="skipped",
        metadata={"input_question": question}
    ))
    state.tasks.append(Task(
        id="task-2",
        type="code_context",
        owner="code_inspector_agent",
        status="skipped",
        metadata={"input_question": question}
    ))
    
    # Set verifier report with gaps
    state.control_flags.last_verifier_report = {
        "coverage_score": 0.4,
        "missing_items": ["Need code examples", "Need usage patterns"],
    }

    planner = TaskPlannerAgent(llm=StubLLM())
    updated_state, new_tasks = planner.plan(state)

    # Should create new tasks because existing tasks are skipped
    assert len(new_tasks) >= 1
    for task in new_tasks:
        assert task.metadata.get("input_question") == question
        assert task.status == "pending"


def test_task_planner_blocks_duplicate_active_tasks():
    """Test that task planner still blocks duplicate tasks when active tasks exist."""
    question = "How do I locate the Foo class implementation?"
    buffer = ConversationBuffer(
        user_questions={"user_question1": question},
        history=[{"iteration": 0, "user_question1": question}],
    )
    state = ConversationState(conversation=buffer)
    
    # Add an active (pending) task
    existing_task = Task(
        id="task-1",
        type="repo_research",
        owner="repo_intelligence_agent",
        status="pending",  # This task is still active
        metadata={"input_question": question}
    )
    state.tasks.append(existing_task)

    planner = TaskPlannerAgent(llm=StubLLM())
    updated_state, new_tasks = planner.plan(state)

    # Should NOT create new tasks because there's an active task
    assert len(new_tasks) == 0


def test_task_planner_blocks_duplicate_in_progress_tasks():
    """Test that task planner blocks duplicate tasks when in_progress tasks exist."""
    question = "How do I locate the Foo class implementation?"
    buffer = ConversationBuffer(
        user_questions={"user_question1": question},
        history=[{"iteration": 0, "user_question1": question}],
    )
    state = ConversationState(conversation=buffer)
    
    # Add an in_progress task
    existing_task = Task(
        id="task-1",
        type="code_context",
        owner="code_inspector_agent",
        status="in_progress",  # This task is currently running
        metadata={"input_question": question}
    )
    state.tasks.append(existing_task)

    planner = TaskPlannerAgent(llm=StubLLM())
    updated_state, new_tasks = planner.plan(state)

    # Should NOT create new tasks because there's an in_progress task
    assert len(new_tasks) == 0
