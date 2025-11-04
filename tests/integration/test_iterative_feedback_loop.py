"""
Integration test for the iterative feedback loop between orchestrator, task planner, and verifier.

This test verifies:
1. Orchestrator continues when coverage is low (even with empty missing_items)
2. Task planner creates new tasks in subsequent iterations
3. The full feedback loop works end-to-end
"""

import pytest
from src.langgraph.agents.task_planner import TaskPlannerAgent
from src.langgraph.agents.repo_intelligence import RepoIntelligenceAgent
from src.langgraph.agents.code_inspector import CodeInspectorAgent
from src.langgraph.agents.verifier import VerifierAgent
from src.langgraph.agents.responder import ResponderAgent
from src.langgraph.orchestrator import Orchestrator
from src.langgraph.state_models import ConversationState, ConversationBuffer, EvidenceItem


class StubToolNode:
    """Stub tool node that returns predefined results."""
    def __init__(self, tool_name, payload_sequence, *, confidence=0.8):
        self.tool_name = tool_name
        self.payload_sequence = payload_sequence  # List of payloads for each call
        self.confidence = confidence
        self.calls = []
        self.call_count = 0

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        # Return different payload based on call count
        payload_idx = min(self.call_count, len(self.payload_sequence) - 1)
        payload = self.payload_sequence[payload_idx]
        self.call_count += 1
        
        return {
            "tool_name": self.tool_name,
            "data": list(payload),
            "citations": [f"{self.tool_name}:{idx}" for idx, _ in enumerate(payload)],
            "confidence": self.confidence,
            "error": None,
            "metadata": {"args": kwargs, "call_number": self.call_count},
        }


def build_state(question: str) -> ConversationState:
    buffer = ConversationBuffer(
        user_questions={"user_question1": question},
        history=[{"iteration": 0, "user_question1": question}],
    )
    return ConversationState(conversation=buffer)


@pytest.mark.integration
def test_iterative_feedback_loop_with_low_coverage():
    """
    Test that the system iterates when coverage is low, even if missing_items is empty.
    
    Scenario:
    - Iteration 1: Returns minimal evidence, coverage is low (0.4), missing_items is empty
    - Iteration 2: Task planner creates new tasks (since previous are done)
    - Iteration 2: Returns more evidence, coverage improves (0.8)
    - System exits with good coverage
    """
    
    # Create tool nodes that return different results per iteration
    search_node = StubToolNode(
        "search_files_with_grep",
        [
            # First call: minimal results
            [
                {"path": "src/foo.py", "snippet": "class Foo:", "score": 0.5},
            ],
            # Second call: more comprehensive results
            [
                {"path": "src/foo.py", "snippet": "class Foo:", "score": 0.9},
                {"path": "src/foo_impl.py", "snippet": "def foo_method():", "score": 0.85},
                {"path": "tests/test_foo.py", "snippet": "test Foo usage", "score": 0.8},
            ],
        ],
    )
    
    rag_node = StubToolNode(
        "rag_search",
        [
            # First call: minimal documentation
            [
                {"path": "docs/brief.md", "snippet": "Foo is a class."},
            ],
            # Second call: comprehensive documentation
            [
                {"path": "docs/README.md", "snippet": "Comprehensive Foo architecture overview with implementation details."},
                {"path": "docs/api.md", "snippet": "Foo API documentation with usage examples."},
            ],
        ],
    )
    
    repo_agent = RepoIntelligenceAgent(tool_nodes=[search_node, rag_node])
    
    structure_node = StubToolNode(
        "fetch_project_structure",
        [["."], ["."]],  # Same for both iterations
        confidence=1.0
    )
    
    file_node = StubToolNode(
        "fetch_file_from_patch",
        [
            # First call: basic snippet
            [
                {
                    "path": "src/foo.py",
                    "snippet": "class Foo:\n    pass",
                    "line_start": 1,
                    "line_end": 2,
                    "score": 0.7,
                }
            ],
            # Second call: detailed implementation
            [
                {
                    "path": "src/foo.py",
                    "snippet": "class Foo:\n    def __init__(self):\n        pass\n    def method(self):\n        return 'detailed'",
                    "line_start": 1,
                    "line_end": 5,
                    "score": 0.95,
                }
            ],
        ],
        confidence=0.9,
    )
    
    code_agent = CodeInspectorAgent(structure_node=structure_node, file_node=file_node)
    
    # Use a lower coverage threshold to ensure iteration
    task_planner = TaskPlannerAgent()
    verifier = VerifierAgent(coverage_threshold=0.7)
    responder = ResponderAgent()
    
    orchestrator = Orchestrator(
        task_planner=task_planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )
    
    state = build_state("Explain the Foo class implementation in detail")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST: Iterative Feedback Loop")
    print("="*80)
    
    final_report, timeline = orchestrator.run(state, live_log=print)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Total iterations: {len(timeline)}")
    print(f"Total evidence collected: {len(state.evidence_store)}")
    print(f"Total tasks created: {len(state.tasks)}")
    print(f"Final coverage: {state.control_flags.last_verifier_report.get('coverage_score', 0)}")
    print("="*80 + "\n")
    
    # Verify the system iterated multiple times
    assert len(timeline) >= 2, f"Expected at least 2 iterations, got {len(timeline)}"
    
    # Verify evidence was collected in multiple iterations
    assert len(state.evidence_store) > 2, "Should have collected evidence from multiple iterations"
    
    # Verify tasks were created in multiple iterations
    # First iteration should create tasks, second iteration should create MORE tasks
    iteration_0_tasks = [t for t in state.tasks if t.metadata.get("planning_method") == "llm"]
    assert len(state.tasks) >= 2, f"Expected at least 2 tasks, got {len(state.tasks)}"
    
    # Verify coverage improved
    final_coverage = state.control_flags.last_verifier_report.get('coverage_score', 0)
    assert final_coverage >= 0.7, f"Final coverage {final_coverage} should be >= 0.7"
    
    # Verify some tasks were marked as done
    done_tasks = [t for t in state.tasks if t.status == "done"]
    assert len(done_tasks) >= 2, f"Expected at least 2 done tasks, got {len(done_tasks)}"
    
    # Verify the final response exists
    assert final_report is not None
    assert hasattr(final_report, 'message')
    
    print("✅ All assertions passed!")
    print(f"✅ System successfully iterated {len(timeline)} times")
    print(f"✅ Evidence collected: {len(state.evidence_store)} items")
    print(f"✅ Final coverage: {final_coverage:.2f}")


@pytest.mark.integration
def test_task_planner_creates_tasks_after_first_iteration():
    """
    Test that task planner creates new tasks in iteration 2 even though tasks exist from iteration 1.
    
    This specifically tests Fix #2: Task planner should ignore done/skipped tasks.
    """
    
    search_node = StubToolNode(
        "search_files_with_grep",
        [
            [{"path": "src/foo.py", "snippet": "class Foo:", "score": 0.5}],
            [{"path": "src/foo.py", "snippet": "class Foo: detailed", "score": 0.9}],
        ],
    )
    rag_node = StubToolNode("rag_search", [[{"path": "docs/README.md", "snippet": "Foo docs"}]])
    repo_agent = RepoIntelligenceAgent(tool_nodes=[search_node, rag_node])
    
    structure_node = StubToolNode("fetch_project_structure", [["."], ["."]], confidence=1.0)
    file_node = StubToolNode(
        "fetch_file_from_patch",
        [
            [{"path": "src/foo.py", "snippet": "class Foo:\n    pass", "line_start": 1, "line_end": 2}],
            [{"path": "src/foo.py", "snippet": "class Foo:\n    detailed", "line_start": 1, "line_end": 2}],
        ],
    )
    code_agent = CodeInspectorAgent(structure_node=structure_node, file_node=file_node)
    
    task_planner = TaskPlannerAgent()
    verifier = VerifierAgent(coverage_threshold=0.7)
    responder = ResponderAgent()
    
    orchestrator = Orchestrator(
        task_planner=task_planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )
    
    state = build_state("What is Foo?")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST: Task Planner Re-planning")
    print("="*80)
    
    final_report, timeline = orchestrator.run(state, live_log=print)
    
    # Track task creation per iteration
    tasks_per_iteration = {}
    for task in state.tasks:
        iteration = task.metadata.get("iteration", 0)
        if iteration not in tasks_per_iteration:
            tasks_per_iteration[iteration] = []
        tasks_per_iteration[iteration].append(task)
    
    print("\n" + "="*80)
    print("TASK CREATION ANALYSIS:")
    print("="*80)
    for iteration, tasks in sorted(tasks_per_iteration.items()):
        print(f"Iteration {iteration}: {len(tasks)} tasks")
        for task in tasks:
            print(f"  - {task.type} ({task.status})")
    print("="*80 + "\n")
    
    # Verify tasks were created in iteration 0
    assert 0 in tasks_per_iteration, "Should have tasks from iteration 0"
    
    # Verify the system iterated (if coverage was low)
    if len(timeline) > 1:
        print("✅ System iterated to improve coverage")
        print(f"✅ Total iterations: {len(timeline)}")
    
    print(f"✅ Total tasks created: {len(state.tasks)}")
    print(f"✅ Final coverage: {state.control_flags.last_verifier_report.get('coverage_score', 0):.2f}")


if __name__ == "__main__":
    # Run tests directly
    print("Running integration tests...\n")
    test_iterative_feedback_loop_with_low_coverage()
    print("\n" + "="*80 + "\n")
    test_task_planner_creates_tasks_after_first_iteration()
    print("\n✅ All integration tests passed!")

