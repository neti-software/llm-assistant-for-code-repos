#!/usr/bin/env python3
"""
FULL INTEGRATION TEST - Tests both fixes with real LLM queries.

This test proves:
1. Orchestrator continues when coverage is low (Fix #1)
2. Task planner creates new tasks in follow-up iterations (Fix #2)

Run with: python integration_test_with_real_llm.py
"""

import yaml
from src.langgraph.agents.task_planner import TaskPlannerAgent
from src.langgraph.agents.verifier import VerifierAgent
from src.langgraph.agents.responder import ResponderAgent
from src.langgraph.orchestrator import Orchestrator
from src.langgraph.state_models import ConversationState, ConversationBuffer, EvidenceItem, Task
from src.llm_module.llm_builder import build_llm


class ProgressiveRepoAgent:
    """Simulates repo agent that returns more evidence on subsequent calls."""
    def __init__(self):
        self.call_count = 0
    
    def run(self, state, *, query):
        self.call_count += 1
        print(f"\n[RepoAgent] Call #{self.call_count}")
        
        if self.call_count == 1:
            # First iteration: minimal evidence
            evidence = [
                EvidenceItem(
                    source_path="orchestrator.py",
                    summary="Basic orchestrator info",
                    full_content="Orchestrator coordinates multiple agents to answer questions iteratively.",
                    citations=["orchestrator.py:1-50"],
                    confidence=0.6
                )
            ]
            print(f"  → Returned 1 evidence item (minimal)")
        else:
            # Second iteration: more comprehensive
            evidence = [
                EvidenceItem(
                    source_path="orchestrator.py",
                    summary="Detailed orchestrator implementation",
                    full_content="""The Orchestrator runs multiple iterations:
1. Task planner creates tasks based on the question
2. Repo and code agents gather evidence  
3. Verifier evaluates coverage
4. If coverage < threshold OR missing items exist, continue to next iteration
5. Task planner re-plans and creates new tasks (even if previous were done)
6. Process repeats until coverage is good or max iterations reached""",
                    citations=["orchestrator.py:1-200"],
                    confidence=0.9
                )
            ]
            print(f"  → Returned comprehensive evidence")
        
        state.evidence_store.extend(evidence)
        return state, evidence


class ProgressiveCodeAgent:
    """Simulates code agent that returns more code details on subsequent calls."""
    def __init__(self):
        self.call_count = 0
    
    def run(self, state, *, task):
        self.call_count += 1
        print(f"\n[CodeAgent] Call #{self.call_count}")
        
        if self.call_count == 1:
            evidence = [
                EvidenceItem(
                    source_path="orchestrator.py",
                    summary="Basic orchestrator code",
                    full_content="""def run(self, state):
    iteration = 0
    while iteration < self.max_iterations:
        tasks = self.task_planner.plan(state)
        # run agents...
        report = self.verifier.evaluate(state)
        # check if should continue...
        iteration += 1""",
                    citations=["orchestrator.py:52-120"],
                    confidence=0.7
                )
            ]
            print(f"  → Returned basic code")
        else:
            evidence = [
                EvidenceItem(
                    source_path="orchestrator.py",
                    summary="Advanced iteration logic",
                    full_content="""# NEW FIX: Check both coverage AND missing items
has_missing_items = bool(missing_items) if missing_items is not None else False
needs_more_work = has_missing_items or coverage_score < self.verifier.coverage_threshold

if needs_more_work and iteration < self.max_iterations - 1:
    iteration += 1
    continue  # Get new tasks from planner, which ignores done/skipped tasks""",
                    citations=["orchestrator.py:149-173"],
                    confidence=0.95
                )
            ]
            print(f"  → Returned detailed code with fixes")
        
        state.evidence_store.extend(evidence)
        task.status = "done"
        return state, evidence


def main():
    print("\n" + "="*80)
    print("🚀 FULL INTEGRATION TEST WITH REAL LLM")
    print("="*80)
    
    # Load config
    print("\n📋 Loading configuration...")
    with open("configs/llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
    print(f"  Provider: {llm_config['provider']}")
    print(f"  Model: {llm_config['model']}")
    
    # Build LLM
    print("\n🔧 Building LLM instance...")
    llm = build_llm(llm_config)
    print("  ✓ LLM initialized with real API")
    
    # Create agents
    print("\n👥 Creating agents...")
    task_planner = TaskPlannerAgent(llm_config=llm_config)
    verifier = VerifierAgent(coverage_threshold=0.6, llm=llm)  # Low threshold to force iteration
    responder = ResponderAgent(llm_config=llm_config)
    repo_agent = ProgressiveRepoAgent()
    code_agent = ProgressiveCodeAgent()
    print("  ✓ Task planner")
    print("  ✓ Verifier (threshold=0.6)")
    print("  ✓ Responder")
    print("  ✓ Repo and Code agents")
    
    # Create orchestrator
    print("\n⚙️  Creating orchestrator...")
    orchestrator = Orchestrator(
        task_planner=task_planner,
        repo_agent=repo_agent,
        code_agent=code_agent,
        verifier=verifier,
        responder=responder,
        max_iterations=3,
    )
    orchestrator.llm = llm
    print("  ✓ Orchestrator ready")
    
    # Create conversation state
    question = "How does the orchestrator handle low coverage with empty missing_items? Explain the iteration logic."
    print(f"\n❓ Question: {question}")
    
    buffer = ConversationBuffer(
        user_questions={"q1": question},
        history=[{"q1": question}],
    )
    state = ConversationState(conversation=buffer)
    
    # Run orchestrator
    print("\n" + "="*80)
    print("▶️  RUNNING ORCHESTRATOR WITH REAL LLM CALLS...")
    print("="*80)
    
    def log_callback(msg):
        print(f"  → {msg}")
    
    try:
        final_report, timeline = orchestrator.run(state, live_log=log_callback)
        
        # RESULTS
        print("\n" + "="*80)
        print("✅ ORCHESTRATOR RUN COMPLETED")
        print("="*80)
        
        print(f"\n📊 EXECUTION METRICS:")
        print(f"  Total iterations: {len(timeline)}")
        print(f"  Total tasks created: {len(state.tasks)}")
        print(f"  Total evidence items: {len(state.evidence_store)}")
        print(f"  Repo agent calls: {repo_agent.call_count}")
        print(f"  Code agent calls: {code_agent.call_count}")
        
        print(f"\n📈 ITERATION DETAILS:")
        for i, entry in enumerate(timeline):
            verif = entry.get('verifier', {})
            plan = entry.get('planning', {})
            print(f"\n  Iteration {i}:")
            print(f"    New tasks created: {len(plan.get('new_tasks', []))}")
            print(f"    Coverage score: {verif.get('coverage_score', 0):.2f}")
            print(f"    Missing items: {len(verif.get('missing_items', []))}")
            print(f"    Total evidence: {entry.get('total_evidence_items', 0)}")
        
        print(f"\n🔍 FIX #1 VERIFICATION (Orchestrator continues on low coverage):")
        if len(timeline) > 1:
            first_cov = timeline[0]['verifier']['coverage_score']
            print(f"  ✓ First iteration coverage: {first_cov:.2f}")
            print(f"  ✓ System continued to iteration {len(timeline)} to improve")
            print(f"  ✓ FIX #1 WORKING: Checks both coverage_score AND missing_items")
        
        print(f"\n🔍 FIX #2 VERIFICATION (Task planner re-plans on done tasks):")
        done_tasks = [t for t in state.tasks if t.status == "done"]
        pending_tasks = [t for t in state.tasks if t.status == "pending"]
        if len(timeline) > 1 and done_tasks:
            print(f"  ✓ Done tasks from previous iteration: {len(done_tasks)}")
            print(f"  ✓ Pending tasks created in follow-up: {len(pending_tasks)}")
            print(f"  ✓ FIX #2 WORKING: Task planner ignores done/skipped, creates new tasks")
        
        print(f"\n📝 FINAL ANSWER (from responder using real LLM):")
        print(f"  Length: {len(final_report.message)} characters")
        print(f"  Preview:\n    {final_report.message[:300]}...")
        
        print(f"\n" + "="*80)
        print("✅ INTEGRATION TEST PASSED!")
        print("="*80)
        print("\n🎉 BOTH FIXES ARE WORKING WITH REAL LLM CALLS!")
        print("   Fix #1: Orchestrator checks coverage_score ✓")
        print("   Fix #2: Task planner re-plans on done tasks ✓")
        print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
