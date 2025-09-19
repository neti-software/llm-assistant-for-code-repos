# LangGraph Migration Plan

## 1. Objectives and Success Criteria
- Introduce a LangGraph-driven multi-agent architecture that preserves the current CLI experience while enabling richer, tool-aware conversations.
- Ensure every tool can be triggered by a dedicated specialist agent and report structured results to a primary decision-maker that judges data sufficiency.
- Support iterative refinement loops, guardrails, and observability (LangSmith traces, internal telemetry) through the graph structure.
- Minimise disruption to existing components (`ToolManager`, Qdrant, `ConversationHistory`) and provide progressive rollout plus rollback options.

## 2. Current State Snapshot
- **Interaction model**: Single `llm_loop` in `main_cli.py` orchestrates tool calls until the LLM returns a final answer.
- **Tool surface**: `ToolManager` exposes `rag_search`, `rag_search_project_readme`, `fetch_project_structure`, `fetch_file_from_patch`, and `search_files_with_grep` with repo-root injection.
- **Memory and state**: `ConversationHistory` persists turns; Qdrant vector store powers retrieval through `ManagerQdrantVectorDb`.
- **Tracing**: LangSmith wraps each chat turn but lacks nested span detail for individual tool executions.
- **Limitations**: Tool choice, stopping criteria, and validation are all implicit in a single prompt; no explicit sufficiency checks or specialist agents exist.

## 3. Target Multi-Agent LangGraph Architecture
### 3.1 High-Level Flow (ASCII sketch)
```
User Question
  -> Conversation Loader (hydrate shared state)
  -> Orchestrator Agent (decide next action)
      -> Task Sufficient? (coverage gate)
          No  -> Task Planner Agent (decompose + route)
                    -> Repo Intelligence Agent
                    -> Code Inspector Agent
                    -> External Knowledge Agent (optional)
                <- Structured evidence with confidence + citations
                -> Orchestrator updates state (loop)
          Yes -> Verifier Agent (draft + evaluate coverage)
                    -> requires more data? Yes -> Orchestrator (loop)
                    -> No -> Responder -> Final answer & persistence
```

### 3.2 Agent Roles and Responsibilities
| Agent | Primary Goal | Tool Access | Notes |
| --- | --- | --- | --- |
| Orchestrator (Conductor) | Maintain global plan, decide next steps, track sufficiency | Graph state only | Central control node; updates shared state, enforces budgets.
| Task Planner Agent | Break questions into subtasks and assign the right specialist | Tool registry metadata | Combines former decomposer + router roles; outputs structured tasks with routing directives.
| Repo Intelligence Agent | Retrieve textual + metadata context from the repository | `rag_search`, `rag_search_project_readme`, `search_files_with_grep`, future metadata helpers | Merges prior Repo Research + metadata duties; emits summaries with provenance.
| Code Inspector Agent | Fetch precise code and structural context | `fetch_file_from_patch`, `fetch_project_structure` | Maintains call budget; can request AST or diff parsing in future.
| External Knowledge Agent | Query external sources (issues, docs, APIs) | HTTP/OpenAPI connectors (future) | Initially stubbed; activated once connectors exist.
| Verifier Agent | Compose answer, judge coverage, and police hallucinations | Prompt-only with optional read-only tool reuse | Produces final draft, `coverage_score`, `missing_items`, and go/no-go signal.
| Responder | Deliver final answer, update memory, emit telemetry | `ConversationHistory` adapter | Writes final response, citations, and summarised state.

### 3.3 Tool Assignment Strategy
- Keep `ToolManager` as the single source of truth; wrap each tool inside LangGraph `ToolNode` adapters that emit structured `ToolResult` payloads `{data, citations, confidence, errors}`.
- Assign tools per agent:
  - Repo Intelligence Agent -> `rag_search`, `rag_search_project_readme`, `search_files_with_grep`, plus future metadata readers (README summary, repo metadata manager adapters).
  - Code Inspector Agent -> `fetch_file_from_patch`, `fetch_project_structure` (later extend with AST utilities).
  - External Knowledge Agent -> HTTP/OpenAPI connectors once implemented; keep disabled behind feature switch initially.
- Allow Verifier Agent to request limited reruns of Repo Intelligence tools via orchestrator-mediated approvals (guarded by loop counter and cost budget).
- Provide a legacy `general_tool_call` fallback node that proxies the old single-agent behaviour for safe rollback.

### 3.4 Current Tools and Owning Agents
- `rag_search` -> Repo Intelligence Agent (broad semantic retrieval across repo content).
- `rag_search_project_readme` -> Repo Intelligence Agent (README-focused context gathering).
- `search_files_with_grep` -> Repo Intelligence Agent (pattern-based text search with ignore rules).
- `fetch_project_structure` -> Code Inspector Agent (directory tree snapshots with ignore patterns).
- `fetch_file_from_patch` -> Code Inspector Agent (targeted file/diff retrieval for precise context).

### 3.5 Control Flow, Loops, and Double Checks
- **Task Expansion Loop**: Orchestrator -> Task Planner -> Specialist Agents. Orchestrator reviews evidence metadata after each pass; if gaps remain, it schedules additional tasks until the Verifier signals readiness or iteration caps hit.
- **Verifier Synthesis Loop**: When triggered, the Verifier drafts an answer, scores coverage/confidence, and surfaces `missing_items`. If it flags insufficiency, the orchestrator adds targeted follow-up tasks before re-invoking the Verifier.
- **Tool Failure Loop**: ToolNode errors captured in `ToolResult.errors`; orchestrator retries with adjusted arguments (max two retries) or escalates to fallback node / user clarification.
- **Budget Enforcement**: State stores per-agent token/tool budgets; orchestrator halts loops when limits hit and surfaces partial findings.
- **Exit Criteria**: Final response sent only when the Verifier approves (no blocking `missing_items`) and Responder persists the answer alongside citations.

## 4. Shared State and Memory Strategy
- Use a LangGraph `StateGraph` (or typed state store) with primary fields:
  - `conversation`: running dialogue plus summarised context, backed by `ConversationHistory` adapter for persistence.
  - `tasks`: queue of outstanding subtasks with status (`pending`, `in_progress`, `done`), owning agent, priority, and retry count.
  - `evidence_store`: normalised list of evidence items `{source_path, span, summary, citations, confidence}`.
  - `control_flags`: loop counters, cost budget, coverage thresholds, verification status.
- Persist raw tool payloads to disk cache/logs for audit; maintain summarised snapshots to keep prompts within token limits.
- Validate state transitions with Pydantic models to catch schema drift during development.

## 5. Migration Phases
1. **Foundation (Week 1-2)**
   - Introduce LangGraph state models and wrap existing tools into `ToolNode` adapters.
   - Implement Orchestrator node that still calls the legacy `llm_loop` logic internally (bridge mode).
   - Enable LangSmith sub-traces per node and collect baseline metrics (latency, tokens, tool usage).
2. **Specialist Agents (Week 3-4)**
   - Add Repo Intelligence and Code Inspector agents with dedicated prompts and ToolNode bindings.
   - Implement the Task Planner agent; enforce iteration and budget guards.
   - Update CLI to show agent activity (e.g., "Researching...", "Inspecting code...").
3. **Verifier & Finalisation Layer (Week 5)**
   - Empower the Verifier agent to draft responses, score coverage, and emit `missing_items` plus confidence.
   - Wire the feedback loop that routes Verifier "insufficient" signals back into orchestrated task generation.
   - Produce structured final responses (citations, confidence) and persist them through `ConversationHistory`.
4. **Expansion and Hardening (Week 6+)**
   - Introduce External Knowledge connectors as they become available.
   - Add confidence calibration, telemetry dashboards, and automated regression checks.
   - Implement feature flag driven rollout (`USE_LANGGRAPH_MULTI_AGENT`) with clear rollback path.

## 6. Implementation Checklist
- [x] Design Pydantic state schemas (`ConversationState`, `Task`, `EvidenceItem`, `ControlFlags`).
  - Implemented as dataclasses in `src/langgraph/state_models.py` with pytest coverage in `tests/langgraph/test_state_models.py`.
- [x] Build ToolNode adapters that normalise outputs and handle error reporting.
  - Added reusable adapter in `src/langgraph/tool_nodes/base.py`, ensured metadata exposure via `ToolManager.list_tools()`, and covered with pytest in `tests/langgraph/test_tool_nodes.py`.
- [ ] Author orchestrator prompt with explicit sufficiency rubric and loop budgeting instructions.
- [ ] Implement the Task Planner agent that produces structured tasks with routing directives.
- [ ] Create specialist agent prompts tuned to their toolsets and expected outputs.
- [ ] Extend Repo Intelligence agent to surface metadata summaries alongside retrieval snippets.
- [ ] Implement Verifier agent logic for drafting, coverage scoring, `missing_items`, and follow-up task suggestions.
- [ ] Integrate `ConversationHistory` persistence with graph state transitions and CLI display.
- [ ] Add LangSmith span trees per node plus internal telemetry (iteration counts, tool calls, latency).
- [ ] Update CLI to stream graph progress and handle orchestrator fallback notices gracefully.

## 7. Testing and Validation Plan
- **Unit tests**: Validate state schema conversions, routing decisions, and ToolNode adapters (including error paths).
- **Integration tests**: Simulate representative tasks (bug triage, feature request explanation) and assert loop termination plus evidence sufficiency.
- **Regression harness**: Compare outputs between legacy loop and LangGraph flow; flag regressions in accuracy or latency.
- **Load tests**: Exercise long conversations to observe memory growth, loop behaviour, and token usage.
- **Manual QA**: Review LangSmith traces to verify agent sequencing, loop exits, and citation integrity.

## 8. Observability and Guardrails
- Leverage LangSmith run trees with node-specific tags to visualise execution.
- Record `coverage_score`, `verification_score`, token usage, and tool retry counts for dashboards (all sourced from the Verifier agent).
- Enforce per-agent budgets and global iteration caps to prevent runaway costs.
- Provide graceful degradation: on graph failure, fall back to legacy single-agent mode and notify the user.

## 9. Risks and Mitigations
- **Prompt drift**: Use prompt versioning and regression tests to catch behavioural changes early.
- **Tool brittleness**: Maintain contract tests for each ToolNode; add CI checks when tool signatures change.
- **Loop non-termination**: Guard with iteration counters, stateful stop conditions, and watchdog alerts in telemetry.
- **Latency inflation**: Measure per-node timing; parallelise independent retrieval tasks when safe; restrict heavy agents to essential calls.
- **State explosion**: Cap evidence store size and summarise or evict low-confidence entries beyond the budget.

## 10. Rollout Strategy
- Introduce config flag `USE_LANGGRAPH_MULTI_AGENT` defaulting to false; document enabling steps.
- Run canary tests with internal users; collect qualitative and quantitative feedback.
- After parity with legacy flow is confirmed, flip the default while keeping rollback switch for at least one release cycle.
- Update README and internal docs to describe the graph architecture, troubleshooting steps, and new telemetry signals.

## 11. Follow-Up Enhancements
- Add an autonomous ticket-generation capability for unresolved gaps with structured handover notes.
- Introduce long-term memory (vector plus episodic summaries) for cross-session recall and agent grounding.
- Experiment with speculative execution: spawn multiple Repo Intelligence strategies in parallel and merge evidence automatically.

This plan enables an incremental migration where specialist agents can be validated in isolation while the orchestrator and Verifier enforce data sufficiency, double checks, and safe fallbacks before final responses are delivered.

## 12. Ticketed Implementation Plan
Each ticket groups work into reviewable units (~1-2 dev days) with explicit predecessors and code touchpoints. IDs are suggestions; adjust to your internal tracker naming.

### LG-01: Establish LangGraph State Models
- **Status**: ✅ Completed.
- **Summary**: Create typed conversation state (conversation, tasks, evidence, control flags) to back the graph and expose adapters to current persistence.
- **Depends on**: None.
- **Key changes**:
  - Added dataclass-based models in `src/langgraph/state_models.py` translating `ConversationHistory` content into structured buffers.
  - Extended `ConversationHistory` with `to_state_snapshot()` / `apply_state_delta()` adapters, keeping legacy API untouched.
  - Placeholder feature flag entry added to `configs/llm_config.yaml`.
- **Tests**: `python3 -m pytest tests/langgraph/test_state_models.py` (note: ensure pytest available to interpreter).

### LG-02: Wrap Existing Tools as ToolNode Adapters
- **Status**: ✅ Completed.
- **Summary**: Convert the current `ToolManager` surface into LangGraph-compatible adapters returning structured payloads.
- **Depends on**: LG-01.
- **Key changes**:
  - Introduced `src/langgraph/tool_nodes/base.py` with `ToolNodeAdapter` and `build_tool_result` helpers normalising responses.
  - Enhanced `ToolManager` to expose metadata via `list_tools()` and support adapter-friendly registration data.
  - Added pytest suite in `tests/langgraph/test_tool_nodes.py` covering success/error payloads and tool registry introspection.
- **Tests**: `python3 -m pytest tests/langgraph/test_tool_nodes.py` (note: ensure pytest available to interpreter).

### LG-03: Feature Flag and Graph Bootstrap in CLI
- **Status**: ✅ Completed.
- **Summary**: Introduce feature flag plumbing and skeleton to run LangGraph graph while preserving legacy loop fallback.
- **Depends on**: LG-01, LG-02.
- **Key changes**:
  - Read `use_langgraph_multi_agent` in `build_core()` and pass flag through (`main_cli.py:15`).
  - Added `LangGraphRunner` bridging legacy `llm_loop` with future graph executors, including state snapshot/apply hooks (`src/langgraph/runner.py`).
  - Wired `chat_loop()` to route turns through the runner while keeping conversation persistence (`main_cli.py:115`).
  - Registered a LangGraph executor that wires the planner, specialists, verifier, and responder before returning the final response (`src/langgraph/executor.py`).
  - Pytest suite (`tests/langgraph/test_runner.py`) verifies fallback, state hydration, and error-free state application (requires local pytest install).
- **Next steps**: Tune executor prompts/heuristics once remaining agents and guardrails land.

### LG-04: LangSmith Sub-Traces per Node
- **Summary**: Instrument LangGraph nodes with nested LangSmith spans for observability.
- **Depends on**: LG-03 (graph runner shell).
- **Key changes**:
  - Enhance `runner.py` to wrap each node execution with `with trace(...):` using node IDs.
  - Keep existing top-level trace (`main_cli.py:133`) but enrich metadata with node-level outputs for dashboards.
- **Example delta**: When Repo Intelligence agent runs, emit `trace("Repo Intelligence")` containing retrieved sources in metadata instead of printing from `llm_loop`.

### LG-05: Task Planner Agent (Decompose + Route)
- **Status**: ✅ Completed.
- **Summary**: Implement the combined task decomposition/routing node with deterministic JSON output.
- **Depends on**: LG-03 (graph shell), LG-02 (tool registry access).
- **Key changes so far**:
  - Added heuristic-based `TaskPlannerAgent` generating structured `Task` entries for repo research and code context (`src/langgraph/agents/task_planner.py`).
  - Introduced pytest coverage ensuring task creation, deduplication, and fallback behaviour (`tests/langgraph/test_task_planner.py`).
- **Next steps**:
  - Connect planner output to orchestrator once implemented and replace heuristics with LLM-backed decomposition.

### LG-06: Repo Intelligence Agent
- **Status**: ✅ Completed.
- **Summary**: Build the retrieval-focused agent covering repo search plus metadata summaries.
- **Depends on**: LG-02, LG-05.
- **Key changes so far**:
  - Implemented `RepoIntelligenceAgent` that consumes ToolNode adapters and converts results into `EvidenceItem`s (`src/langgraph/agents/repo_intelligence.py`).
  - Added pytest coverage to validate evidence generation, metadata capture, and error handling (`tests/langgraph/test_repo_intelligence_agent.py`).
- **Next steps**:
  - Expand evidence normalisation for README summaries and future metadata helpers.

### LG-07: Code Inspector Agent
- **Status**: ✅ Completed.
- **Summary**: Provide narrow, file-level retrieval agent for structural/code context.
- **Depends on**: LG-02, LG-05.
- **Key changes**:
  - Added `CodeInspectorAgent` that coordinates structure discovery and precise file fetching, translating results into `EvidenceItem`s (`src/langgraph/agents/code_inspector.py`).
  - Introduced pytest coverage to validate snippet extraction, metadata capture, and error handling (`tests/langgraph/test_code_inspector_agent.py`).
- **Next steps**: Wire the agent into the orchestrator loop and extend metadata enrichment (AST, diff context) in later phases.

### LG-08: Verifier Agent (Draft + Coverage Gate)
- **Status**: ✅ Completed.
- **Summary**: Implement combined drafting and coverage evaluation responsibilities.
- **Depends on**: LG-05, LG-06, LG-07 (needs evidence supply).
- **Key changes**:
  - Added `VerifierAgent` that summarises evidence, estimates coverage, and records verifier reports on the shared state (`src/langgraph/agents/verifier.py`).
  - Pytest suite validates summary drafting, coverage scoring, missing-item feedback, and state updates (`tests/langgraph/test_verifier_agent.py`).
- **Next steps**: Surface verifier feedback through the orchestrator so low-coverage reports trigger follow-up tasks.

### LG-09: Orchestrator Node & Loop Controller
- **Status**: ✅ Completed.
- **Summary**: Formalise the orchestrator logic managing loops, budgets, and dispatch.
- **Depends on**: LG-05, LG-08.
- **Key changes**:
  - Added `Orchestrator` coordinating the planner, Repo Intelligence, Code Inspector, and Verifier agents with iteration caps (`src/langgraph/orchestrator.py`).
  - Pytest coverage verifies iteration control, agent dispatch, and verifier-driven stopping (`tests/langgraph/test_orchestrator.py`).
- **Next steps**: Integrate the orchestrator into the CLI runner once remaining agents are wired.

- ### LG-10: Responder & Conversation Persistence Adapter
- **Status**: ✅ Completed.
- **Summary**: Finalise answer output and ensure conversation history stays in sync.
- **Depends on**: LG-08, LG-09.
- **Key changes**:
  - Added `ResponderAgent` that formats the final answer, returns citations, and persists the message via conversation history (`src/langgraph/agents/responder.py`).
  - Pytest coverage validates formatted output and persistence hooks (`tests/langgraph/test_responder_agent.py`).

### LG-11: CLI Progress & Telemetry Updates
- **Status**: ✅ Completed.
- **Summary**: Surface graph progress to users and emit telemetry counters.
- **Depends on**: LG-09 (orchestrator signals).
- **Key changes**:
  - Added lightweight LangGraph status messaging and event telemetry in the CLI when the feature flag is enabled (`main_cli.py`).
  - Extended the global `execution_profiler` with structured event recording for downstream telemetry consumers (`src/utils/profiler.py`).
- **Next steps**: Feed recorded events into LangSmith/OpenTelemetry pipelines once the full graph replaces the stub executor.

### LG-12: Testing & Regression Harness
- **Status**: ✅ Completed (initial harness).
- **Summary**: Build automated tests covering state transitions, agent outputs, and parity with legacy loop.
- **Depends on**: LG-05 through LG-10 (core graph functionality).
- **Key changes**:
  - Extended unit coverage for planner, agents, and orchestrator under `tests/langgraph/`.
  - Added an integration test that exercises the stubbed LangGraph pipeline end to end (`tests/integration/test_langgraph_stub_flow.py`).
- **Next steps**: Add parity tests against the legacy loop once the full LangGraph graph replaces the stub executor.

### LG-13: Observability & Guardrails Hardening
- **Summary**: Final pass adding budgets, failure handling, and dashboards.
- **Depends on**: LG-11, LG-12.
- **Key changes**:
  - Implement budget enforcement utilities referenced by orchestrator (token/tool retry caps).
  - Configure LangSmith tags per agent and export metrics to dashboards.
  - Add fallback path that toggles `USE_LANGGRAPH_MULTI_AGENT` off when severe errors detected.
- **Example delta**: When a tool adapter raises repeated errors, orchestrator records failure in state and returns legacy-mode response explaining the fallback.

### LG-14: External Knowledge Connectors (Optional Phase)
- **Summary**: Introduce future HTTP/OpenAPI agent once upstream dependencies ready.
- **Depends on**: LG-09 (orchestrator), LG-13 (guardrails).
- **Key changes**:
  - Define connector interface under `src/langgraph/tool_nodes/http_connector.py` (stub until keys provided).
  - Update Task Planner to emit `external_research` task type when appropriate.
  - Document new configuration in `configs/prompt_config.yaml`.
- **Example delta**: Planner emits a task targeting External Knowledge agent which invokes HTTP connector and stores external citations alongside repo evidence.
Ticket dependencies follow the migration phases: LG-01 -> LG-02 -> LG-03/04 -> LG-05/06/07 -> LG-08/09 -> LG-10/11 -> LG-12 -> LG-13, with LG-14 optional once external connectors are prioritised.
