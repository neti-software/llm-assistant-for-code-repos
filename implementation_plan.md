# Migration Plan: LangGraph Multi‑Agent Orchestration

## Goals

- Consolidate current tool-based workflow into a LangGraph multi-agent system.
- Promote existing tools to specialist agents that produce structured summaries with code examples for a supervising agent.
- Build and maintain a shared context across agents to minimize redundant work and enable compound reasoning.
- Preserve current capabilities (LLMs, embeddings, vector DB, AST analysis, repo search) while improving orchestration and observability.
- Minimize risk: incremental rollout with a feature flag and parity tests.

## Current Capabilities Inventory (from repo layout)

The repository structure and compiled module names indicate these capabilities:

- LLMs: `src/llm_module/{cloud_llm, local_llm, llm_builder}`
- Embeddings: `src/embedding_module/{cloud_embedding, local_embedding, emmbeding_builder}`
- Vector DB (Qdrant): `src/vector_db/{qdrant_vector_db, manager_qdrant_vector_db, helpers_vector_db}`
- Tools: `src/tools_to_call/{fetch_project_structure, search_files_with_grep, fetch_file_from_patch, tool_manager}`
- AST analysis: `src/ast/{metadata_extractor_python, metadata_extractor_go, metadata_extractor_javascript, metadata_extractor_rust, metadata_extractor_manager, metadata_validator}`
- Conversation: `src/conversation/conversation_history`
- Utilities: `src/utils/{logger, helper, profiler}`
- Configs: `configs/{openai_key.yaml, openrouter_key.yaml, qdrant_api_key.yaml, llm_config.yaml, qdrant_config.yaml}`
- CLI: `main_cli.py` (referenced by your IDE; integrate during migration)

These will map cleanly to LangGraph nodes with typed inputs/outputs and a shared state.

## Target Architecture (LangGraph)

### Agents (Nodes)

- Supervisor: Decides which specialist to invoke next and when to stop.
- Memory: Maintains conversation and working context; composes a running summary.
- Repo Surface Agent: Fast repo overview (files, sizes, languages), respects ignore rules.
- Code Search Agent: Grep/ripgrep-like queries; returns ranked hits with context windows.
- Patch Agent: Fetches files/segments from patches/commits or diffs.
- AST Agent: Runs language-specific metadata extractors and validators; emits typed symbols/relations.
- Embedding Agent: Chunks source, selects backend (cloud/local), and generates vectors.
- VectorDB Agent: Manages Qdrant collections; upserts and performs semantic search with filters.
- Reader Agent: Loads and chunks files on demand; de-duplicates by content hash.
- Synthesis Agent: Produces final answer with citations and code examples from evidence bank.

Notes:
- Some nodes can run in parallel (e.g., Repo Surface + Code Search) with a merge step.
- Existing `tool_manager` is superseded by LangGraph routing and a tool registry.

### Shared Graph State (Pydantic schema concept)

- query: user question/instructions
- session: ids, timestamps, run budget (time/tokens)
- repo: root, include/exclude globs, branch/commit
- artifacts: repo_map, files_read, patches, search_hits, ast_metadata
- retrieval: chunks, embeddings_indexed, vector_hits
- evidence_bank: list[Evidence] (unified data structure across agents)
- worklog: list[ActionEvent] for observability and debugging
- answer: draft text, code_snippets[], citations[], confidence, done flag

Evidence (normalized):
- id, kind (search|file|ast|vector|summary), source_tool, path, spans [start_line, end_line], text_excerpt, score, tags, created_at

CodeSnippet:
- language, path, start_line, end_line, text, rationale

Merging: Use deterministic merge functions per field (e.g., union by id for evidence, max-score dedup for hits, append-only worklog).

### Decision Policy (Supervisor)

- Bootstrap: Always invoke Repo Surface and Code Search in parallel for initial context.
- Routing heuristics:
  - If search coverage low or question implies semantics, call Retrieval (Embedding + VectorDB).
  - If question targets APIs/structure, call AST Agent.
  - If diffs are mentioned, call Patch Agent.
- Stopping:
  - Produce answer when confidence >= threshold OR budget/time cap reached.
  - Optionally one refinement cycle if new evidence improves confidence.

## Tool-to-Agent Mapping and Grouping

Direct mappings:
- fetch_project_structure → Repo Surface Agent
- search_files_with_grep → Code Search Agent
- fetch_file_from_patch → Patch Agent
- qdrant_vector_db/manager/helpers → VectorDB Agent
- cloud/local_embedding/builder → Embedding Agent
- metadata_extractor_* and validator → AST Agent
- conversation_history → Memory Agent
- cloud_llm/local_llm/llm_builder → LLM backends for Supervisor/Synthesis and node prompts

Grouping (subgraphs):
- Inspection Group: Repo Surface + Reader + AST
- Search Group: Code Search + Reader
- Retrieval Group: Embedding + VectorDB
- Synthesis Group: Supervisor + Synthesis + Memory

## Context Passing Protocol

- Observation Contract: Each agent returns Observations and SummaryDelta objects that the state merger consumes.
- Evidence Bank: Central, append-only store; each item includes provenance (tool, inputs), scoring, and optional cost metrics.
- Minimal Payloads: Pass references (path, span, content hash) more than full texts when possible; nodes fetch content lazily via Reader Agent.
- Code Example Policy: Each agent proposes snippets tied to evidence; Synthesis Agent selects and formats for the final answer.

## Migration Plan (Phased)

Phase 0 – Baseline and Safety Nets
- Capture current behavior and outputs for representative questions.
- Add a feature flag to keep legacy execution path available during rollout.

Phase 1 – Foundations
- ✅ Implemented initial shared state scaffolding via dataclasses in `src/langgraph/state_models.py` plus conversation adapters (LG-01).
- Add dependencies: langgraph, langchain-core, langchain-openai (or OpenRouter integration), pydantic v2.
- Define shared schemas (`GraphState`, `Evidence`, `CodeSnippet`, `ActionEvent`).
- Centralize config: keep key files under `configs/*.yaml`; add env var overrides.

Phase 2 – Wrap Tools as Nodes
- ✅ Added base `ToolNodeAdapter`, `ToolManager.list_tools`, and pytest coverage to normalise tool payloads (LG-02).
- Implement Repo Surface, Code Search, Reader, Patch, AST nodes with typed I/O and structured logging.
- Implement Embedding and VectorDB nodes (read from `qdrant_config.yaml`, `qdrant_api_key.yaml`).
- ✅ Repo Intelligence agent consuming adapters now produces structured evidence with tests (`src/langgraph/agents/repo_intelligence.py`).
- ✅ Code Inspector agent converts targeted file retrieval into evidence with tests (`src/langgraph/agents/code_inspector.py`).

Phase 3 – Subgraphs and Parallelism
- Build Inspection, Search, and Retrieval subgraphs; enable parallel execution where independent.
- Implement deterministic merge strategies for shared state.

Phase 4 – Supervisor and Policy
- Implement Supervisor with heuristic routing and stopping criteria.
- ✅ Heuristic Task Planner agent emits repo/code tasks with pytest coverage (LG-05 scaffolding).
- Define node edges explicitly; ensure backpressure and loop caps (max tool cycles).
- ✅ Verifier agent generates coverage reports and stores findings on state (`src/langgraph/agents/verifier.py`).
- ✅ Orchestrator coordinates planner, specialists, and verifier with iteration caps (`src/langgraph/orchestrator.py`).

Phase 5 – Synthesis and Output
- Implement Synthesis Agent that compiles answer, selects code snippets, and adds citations.
- Ensure outputs meet UX needs: concise summary + examples + explicit sources.
- ✅ Responder agent formats final answers and writes to conversation history (`src/langgraph/agents/responder.py`).

Phase 6 – CLI Integration
- ✅ Flag-gated LangGraph runner stub in `main_cli.py` using `LangGraphRunner` (LG-03).
- Update `main_cli.py` to run the graph, stream intermediate events, and support modes: `--explain`, `--dry-run`, `--legacy`.
- Add structured logs and a simple TUI progress view (optional).
- ✅ CLI now emits LangGraph progress messages and profiler events when the feature flag is enabled (`main_cli.py`, `src/utils/profiler.py`).
- ✅ LangGraph executor routes planner → specialists → verifier → responder, replacing the stub (`src/langgraph/executor.py`).

Phase 7 – Tests and Parity
- Unit tests per node with fixtures (small repos, fake qdrant, deterministic LLM stubs).
- E2E tests: ask multi-file questions; assert evidence and answer structure.
- Parity tests against legacy outputs for key prompts.
- ✅ Initial LangGraph integration test covering stubbed end-to-end flow (`tests/integration/test_langgraph_stub_flow.py`).

Phase 8 – Performance and Caching
- Content cache by content hash; memoize embeddings; avoid duplicate reads.
- Batch Qdrant upserts/queries; enable timeouts and retries.
- Add token/time budget enforcement; early stopping if budget exceeded.

Phase 9 – Telemetry and Observability
- Structured JSON logs for each ActionEvent; optional OpenTelemetry spans.
- Audit trail: store state snapshots per cycle for debugging.

## Data Model Sketches (documentation only)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal

class CodeSnippet(BaseModel):
    language: str
    path: str
    start_line: int
    end_line: int
    text: str
    rationale: Optional[str] = None

class Evidence(BaseModel):
    id: str
    kind: Literal["search","file","ast","vector","summary"]
    source_tool: str
    path: Optional[str] = None
    spans: Optional[List[Tuple[int,int]]] = None
    text_excerpt: Optional[str] = None
    score: Optional[float] = None
    tags: List[str] = []
    created_at: float

class ActionEvent(BaseModel):
    at: float
    actor: str
    action: str
    inputs: dict
    outputs: dict
    ok: bool

class GraphState(BaseModel):
    query: str
    session_id: str
    repo_root: str
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    repo_map: dict = Field(default_factory=dict)
    files_read: List[str] = Field(default_factory=list)
    patches: List[dict] = Field(default_factory=list)
    search_hits: List[dict] = Field(default_factory=list)
    ast_metadata: dict = Field(default_factory=dict)
    chunks: List[dict] = Field(default_factory=list)
    embeddings_indexed: bool = False
    vector_hits: List[dict] = Field(default_factory=list)
    evidence_bank: List[Evidence] = Field(default_factory=list)
    worklog: List[ActionEvent] = Field(default_factory=list)
    answer_draft: Optional[str] = None
    snippets: List[CodeSnippet] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    done: bool = False
```

Merge principles:
- Evidence: de-duplicate by `id` and `path:span` tuples; keep higher score.
- Hits: cap lists by score; store only top-k per tool.
- Worklog: append-only; include failure events.

## Error Handling and Guardrails

- Node-level retries with exponential backoff for transient errors (LLM and Qdrant).
- Timeouts per node; Supervisor enforces global budget.
- Input validation on all nodes; fail fast with actionable messages.
- Idempotency: avoid duplicate vector upserts by content hash.

## Security and Config

- Keys: load from `configs/*.yaml` and `.env`; support env overrides.
- Network: isolate calls per backend; optional HTTP proxies.
- Privacy: redact secrets from logs; cap snippet sizes in telemetry.

## Acceptance Criteria

- The graph answers questions by orchestrating at least three specialists and returns:
  - A concise summary.
  - 1–3 relevant code examples with file paths and line ranges.
  - Citations to sources and tools used.
- Parallel initial phase (Repo Surface + Search) works and reduces latency.
- Semantic retrieval via Qdrant is integrated and gated by heuristics.
- State is persisted for the run; reproducible logs exist.
- CLI can run legacy or graph paths via flag.

## Open Questions / Inputs Needed

- Confirm the full tool list and any missing modules not visible in this workspace.
- Define preferred confidence/stopping thresholds and token/time budgets.
- Confirm chunking strategy for embeddings (by lines, AST nodes, or hybrid?).
- Decide on snippet formatting and maximum count per answer.
- Any additional backends (e.g., local models) to enable at graph level?

## Timeline (indicative)

- Week 1: Foundations, schemas, Repo/Search node wrappers.
- Week 2: Embeddings/Qdrant nodes, subgraphs, Supervisor v1.
- Week 3: Synthesis agent, CLI integration, unit tests.
- Week 4: E2E tests, performance, telemetry, rollout flag + docs.

## Next Steps

- Validate the tool inventory and priorities.
- Approve schemas for GraphState/Evidence/Snippet.
- Start Phase 1 and create the graph scaffold.

