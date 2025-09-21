from src.llm_module.llm_builder import build_llm
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.tools_to_call.tool_manager import ToolManager
from src.conversation.conversation_history import ConversationHistory
from src.utils.helper import load_yaml
from src.langgraph.runner import LangGraphRunner
from src.langgraph.executor import execute_turn as langgraph_executor
from src.utils.profiler import execution_profiler, time_it

from colorama import Fore, Style, init
from langsmith.run_helpers import trace  # chain-level parent trace per chat turn

# Initialize colorama
init(autoreset=True)


def build_core():
    """Return initialized llm, tool_manager, conversation_history, and flag."""
    print(f"{Fore.CYAN}Loading configuration files...{Style.RESET_ALL}")

    llm_config = load_yaml("configs/llm_config.yaml")
    embedding_config = load_yaml("configs/embedding_config.yaml")
    reranker_config = load_yaml("configs/reranker_config.yaml")
    qdrant_config = load_yaml("configs/qdrant_config.yaml")
    conversation_history_config = load_yaml("configs/conversation_history_config.yaml")
    repo_metadata_manager_config = load_yaml("configs/json_schema/ast/metadata_schema.json")
    ignore_patterns_config = load_yaml("configs/ignore_patterns_config.yaml")
    repos_config = load_yaml("configs/repos_config.yaml")

    print(f"{Fore.CYAN}Initializing core components...{Style.RESET_ALL}")
    llm = build_llm(llm_config)
    # Indicator: Print PromptLayer status and first line (or issue)
    try:
        get_status = getattr(llm, "get_promptlayer_status", None)
        if callable(get_status):
            st = get_status()
            if isinstance(st, dict) and st.get("active"):
                yaml_name = st.get("yaml_prompt_name", "unknown")
                pl_name = st.get("prompt_name", "unknown")

                if isinstance(st, dict) and st.get("template_fetched") and st.get("first_line"):
                    print(f"{Fore.GREEN}PromptLayer loaded:{Style.RESET_ALL} name: {yaml_name} → {pl_name}")
                    print(f"{Fore.GREEN}Prompt preview:{Style.RESET_ALL} {st['first_line']}")
                elif isinstance(st, dict) and st.get("template_fetched"):
                    print(
                        f"{Fore.GREEN}PromptLayer loaded:{Style.RESET_ALL} name: {yaml_name} → {pl_name} (no preview)")
                else:
                    print(f"{Fore.YELLOW}PromptLayer connected:{Style.RESET_ALL} name: {yaml_name} → {pl_name}")
                    if isinstance(st, dict) and st.get("prompt_fetch_error"):
                        print(f"{Fore.YELLOW}Template fetch error:{Style.RESET_ALL} {st['prompt_fetch_error']}")
                        print(f"{Fore.YELLOW}This might cause fallback to generic prompt!{Style.RESET_ALL}")
            else:
                if not isinstance(st, dict):
                    st = {}
                yaml_name = st.get("yaml_prompt_name", "unknown")
                reasons = []
                if isinstance(st, dict) and not st.get("import_ok", True):
                    reasons.append("promptlayer not installed")

                if isinstance(st, dict) and not st.get("key_present", True):
                    reasons.append("PROMPT_LAYER_API_KEY missing in .env")
                if isinstance(st, dict) and st.get("client_init_error"):
                    reasons.append(f"client init error: {st['client_init_error']}")
                reason_text = "; ".join(reasons) or "unknown"
                print(
                    f"{Fore.YELLOW}PromptLayer disabled:{Style.RESET_ALL} {reason_text}. Using YAML prompt '{yaml_name}' instead.")
        else:
            # Backward-compatible minimal indicator
            pl_first_line = getattr(llm, "get_promptlayer_prompt_first_line", lambda: None)()
            if pl_first_line:
                print(f"{Fore.GREEN}PromptLayer loaded:{Style.RESET_ALL} {pl_first_line}")
    except Exception as e:
        print(f"{Fore.YELLOW}PromptLayer status error:{Style.RESET_ALL} {e}")
    manager_qdrant_vector_db = ManagerQdrantVectorDb(
        config=qdrant_config,
        embedding_config=embedding_config,
        reranker_config=reranker_config,
        repo_metadata_manager_config=repo_metadata_manager_config,
        ignore_patterns_config=ignore_patterns_config,
    )

    tool_manager = ToolManager(repos_config["path_to_repos"], ignore_patterns_config=ignore_patterns_config) # TODO , what to do with that path?
    tool_manager.add_tool_pointer("rag_search", manager_qdrant_vector_db.search)
    tool_manager.add_tool_pointer("rag_search_project_readme", manager_qdrant_vector_db.search_project_readme)

    conversation_history = ConversationHistory(conversation_history_config)

    print(f"{Fore.CYAN}Core initialization complete.{Style.RESET_ALL}")
    use_langgraph_multi_agent = llm_config.get("use_langgraph_multi_agent", False)

    return llm, tool_manager, conversation_history, use_langgraph_multi_agent


@time_it
def llm_loop(llm, tool_manager: ToolManager, conversation_history: ConversationHistory) -> dict:
    """Run a single LLM self-loop until it produces a final response, with step tracing."""
    iteration = 0
    while True:
        print(f"\n{Fore.CYAN}--- ITERATION: {iteration} ---{Style.RESET_ALL}\n")

        formated_input = conversation_history.to_json()

        # --- LLM Call ---
        want_tool, resp = llm.generate(formated_input)
        print(f"{Fore.GREEN}LLM RESPONSE:{Style.RESET_ALL} {resp}\n")

        if not want_tool:
            print(f"{Fore.MAGENTA}--- FINAL RESPONSE ---{Style.RESET_ALL}")
            return resp

        # --- Tool Call ---
        print(f"{Fore.YELLOW}Calling tool: {resp.get('action', 'N/A')}{Style.RESET_ALL}")
        tool_result = tool_manager.call_tool(resp)
        print(f"{Fore.YELLOW}Tool Result:{Style.RESET_ALL} {tool_result}\n")

        conversation_history.add_tool_call(resp["action"], resp.get("args", {}), tool_result)
        iteration += 1


def _extract_message_and_citations(response) -> tuple[str, list]:
    if response is None:
        return "", []
    if isinstance(response, str):
        return response, []
    if hasattr(response, "message"):
        return getattr(response, "message"), getattr(response, "citations", [])
    if isinstance(response, dict):
        message = response.get("message") or response.get("response") or str(response)
        citations = response.get("citations", [])
        return message, citations
    return str(response), []


@execution_profiler
def chat_loop(user_question, llm, tool_manager, conversation_history, runner: LangGraphRunner, live_log=None):
    """Add a user question, run loop, and return final model response."""
    conversation_history.add_user_question(user_question)
    using_graph = getattr(runner, "use_graph", False)
    if using_graph:
        print(f"{Fore.CYAN}LangGraph: planning tasks...{Style.RESET_ALL}")
        execution_profiler.record_event("graph_turn_start", question=user_question)

    # Use custom live_log if provided, otherwise use default
    if using_graph and live_log is None:
        live_log = lambda msg: print(f"{Fore.CYAN}{msg}{Style.RESET_ALL}")

    resp = runner.run_turn(
        llm,
        tool_manager,
        conversation_history,
        live_log=live_log,
    )

    if using_graph:
        execution_profiler.record_event("graph_turn_end", response_type=type(resp).__name__)

    message, citations = _extract_message_and_citations(resp)

    # Capture execution context if available
    execution_context = None
    if isinstance(resp, dict) and "execution_context" in resp:
        execution_context = resp["execution_context"]
        conversation_history.add_agent_execution_context(execution_context)

        # Add detailed evidence collection if available
        if "evidence_collection" in execution_context:
            conversation_history.add_evidence_collection(execution_context["evidence_collection"])

        # Add verifier report if available
        if "verifier_report" in execution_context:
            conversation_history.add_verifier_report(execution_context["verifier_report"])

        # Add execution metrics
        metrics = {
            "total_time": execution_context.get("total_time"),
            "total_iterations": execution_context.get("total_iterations"),
            "success": execution_context.get("success")
        }
        conversation_history.add_execution_metrics(metrics)

    conversation_history.add_model_response(message, metadata={
        "citations": citations,
        "execution_context_available": execution_context is not None
    })
    conversation_history.save()

    if using_graph:
        execution_profiler.record_event("graph_turn_persisted", citations=citations)
        print(f"{Fore.CYAN}LangGraph: turn complete.{Style.RESET_ALL}")

    return resp, message


def main():
    llm, tool_manager, conversation_history, use_langgraph_multi_agent = build_core()
    runner = LangGraphRunner(
        use_graph=use_langgraph_multi_agent,
        legacy_llm_loop=llm_loop,
    )

    if use_langgraph_multi_agent:
        runner.set_graph_executor(langgraph_executor)

    while True:
        user_question = input(f"{Fore.BLUE}USER QUESTION:{Style.RESET_ALL} ")
        if not user_question or user_question.lower().strip() in {"exit", "quit"}:
            print(f"{Fore.CYAN}Exiting...{Style.RESET_ALL}")
            break

        # Group all nested LLM/tool calls under one parent trace
        with trace(
            "RepoAssistant Chat Turn",
            run_type="chain",
            tags=["cli", "repo-assistant"],
            metadata={"question": user_question},
        ) as run:
            resp, final_message = chat_loop(user_question, llm, tool_manager, conversation_history, runner)
            # Record final output on the parent run (best-effort)
            try:
                run.end(outputs={"final_response": final_message})
            except Exception:
                pass
        output_text, _ = _extract_message_and_citations(resp)
        print(f"{Fore.MAGENTA}FINAL RESPONSE:{Style.RESET_ALL} {output_text}\n")

        execution_profiler.print_info()
        execution_profiler.clean()


if __name__ == "__main__":
    main()
