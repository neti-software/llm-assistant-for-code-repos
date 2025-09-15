from src.llm_module.llm_builder import build_llm
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.tools_to_call.tool_manager import ToolManager
from src.conversation.conversation_history import ConversationHistory
from src.utils.helper import load_yaml
from src.utils.profiler import execution_profiler, time_it

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def build_core():
    """Return initialized llm, tool_manager, conversation_history."""
    print(f"{Fore.CYAN}Loading configuration files...{Style.RESET_ALL}")

    llm_config = load_yaml("configs/llm_config.yaml")
    embedding_config = load_yaml("configs/embedding_config.yaml")
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
            if st.get("active"):
                yaml_name = st.get("yaml_prompt_name", "unknown")
                pl_name = st.get("prompt_name", "unknown")

                if st.get("template_fetched") and st.get("first_line"):
                    print(f"{Fore.GREEN}PromptLayer loaded:{Style.RESET_ALL} name: {yaml_name} → {pl_name}")
                    print(f"{Fore.GREEN}Prompt preview:{Style.RESET_ALL} {st['first_line']}")
                elif st.get("template_fetched"):
                    print(
                        f"{Fore.GREEN}PromptLayer loaded:{Style.RESET_ALL} name: {yaml_name} → {pl_name} (no preview)")
                else:
                    print(f"{Fore.YELLOW}PromptLayer connected:{Style.RESET_ALL} name: {yaml_name} → {pl_name}")
                    if st.get("prompt_fetch_error"):
                        print(f"{Fore.YELLOW}Template fetch error:{Style.RESET_ALL} {st['prompt_fetch_error']}")
                        print(f"{Fore.YELLOW}This might cause fallback to generic prompt!{Style.RESET_ALL}")
            else:
                yaml_name = st.get("yaml_prompt_name", "unknown")
                reasons = []
                if not st.get("import_ok", True):
                    reasons.append("promptlayer not installed")

                if not st.get("key_present", True):
                    reasons.append("PROMPT_LAYER_API_KEY missing in .env")
                if st.get("client_init_error"):
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
        qdrant_config,
        embedding_config,
        repo_metadata_manager_config,
        ignore_patterns_config,
    )

    tool_manager = ToolManager(repos_config["path_to_repos"])  # TODO , what to do with that path?
    tool_manager.add_tool_pointer("rag_search", manager_qdrant_vector_db.search)
    tool_manager.add_tool_pointer("rag_search_project_readme", manager_qdrant_vector_db.search_project_readme)

    conversation_history = ConversationHistory(conversation_history_config)

    print(f"{Fore.CYAN}Core initialization complete.{Style.RESET_ALL}")
    return llm, tool_manager, conversation_history


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


@execution_profiler
def chat_loop(user_question, llm, tool_manager, conversation_history):
    """Add a user question, run loop, and return final model response."""
    conversation_history.add_user_question(user_question)
    resp = llm_loop(llm, tool_manager, conversation_history)
    conversation_history.add_model_response(resp)
    conversation_history.save()
    return resp


def main():
    llm, tool_manager, conversation_history = build_core()

    while True:
        user_question = input(f"{Fore.BLUE}USER QUESTION:{Style.RESET_ALL} ")
        if not user_question or user_question.lower().strip() in {"exit", "quit"}:
            print(f"{Fore.CYAN}Exiting...{Style.RESET_ALL}")
            break

        resp = chat_loop(user_question, llm, tool_manager, conversation_history)
        print(f"{Fore.MAGENTA}FINAL RESPONSE:{Style.RESET_ALL} {resp}\n")

        execution_profiler.print_info()
        execution_profiler.clean()


if __name__ == "__main__":
    main()
