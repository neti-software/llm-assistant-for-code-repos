from src.llm_module.cloud_llm import CloudLLM
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.tools_to_call.tool_manager import ToolManager
from src.conversation.conversation_history import ConversationHistory
from src.utils.helper import load_yaml

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
    llm = CloudLLM(llm_config)
    manager_qdrant_vector_db = ManagerQdrantVectorDb(
        qdrant_config,
        embedding_config,
        repo_metadata_manager_config,
        ignore_patterns_config,
    )

    tool_manager = ToolManager(repos_config["path_to_repos"]) # TODO , what to do with that path?
    tool_manager.add_tool_pointer("rag_search", manager_qdrant_vector_db.search)

    conversation_history = ConversationHistory(conversation_history_config)

    print(f"{Fore.CYAN}Core initialization complete.{Style.RESET_ALL}")
    return llm, tool_manager, conversation_history


def llm_loop(llm: CloudLLM, tool_manager: ToolManager, conversation_history: ConversationHistory) -> dict:
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


if __name__ == "__main__":
    main()
