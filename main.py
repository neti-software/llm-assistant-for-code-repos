from src.llm_module.cloud_llm import CloudLLM
from src.utils.helper import load_yaml, load_json

from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.tools_to_call.tool_manager import ToolManager
from src.conversation.conversation_history import ConversationHistory

from colorama import Fore, Style, init


def chat_loop(llm, manager_qdrant_vector_db, tool_manager, conversation_history):
    question_msg = conversation_history.get_user_question()
    results = manager_qdrant_vector_db.search(question_msg, top_k=3)

    # Step 0: RAG search
    conversation_history.add_rag_results(results)

    # Format for LLM

    # Initialize colorama
    init(autoreset=True)

    iteration = 0
    while True:
        # --- Iteration Header ---
        print(f"\n{Fore.CYAN}--- ITERATION: {iteration} ---{Style.RESET_ALL}\n")

        formated_input = conversation_history.to_json()

        # --- LLM Call and Response ---
        want_tool, resp = llm.generate(formated_input)
        print(f"{Fore.GREEN}LLM RESPONSE:{Style.RESET_ALL} {resp}\n")

        if not want_tool:
            # --- Final Response ---
            print(f"{Fore.MAGENTA}--- FINAL RESPONSE ---{Style.RESET_ALL}")
            break

        # --- Tool Call ---
        print(f"{Fore.YELLOW}Calling tool: {resp.get('action', 'N/A')}")
        tool_result = tool_manager.call_tool(results, resp)
        print(f"{Fore.YELLOW}Tool Result:{Style.RESET_ALL} {tool_result}\n")

        conversation_history.add_tool_call(resp["action"], resp.get("args", {}), tool_result)

        iteration += 1

    # Final model response
    conversation_history.add_model_response(resp)
    conversation_history.save()


# ----------------------
# Main runner
# ----------------------
llm_config = load_yaml("/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/configs/llm_config.yaml")
embedding_config = load_yaml("/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/configs/embedding_config.yaml")
qdrant_config = load_yaml("/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/configs/qdrant_config.yaml")
conversation_history_config = load_yaml(
    "/home/dawid/Desktop/Neti/llm-assistant-for-code-repos/configs/conversation_history.yaml")

llm = CloudLLM(llm_config)
tool_manager = ToolManager()

manager_qdrant_vector_db = ManagerQdrantVectorDb(config=qdrant_config, embedding_config=embedding_config)
# manager_qdrant_vector_db.create_vector_db_from_dir("DATA_TO_TEST")  # For first time only. Use 'docker run -p 6333:6333 qdrant/qdrant'

question_to_test = load_json('json_question_to_test.json')["2"]
conversation_history = ConversationHistory(conversation_history_config, question_to_test)
chat_loop(llm, manager_qdrant_vector_db, tool_manager, conversation_history)
