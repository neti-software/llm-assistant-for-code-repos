from src.llm_module.cloud_llm import CloudLLM
from src.utils.helper import load_yaml, load_json

from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.tools_to_call.tool_manager import ToolManager
from src.conversation.conversation_history import ConversationHistory
from src.tools_to_call.fetch_file_from_patch import fetch_file_from_patch


from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def foo_minimalize_rag_results(res, full_mode: bool = False):
    formated_results = []
    for r in res:
        formated_dcit= {}
        formated_dcit['path_to_file'] = r['metadata']['project'] + "/" + r['metadata']['path']
        if full_mode:
            formated_dcit['value']  = fetch_file_from_patch(formated_dcit['path_to_file'])
        else:
            formated_dcit['value'] = r['value']

        formated_dcit['score'] = round(r['score'],3)


        formated_results.append(formated_dcit)
    return formated_results


def chat_loop(llm, tool_manager, conversation_history):
    print(f"{Fore.BLUE}USER QUESTION:{Style.RESET_ALL} {conversation_history.history['user_question']}\n")

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
        tool_result = tool_manager.call_tool(resp)
        tool_result = foo_minimalize_rag_results(tool_result)
        print(f"{Fore.YELLOW}Tool Result:{Style.RESET_ALL} {tool_result}\n")

        conversation_history.add_tool_call(resp["action"], resp.get("args", {}), tool_result)

        iteration += 1

    # Final model response
    conversation_history.add_model_response(resp)
    conversation_history.save()


# ----------------------
# Main runner
# ----------------------
llm_config = load_yaml("configs/llm_config.yaml")
embedding_config = load_yaml("configs/embedding_config.yaml")
qdrant_config = load_yaml("configs/qdrant_config.yaml")
conversation_history_config = load_yaml("configs/conversation_history_config.yaml")
repo_metadata_manager_config = load_yaml("configs/json_schema/ast/metadata_schema.json")

llm = CloudLLM(llm_config)

manager_qdrant_vector_db = ManagerQdrantVectorDb(config=qdrant_config,
                                                 embedding_config=embedding_config,
                                                 repo_metadata_manager_config=repo_metadata_manager_config)

# For first time only. Use 'docker run -p 6333:6333 qdrant/qdrant'
# manager_qdrant_vector_db.delete_db()
# manager_qdrant_vector_db.create_vector_db_from_dir("DATA_TO_TEST")

question_to_test = load_json('json_question_to_test.json')["12"]
conversation_history = ConversationHistory(conversation_history_config, question_to_test)

tool_manager = ToolManager()
tool_manager.add_tool_pointer("rag_search", manager_qdrant_vector_db.search)
xx = manager_qdrant_vector_db.search(question_to_test)
# y = foo_minimalize_rag_results(xx)

chat_loop(llm, tool_manager, conversation_history)

