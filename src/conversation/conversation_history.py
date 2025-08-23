import json
from typing import Any, Dict, List
import datetime
import os


class ConversationHistory:
    def __init__(self, config: Dict, question: str):
        self.dir_to_save_chat_history = config["dir_to_save_chat_history"]
        self.history: Dict[str, Any] = {
            "user_question": question,
            "history": []
        }
        self.iteration = 0

    def get_user_question(self):
        return self.history['user_question']

    def add_rag_results(self, results: List[Dict[str, Any]]):
        """Add initial RAG search results as iteration 0"""
        self.history["history"].append({
            "iteration": self.iteration,
            "rag_results": [
                {"id": i + 1, "snippet": r["value"].strip()}
                for i, r in enumerate(results)
            ]
        })
        self.iteration += 1

    def add_tool_call(self, action: str, args: Dict[str, Any], results: Any):
        """Add a tool call + results for a given iteration"""
        self.history["history"].append({
            "iteration": self.iteration,
            "function_call": {
                "name": action,
                "args": args
            },
            "results": results
        })
        self.iteration += 1

    def add_model_response(self, response: str):
        """Add a raw LLM response (without tool call)"""
        self.history["history"].append({
            "iteration": self.iteration,
            "model_response": response
        })
        self.iteration += 1

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.history, indent=indent)

    def save(self) -> str:
        """
        Save the conversation history to a timestamped file.

        Args:
            directory (str): Folder to save in (default: current dir).
            suffix (str): File extension (default: .json).

        Returns:
            str: Full path to the saved file.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{timestamp}.json"
        full_path = os.path.join(self.dir_to_save_chat_history, file_name)

        os.makedirs(self.dir_to_save_chat_history, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=2))

        print(f"✅ Conversation history saved to {full_path}")
        return full_path
