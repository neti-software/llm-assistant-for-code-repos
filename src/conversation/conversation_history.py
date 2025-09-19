import json
from typing import Any, Dict, List
import datetime
import os
import copy

from src.langgraph.state_models import ConversationState, conversation_from_raw, ControlFlags


class ConversationHistory:
    def __init__(self, config: Dict):
        self.dir_to_save_chat_history = config["dir_to_save_chat_history"]
        self.history: Dict[str, Any] = {
            "user_questions": {},  # dynamic keys user_question1, user_question2, ...
            "history": []
        }
        self.iteration = 0
        self.question_counter = 0  # track how many user questions so far

    def get_current_question(self) -> str:
        """Return the latest user question."""
        if self.question_counter == 0:
            return None
        return self.history["user_questions"].get(f"user_question{self.question_counter}")

    def add_user_question(self, question: str):
        """Append a new user question both to 'user_questions' dict and to history list."""
        self.question_counter += 1
        key = f"user_question{self.question_counter}"

        # store in top-level dict
        self.history["user_questions"][key] = question

        # store in chronological history
        self.history["history"].append({
            "iteration": self.iteration,
            key: question
        })

        self.iteration += 1

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
        return json.dumps(self.history, indent=indent, default=str)

    # LangGraph adapters -------------------------------------------------

    def to_state_snapshot(self) -> ConversationState:
        """Return a structured ConversationState snapshot.

        The snapshot keeps the raw history data intact while exposing
        iteration counters through ControlFlags for orchestrator bookkeeping.
        """

        buffer = conversation_from_raw(self.history)
        state = ConversationState(conversation=buffer)
        state.control_flags = ControlFlags(iteration=self.iteration)
        return state

    def apply_state_delta(self, state: ConversationState) -> None:
        """Apply updates from a ConversationState back into the history."""

        snapshot = state.conversation.dict()
        self.history = copy.deepcopy(snapshot)
        self.iteration = state.control_flags.iteration
        self.question_counter = len(self.history.get("user_questions", {}))

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
