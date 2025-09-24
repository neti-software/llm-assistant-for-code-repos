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

    def add_model_response(self, response: str, metadata: Dict[str, Any] = None):
        """Add a raw LLM response (without tool call)"""
        entry = {
            "iteration": self.iteration,
            "model_response": response
        }
        if metadata:
            entry["metadata"] = metadata
        self.history["history"].append(entry)
        self.iteration += 1

    def add_agent_execution_context(self, context: Dict[str, Any]):
        """Add comprehensive agent execution context including timing, iterations, and results"""
        self.history["agent_execution_context"] = context

    def add_evidence_collection(self, evidence_list: List[Dict[str, Any]]):
        """Add detailed evidence collection results"""
        if "evidence_collection" not in self.history:
            self.history["evidence_collection"] = []
        self.history["evidence_collection"].extend(evidence_list)

    def add_verifier_report(self, report: Dict[str, Any]):
        """Add verifier evaluation report"""
        self.history["verifier_report"] = report

    def add_execution_metrics(self, metrics: Dict[str, Any]):
        """Add execution timing and performance metrics"""
        if "execution_metrics" not in self.history:
            self.history["execution_metrics"] = {}
        self.history["execution_metrics"].update(metrics)

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

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation execution for analysis."""
        summary = {
            "total_iterations": self.iteration,
            "question_count": self.question_counter,
            "has_execution_context": "agent_execution_context" in self.history,
            "has_evidence": "evidence_collection" in self.history and len(self.history.get("evidence_collection", [])) > 0,
            "has_verifier_report": "verifier_report" in self.history,
            "has_metrics": "execution_metrics" in self.history,
        }

        # Add execution metrics if available
        if summary["has_metrics"]:
            summary["metrics"] = self.history["execution_metrics"]

        return summary

    @staticmethod
    def rank_conversation_quality(conversation_file: str) -> Dict[str, Any]:
        """Analyze and rank a conversation file's quality and completeness."""
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            ranking = {
                "filename": conversation_file,
                "score": 0,
                "max_score": 10,
                "criteria": {}
            }

            # Check for complete answer (1 point)
            has_final_answer = False
            for entry in data.get("history", []):
                if "model_response" in entry and entry["model_response"]:
                    has_final_answer = True
                    break
            ranking["criteria"]["has_final_answer"] = has_final_answer
            if has_final_answer:
                ranking["score"] += 1

            # Check for evidence collection (2 points)
            evidence_count = len(data.get("evidence_collection", []))
            ranking["criteria"]["evidence_count"] = evidence_count
            if evidence_count > 0:
                ranking["score"] += 1
            if evidence_count >= 3:
                ranking["score"] += 1

            # Check for execution context (2 points)
            has_execution_context = "agent_execution_context" in data
            ranking["criteria"]["has_execution_context"] = has_execution_context
            if has_execution_context:
                ranking["score"] += 2

            # Check for verifier report (1 point)
            has_verifier = "verifier_report" in data
            ranking["criteria"]["has_verifier_report"] = has_verifier
            if has_verifier:
                ranking["score"] += 1

            # Check for execution metrics (1 point)
            has_metrics = "execution_metrics" in data
            ranking["criteria"]["has_execution_metrics"] = has_metrics
            if has_metrics:
                ranking["score"] += 1

            # Check for tool calls (1 point)
            tool_calls = 0
            for entry in data.get("history", []):
                if "function_call" in entry:
                    tool_calls += 1
            ranking["criteria"]["tool_calls_count"] = tool_calls
            if tool_calls > 0:
                ranking["score"] += 1

            # Check for iterations (1 point)
            iterations = len([e for e in data.get("history", []) if e.get("iteration", 0) > 0])
            ranking["criteria"]["iterations"] = iterations
            if iterations >= 2:
                ranking["score"] += 1

            # Calculate percentage
            ranking["percentage"] = (ranking["score"] / ranking["max_score"]) * 100

            # Generate grade
            if ranking["percentage"] >= 90:
                ranking["grade"] = "A+ (Excellent)"
            elif ranking["percentage"] >= 80:
                ranking["grade"] = "A (Very Good)"
            elif ranking["percentage"] >= 70:
                ranking["grade"] = "B (Good)"
            elif ranking["percentage"] >= 60:
                ranking["grade"] = "C (Satisfactory)"
            elif ranking["percentage"] >= 50:
                ranking["grade"] = "D (Needs Improvement)"
            else:
                ranking["grade"] = "F (Poor)"

            return ranking

        except Exception as e:
            return {
                "filename": conversation_file,
                "error": str(e),
                "score": 0,
                "grade": "Error"
            }
