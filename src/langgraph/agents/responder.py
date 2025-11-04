"""Responder agent that generates the final answer using LLM with structured context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..state_models import ConversationState
from ..debug_logger import debug_log
from ...llm_module.llm_builder import build_llm


@dataclass
class FinalResponse:
    message: str
    citations: List[str]


class ResponderAgent:
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.llm_config = llm_config
        self.llm = llm
        if self.llm is None and llm_config:
            self.llm = build_llm(llm_config)

    def respond(self, state: ConversationState) -> FinalResponse:
        # Get the original question to provide context
        question = self._get_latest_question(state)

        if not self.llm:
            raise RuntimeError("ResponderAgent requires an LLM configuration. None was provided.")

        # If we have evidence, synthesize a focused response
        if state.evidence_store:
            response_text = self._synthesize_response(state, question)
        else:
            response_text = f"No evidence found to answer: {question}"

        # Collect citations
        citations = []
        if state.control_flags.last_verifier_report:
            citations.extend(state.control_flags.last_verifier_report.get("citations", []))
        else:
            for item in state.evidence_store:
                citations.extend(item.citations)

        return FinalResponse(message=response_text, citations=list(dict.fromkeys(citations)))


    def _get_latest_question(self, state: ConversationState) -> str:
        """Extract the latest user question from the state."""
        if state.conversation.history:
            for entry in reversed(state.conversation.history):
                for key, value in entry.items():
                    if key.startswith("user_question"):
                        return value
        if state.conversation.user_questions:
            last_key = sorted(state.conversation.user_questions.keys())[-1]
            return state.conversation.user_questions[last_key]
        return "Unknown question"

    def _synthesize_response(self, state: ConversationState, question: str) -> str:
        """Generate final answer using LLM with structured context from all agents."""
        assert self.llm is not None, "LLM must be configured"
        
        if not state.evidence_store:
            return f"Insufficient evidence to answer: {question}"

        # Group evidence by agent type for structured context
        agent_context = self._group_evidence_by_agent(state)

        # Log the complete context for debugging
        self._log_context_for_debugging(question, agent_context)

        # Generate final answer using LLM
        return self._generate_llm_response(question, agent_context)

    def _group_evidence_by_agent(self, state: ConversationState) -> Dict[str, List[str]]:
        """Group evidence by agent type based on metadata."""
        agent_context = {
            "repo_intelligence": [],
            "code_inspector": [],
            "other": []
        }

        for item in state.evidence_store:
            agent_type = "other"
            if item.metadata and "tool" in item.metadata:
                tool_name = item.metadata["tool"]
                if "repo" in tool_name.lower():
                    agent_type = "repo_intelligence"
                elif "code" in tool_name.lower():
                    agent_type = "code_inspector"

            if item.full_content:
                agent_context[agent_type].append(item.full_content)

        return agent_context

    def _log_context_for_debugging(self, question: str, agent_context: Dict[str, List[str]]) -> None:
        """Log the complete context for debugging purposes."""
        debug_log("ResponderAgent", f"=== FINAL ANSWER CONTEXT ===")
        debug_log("ResponderAgent", f"User Question: {question}")
        debug_log("ResponderAgent", f"Total evidence items: {sum(len(answers) for answers in agent_context.values())}")

        for agent_type, answers in agent_context.items():
            if answers:
                debug_log("ResponderAgent", f"**{agent_type.upper()} AGENT**")
                debug_log("ResponderAgent", f"Answers collected: {len(answers)}")

                for i, answer in enumerate(answers, 1):
                    # Log first 500 chars of each answer
                    preview = answer[:500] + "..." if len(answer) > 500 else answer
                    debug_log("ResponderAgent", f"Answer {i} preview: {preview}")

                debug_log("ResponderAgent", f"**END {agent_type.upper()} AGENT**")
                debug_log("ResponderAgent", "")

        debug_log("ResponderAgent", f"=== END CONTEXT ===")

    def _generate_llm_response(self, question: str, agent_context: Dict[str, List[str]]) -> str:
        """Generate final answer using LLM with structured context."""
        assert self.llm is not None, "LLM must be configured for _generate_llm_response"
        # Build structured prompt for LLM
        prompt_parts = [
            "You are an expert assistant helping to answer a user's question based on information gathered from different specialized agents.",
            "",
            f"USER QUESTION: {question}",
            "",
            "Below is the information gathered from different agents:",
            ""
        ]

        # Add context from each agent type
        for agent_type, answers in agent_context.items():
            if answers:
                prompt_parts.append(f"**{agent_type.replace('_', ' ').upper()} AGENT**")
                for i, answer in enumerate(answers, 1):
                    prompt_parts.append(f"Answer {i} from {agent_type} agent:")
                    prompt_parts.append(answer)
                    prompt_parts.append("")
                prompt_parts.append("")

        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Analyze all the information provided by the different agents",
            "2. Synthesize a comprehensive answer to the user's question",
            "3. Include relevant details from each agent's response",
            "4. Make the answer clear, well-structured, and easy to understand",
            "5. If there are conflicting answers, acknowledge the differences",
            "",
            "Provide a comprehensive answer based on all the agent responses above."
        ])

        prompt = "\n".join(prompt_parts)

        # Save the full prompt in debug logs
        debug_log("ResponderAgent", f"=== LLM PROMPT ===")
        debug_log("ResponderAgent", prompt)
        debug_log("ResponderAgent", f"=== END PROMPT ===")

        # Make LLM call
        want_tool, response = self.llm.generate(prompt)

        if want_tool:
            # If LLM wants to call a tool, return a message indicating this
            return f"LLM requested tool call: {response.get('action', 'unknown')}"

        # Extract the response content
        if isinstance(response, dict) and "content" in response:
            final_answer = response["content"].strip()
        else:
            final_answer = str(response).strip()

        # Save the final answer in debug logs
        debug_log("ResponderAgent", f"=== LLM RESPONSE ===")
        debug_log("ResponderAgent", final_answer)
        debug_log("ResponderAgent", f"=== END RESPONSE ===")

        return final_answer

    def persist_response(self, conversation_history, response: FinalResponse) -> None:
        conversation_history.add_model_response(response.message)
