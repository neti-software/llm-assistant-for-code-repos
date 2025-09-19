import pytest

from src.conversation.conversation_history import ConversationHistory

from src.langgraph.state_models import ConversationState


def test_conversation_state_defaults():
    state = ConversationState()
    assert state.tasks == []
    assert state.evidence_store == []
    assert state.conversation.user_questions == {}
    assert state.conversation.history == []
    assert state.control_flags.iteration == 0


def test_conversation_history_snapshot_and_apply(tmp_path):
    history = ConversationHistory({"dir_to_save_chat_history": str(tmp_path)})
    history.add_user_question("What is LangGraph?")
    history.add_tool_call("search_files_with_grep", {"pattern": "foo"}, {"matches": 1})

    state = history.to_state_snapshot()

    assert state.conversation.user_questions["user_question1"] == "What is LangGraph?"
    assert state.control_flags.iteration == history.iteration

    state.conversation.history.append(
        {"iteration": state.control_flags.iteration, "model_response": "ok"}
    )
    state.control_flags.iteration += 1

    history.apply_state_delta(state)

    assert history.history == state.conversation.dict()
    assert history.iteration == state.control_flags.iteration
    assert history.question_counter == len(state.conversation.user_questions)
