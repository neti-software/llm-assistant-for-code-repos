import json
import streamlit as st
from typing import Any, Dict, Callable
from src.llm_module.llm_builder import build_llm
from src.vector_db.manager_qdrant_vector_db import ManagerQdrantVectorDb
from src.tools_to_call.tool_manager import ToolManager
from src.conversation.conversation_history import ConversationHistory
from src.utils.helper import load_yaml
from src.utils.profiler import execution_profiler, time_it
from src.langgraph.runner import LangGraphRunner
from src.langgraph.executor import execute_turn as langgraph_executor

st.set_page_config(page_title="LLM Chat", layout="wide")


# ---------- session state ----------
def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list[(role, msg)]
    if "compact" not in st.session_state:
        st.session_state["compact"] = False
    if "chat" not in st.session_state:
        st.session_state["chat"] = None


# ---------- helpers ----------
def _as_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _md_code(lang: str, content: str) -> str:
    return f"```{lang}\n{content}\n```"


def _render_llm_content(resp: Any) -> str:
    # If response is a dict from your LLM tool-calling format, show its 'content'
    if isinstance(resp, dict) and "content" in resp:
        return str(resp["content"])
    return _as_json(resp) if isinstance(resp, (dict, list)) else str(resp)


def _short(obj: Any, limit: int = 4000) -> str:
    text = _as_json(obj) if isinstance(obj, (dict, list)) else str(obj)
    return text if len(text) <= limit else text[:limit] + "\n… [truncated]"


# ---------- app core ----------
class StreamlitChat:
    def __init__(self):
        # load configs
        llm_config = load_yaml("configs/llm_config.yaml")
        embedding_config = load_yaml("configs/embedding_config.yaml")
        reranker_config = load_yaml("configs/reranker_config.yaml")
        qdrant_config = load_yaml("configs/qdrant_config.yaml")
        conversation_history_config = load_yaml("configs/conversation_history_config.yaml")
        repo_metadata_manager_config = load_yaml("configs/json_schema/ast/metadata_schema.json")
        ignore_patterns_config = load_yaml("configs/ignore_patterns_config.yaml")
        repos_config = load_yaml("configs/repos_config.yaml")

        # init core
        self.llm = build_llm(llm_config)
        self.use_langgraph = llm_config.get("use_langgraph_multi_agent", False)
        manager_qdrant_vector_db = ManagerQdrantVectorDb(
            config=qdrant_config,
            embedding_config=embedding_config,
            reranker_config=reranker_config,
            repo_metadata_manager_config=repo_metadata_manager_config,
            ignore_patterns_config=ignore_patterns_config,
        )
        self.tool_manager = ToolManager(repos_config["path_to_repos"], ignore_patterns_config) # TODO , what to do with that path?
        self.tool_manager.add_tool_pointer("rag_search", manager_qdrant_vector_db.search)
        self.tool_manager.add_tool_pointer("rag_search_project_readme", manager_qdrant_vector_db.search_project_readme)
        self.conversation_history = ConversationHistory(conversation_history_config)

        self.runner = LangGraphRunner(
            use_graph=self.use_langgraph,
            legacy_llm_loop=self._legacy_llm_loop,
        )
        if self.use_langgraph:
            self.runner.set_graph_executor(langgraph_executor)

    # unified logger -> terminal + live UI
    def _log(self, live_log: Callable[[str], None], msg: str, role: str = "trace") -> None:
        print(msg)
        live_log(msg, role)

    @time_it
    @execution_profiler
    def _legacy_llm_loop(self, live_log: Callable[[str, str], None]) -> Dict[str, Any]:
        iteration = 0
        while True:
            self._log(live_log, f"**ITERATION {iteration}**", "trace")

            # call LLM
            want_tool, resp = self.llm.generate(self.conversation_history.to_json())

            # show the raw tool-call object nicely
            if isinstance(resp, dict):
                self._log(live_log, "**LLM RESPONSE (raw)**", "trace")
                self._log(live_log, _md_code("json", _as_json(resp)), "trace")
            else:
                self._log(live_log, "**LLM RESPONSE (text)**", "trace")
                self._log(live_log, _md_code("", str(resp)), "trace")

            if not want_tool:
                self._log(live_log, "**FINAL RESPONSE READY**", "trace")
                return resp

            # tool call
            action = resp.get("action", "N/A") if isinstance(resp, dict) else "N/A"
            args = resp.get("args", {}) if isinstance(resp, dict) else {}
            self._log(live_log, f"**Calling tool:** `{action}`", "trace")
            if args:
                self._log(live_log, _md_code("json", _as_json(args)), "trace")

            tool_result = self.tool_manager.call_tool(resp)

            self._log(live_log, "**Tool Result**", "trace")
            self._log(live_log, _md_code("json", _short(tool_result)), "trace")

            # persist for next iteration
                self.conversation_history.add_tool_call(action, args, tool_result)
                iteration += 1

    def handle_question(self, question: str, live_log: Callable[[str, str], None]):
        # record user turn
        self.conversation_history.add_user_question(question)

        if self.use_langgraph:
            live_log("**LangGraph: planning tasks...**", "trace")
            response = self.runner.run_turn(
                self.llm,
                self.tool_manager,
                self.conversation_history,
                live_log=lambda msg: live_log(f"**{msg}**", "trace"),
            )
            message, _ = self._extract_message_and_citations(response)
            live_log("**LangGraph Response**", "trace")
            live_log(_md_code("", message), "trace")
            self.conversation_history.add_model_response(message)
            self.conversation_history.save()
            execution_profiler.print_info()
            execution_profiler.clean()
            return message

        # legacy inner loop
        resp = self._legacy_llm_loop(live_log)
        self.conversation_history.add_model_response(resp)
        self.conversation_history.save()
        execution_profiler.print_info()
        execution_profiler.clean()
        return resp

    def _extract_message_and_citations(self, response: Any) -> tuple[str, list]:
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


# ---------- streamlit UI ----------
ensure_session_state()

with st.sidebar:
    st.header("Settings")
    st.session_state["compact"] = st.checkbox(
        "Compact mode (hide traces)", value=st.session_state["compact"]
    )


    # --- Reset chat ---
    def _reset_chat():
        st.session_state["messages"] = []
        # new StreamlitChat -> new ConversationHistory inside
        st.session_state["chat"] = StreamlitChat()


    if st.button("Reset chat", type="primary", help="Clear conversation and start fresh"):
        _reset_chat()
        st.rerun()

st.title("LLM Chat with Tools")

# init chat once
if st.session_state["chat"] is None:
    st.session_state["chat"] = StreamlitChat()
chat: StreamlitChat = st.session_state["chat"]

# render persistent messages
for role, msg in st.session_state["messages"]:
    if st.session_state["compact"] and role == "trace":
        continue
    with st.chat_message("assistant" if role in {"assistant", "trace"} else "user"):
        st.markdown(msg)

# input
prompt = st.chat_input("Type your message...")

if prompt:
    # push user question immediately
    st.session_state["messages"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # live trace area
    trace_box = st.expander("Live trace", expanded=not st.session_state["compact"])
    live_area = trace_box.empty()
    logs: list[tuple[str, str]] = []  # [(role, msg)]


    def live_log(msg: str, role: str = "trace") -> None:
        # append and render incrementally
        logs.append((role, msg))
        rendered = []
        for r, m in logs:
            if st.session_state["compact"] and r == "trace":
                continue
            rendered.append(m)
        live_area.markdown("\n\n".join(rendered))
        # persist each step so it remains after rerun
        st.session_state["messages"].append((role, msg))


    # run synchronously for true streaming
    with st.spinner("Working..."):
        resp = chat.handle_question(prompt, live_log)

    # show formatted final answer now
    final_text = _render_llm_content(resp)
    with st.chat_message("assistant"):
        st.markdown(final_text)
    st.session_state["messages"].append(("assistant", final_text))
