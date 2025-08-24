from typing import List, Dict, Any
import json

from src.llm_module.llm_abc import LLMABC
from openai import OpenAI
from src.utils.profiler import execution_profiler
from src.utils.helper import load_yaml


def convert_tools(config_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools = []
    for tool in config_tools:
        properties = {}
        required = []
        for param, spec in tool.get("parameters", {}).items():
            properties[param] = {"type": spec["type"]}
            if "default" in spec:
                properties[param]["default"] = spec["default"]
            else:
                required.append(param)

        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    return tools


class CloudLLM(LLMABC):
    """LLM client for OpenAI Chat API with agent config per call."""

    def __init__(self, config: Dict[str, Any]):
        """
        Top-level config:
        -----------------
        - path_to_api_key : str (YAML with {"key": "..."} )
        - model           : str (default "gpt-4o")
        """
        super().__init__(config)
        api_key = load_yaml(config["path_to_api_key"])["key"]
        self.client = OpenAI(api_key=api_key)
        self.model: str = config["model"]
        self.max_completion_tokens = config["max_completion_tokens"]
        self.tool_choice = config["tool_choice"]
        self.prompt_config = load_yaml(config['path_to_prompt_config'])

    @execution_profiler
    def generate(self, user_input: str) -> str:
        """
        Generate a response using OpenAI Chat Completions API.

        agent_cfg is one entry from your agents.yaml, with:
          - role           : str (system prompt)
          - max_tokens     : int
          - temperature    : float
          - tools          : list (optional)
          - output_schema  : dict (optional JSON schema)
        """
        # --- messages ---
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.prompt_config["role"]},
            {"role": "user", "content": user_input},
        ]

        # --- response_format ---
        response_format = None
        if self.prompt_config.get("output_schema"):
            schema = self.prompt_config["output_schema"]
            if isinstance(schema, str):  # YAML kept it as a string
                schema = json.loads(schema)  # convert string → dict
            response_format = {
                "type": "json_schema",
                "json_schema": schema,
            }

        tools = convert_tools(self.prompt_config.get("tools"))
        # --- call OpenAI ---
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            tools=tools,
            tool_choice=self.tool_choice,
            response_format=response_format,
        )
        choice = resp.choices[0]
        # --- Branch on finish_reason ---
        if choice.finish_reason == "tool_calls":
            tool_call = choice.message.tool_calls[0]
            return True, {
                "action": tool_call.function.name,
                "args": json.loads(tool_call.function.arguments),
            }

        elif choice.finish_reason == "stop":
            return False, {
                "action": "response",
                "content": choice.message.content.strip() if choice.message.content else "",
            }

        else:
            raise RuntimeError(f"Unexpected finish_reason: {choice.finish_reason}")

        # # inside CloudLLM.generate after resp = self.client.chat.completions.create(...)
        # choice = resp.choices[0]
        # msg = choice.message
        #
        # # 1) Prefer explicit tool_calls array if present (some SDKs provide this)
        # tool_calls = getattr(msg, "tool_calls", None) or msg.get("tool_calls") if isinstance(msg, dict) else None
        #
        # if tool_calls and len(tool_calls) > 0:
        #     tool_call = tool_calls[0]
        #     # arguments might already be a dict or a JSON string depending on SDK
        #     args_raw = getattr(tool_call.function, "arguments", None) or tool_call.function.arguments
        #     try:
        #         args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        #     except Exception:
        #         args = {"raw": args_raw}
        #     return True, {"action": tool_call.function.name, "args": args}
        #
        # # 2) Fallback: some responses use finish_reason == "tool_calls"
        # if getattr(choice, "finish_reason", "") == "tool_calls":
        #     tool_call = msg.tool_calls[0]
        #     args_raw = tool_call.function.arguments
        #     try:
        #         args = json.loads(args_raw)
        #     except Exception:
        #         args = args_raw
        #     return True, {"action": tool_call.function.name, "args": args}
        #
        # # 3) Otherwise plain text response
        # if getattr(choice, "finish_reason", "") in ("stop", "completed", None):
        #     content = (msg.content.strip() if getattr(msg, "content", None) else (
        #         msg.get("content") if isinstance(msg, dict) else ""))
        #     return False, {"action": "response", "content": content}
        #
        # # 4) Anything unexpected
        # raise RuntimeError(f"Unexpected finish_reason / message shape: {choice.finish_reason}")

