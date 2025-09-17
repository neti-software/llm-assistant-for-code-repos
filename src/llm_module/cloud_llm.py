from typing import Any, Dict, List, Optional, Tuple
import json
import os

from src.llm_module.llm_abc import LLMABC
from openai import OpenAI
from langsmith.wrappers import wrap_openai  # minimal tracing for OpenAI client
from src.utils.profiler import execution_profiler
from src.utils.helper import load_yaml

# Optional PromptLayer integration via .env (PROMPT_LAYER_API_KEY)
try:
    from promptlayer import PromptLayer  # type: ignore
except Exception:
    PromptLayer = None  # type: ignore

def _manual_load_env():
    """Minimal .env loader if python-dotenv is unavailable or not installed."""
    try:
        candidates = [
            os.getcwd(),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        ]
        for root in candidates:
            path = os.path.join(root, ".env")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"\'')
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
    except Exception:
        # ignore any parsing errors
        pass

# Load .env at repo root if available (prefer python-dotenv)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    _manual_load_env()
else:
    # If PROMPT_LAYER_API_KEY not present after load, try manual fallback
    if os.getenv("PROMPT_LAYER_API_KEY") is None and os.getenv("PROMPTLAYER_API_KEY") is None:
        _manual_load_env()


def convert_tools(config_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools = []
    for tool in config_tools:
        properties = {}
        required = []

        for param, spec in tool.get("parameters", {}).items():
            prop = {"type": spec["type"]}
            if "description" in spec:
                prop["description"] = spec["description"]

            # Build schema dynamically from possible_keys
            if spec["type"] == "object" and "possible_keys" in spec:
                prop["properties"] = {}
                for key in spec["possible_keys"]:
                    prop["properties"][key] = {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ]
                    }
                prop["additionalProperties"] = False  # forbid extra keys

            if spec.get("required", False):
                required.append(param)

            properties[param] = prop

        schema = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            }
        }
        if required:
            schema["function"]["parameters"]["required"] = required

        tools.append(schema)
    return tools



def _build_promptlayer_variables(
    prompt_config: Dict[str, Any],
    user_input: str,
    raw_template_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Populate PromptLayer variables, honoring template-provided placeholders."""
    defaults = {
        "history": user_input,
        "input": user_input,
        "query": user_input,
        "topic": user_input,
    }

    # Prompt templates often expect explicit project/name tokens; provide sane fallbacks
    project_fallback = (
        os.getenv("PROMPTLAYER_DEFAULT_PROJECT")
        or prompt_config.get("promptlayer_project")
        or prompt_config.get("project")
        or prompt_config.get("name")
        or "repo-assistant"
    )
    name_fallback = (
        os.getenv("PROMPTLAYER_DEFAULT_NAME")
        or prompt_config.get("promptlayer_name")
        or prompt_config.get("name")
        or "RepoQAAssistant"
    )

    defaults.setdefault("project", project_fallback)
    defaults.setdefault("name", name_fallback)

    # Mirror raw template keys exactly to avoid PromptLayer warnings
    if raw_template_keys:
        for raw_key in raw_template_keys:
            if not isinstance(raw_key, str):
                continue
            cleaned = raw_key.strip().strip('"').strip("'")
            cleaned = cleaned.strip()
            value = defaults.get(cleaned) or defaults.get(cleaned.lower()) or user_input
            defaults[raw_key] = value

    return defaults




def _extract_template_input_keys(blueprint: Any) -> List[str]:
    """Return raw input-variable keys from a PromptLayer blueprint."""
    if isinstance(blueprint, dict):
        template = blueprint.get("prompt_template")
        if isinstance(template, dict):
            raw_inputs = template.get("input_variables")
            if isinstance(raw_inputs, list):
                return [k for k in raw_inputs if isinstance(k, str)]
    return []


def _normalize_promptlayer_messages(raw_messages: Any, user_input: str) -> List[Dict[str, str]]:
    """Normalize PromptLayer message representations into OpenAI chat format."""
    normalized: List[Dict[str, str]] = []
    if not isinstance(raw_messages, list):
        return normalized

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            continue

        content = msg.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)
        if content is None:
            content = ""
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})

    if not any(msg.get("role") == "user" for msg in normalized):
        normalized.append({"role": "user", "content": user_input})

    return [msg for msg in normalized if msg.get("content") is not None]


def _build_messages_from_promptlayer(
    result: Optional[Dict[str, Any]],
    prompt_config: Dict[str, Any],
    user_input: str,
) -> List[Dict[str, str]]:
    """Derive a chat message stack from PromptLayer data with safe fallbacks."""
    result = result or {}
    candidates = []

    raw_messages = result.get("messages")
    if isinstance(raw_messages, list):
        candidates.append(raw_messages)

    blueprint = result.get("prompt_blueprint")
    if isinstance(blueprint, dict):
        template = blueprint.get("prompt_template")
        if isinstance(template, dict):
            template_messages = template.get("messages")
            if isinstance(template_messages, list):
                candidates.append(template_messages)

    for raw in candidates:
        normalized = _normalize_promptlayer_messages(raw, user_input)
        if normalized:
            return normalized

    return [
        {"role": "system", "content": prompt_config.get("role", "")},
        {"role": "user", "content": user_input},
    ]


def _shrink_to_system_plus_user(
    messages: List[Dict[str, str]],
    user_input: str,
    default_system: str,
) -> List[Dict[str, str]]:
    """Reduce a template-derived stack to a canonical system+user pair."""
    system_content = next(
        (msg["content"] for msg in messages if msg.get("role") == "system" and msg.get("content")),
        None,
    )
    if not isinstance(system_content, str) or not system_content.strip():
        system_content = default_system

    return [
        {"role": "system", "content": system_content or ""},
        {"role": "user", "content": user_input},
    ]





class CloudLLM(LLMABC):
    """LLM client for OpenAI-compatible APIs (OpenAI, OpenRouter, etc.) with agent config per call."""

    def __init__(self, config: Dict[str, Any]):
        """
        Top-level config:
        -----------------
        - path_to_api_key : str (YAML with {"key": "..."} )
        - model           : str (e.g., "gpt-4o", "anthropic/claude-3.5-sonnet")
        - provider        : str (optional, "openai" or "openrouter")
        - base_url        : str (optional, API base URL)
        - context_length  : int (optional, max context window)
        """
        super().__init__(config)
        api_key = load_yaml(config["path_to_api_key"])["key"]
        
        # Configure client based on provider
        client_kwargs = {"api_key": api_key}
        if config.get("base_url"):
            client_kwargs["base_url"] = config["base_url"]

        self.client = OpenAI(**client_kwargs)
        # Enable LangSmith tracing for OpenAI calls (minimal, env-driven)
        try:
            self.client = wrap_openai(self.client)
        except Exception:
            # If wrapping fails for any reason, continue without tracing
            pass
        # Make credentials visible to PromptLayer provider wrappers
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if config.get("base_url"):
                os.environ.setdefault("OPENAI_BASE_URL", config["base_url"])
        except Exception:
            pass
        self.model: str = config["model"]
        self.max_completion_tokens = config["max_completion_tokens"]
        self.tool_choice = config["tool_choice"]
        self.context_length = config.get("context_length", 128000)
        self.provider = config.get("provider", "openai")
        self.base_url = config.get("base_url")
        self.prompt_config = load_yaml(config['path_to_prompt_config'])

        # PromptLayer setup (optional)
        self.pl_client = None
        # Use the prompt name from YAML config, fallback to env var, then default
        self.pl_prompt_name = os.getenv("PROMPTLAYER_PROMPT_NAME", self.prompt_config.get("name", "RepoQAAssistant"))
        pl_key = os.getenv("PROMPT_LAYER_API_KEY") or os.getenv("PROMPTLAYER_API_KEY")

        # Diagnostics container for CLI
        self._pl_diag = {
            "import_ok": PromptLayer is not None,
            "key_present": bool(pl_key),
            "client_init": False,
            "client_init_error": None,
            "prompt_fetch_ok": False,
            "prompt_fetch_error": None,
            "prompt_name": self.pl_prompt_name,
            "yaml_prompt_name": self.prompt_config.get("name", "unknown"),
            "template_fetched": False,
        }
        if pl_key and PromptLayer is not None:
            try:
                self.pl_client = PromptLayer(api_key=pl_key)
                self._pl_diag["client_init"] = True
            except Exception as e:
                # Fallback silently if PromptLayer init fails
                self.pl_client = None
                self._pl_diag["client_init_error"] = str(e)

        # Cache: first line of PromptLayer prompt for CLI indicator
        self._pl_prompt_first_line = None
        self._pl_template_input_keys: List[str] = []
        if self.pl_client is not None:
            try:
                blueprint = self.pl_client.templates.get(self.pl_prompt_name, {"input_variables": {}})
                tpl = blueprint.get("prompt_template", "") if isinstance(blueprint, dict) else ""

                self._pl_template_input_keys = _extract_template_input_keys(blueprint) or []

                def _first_line_from_template(t):
                    # If plain string
                    if isinstance(t, str):
                        for ln in t.splitlines():
                            s = ln.strip()
                            if s:
                                return s
                        return ""
                    # If dict with messages list
                    if isinstance(t, dict):
                        msgs = t.get("messages")
                        if isinstance(msgs, list) and msgs:
                            # Prefer system message
                            candidates = msgs
                            system_msgs = [m for m in candidates if isinstance(m, dict) and m.get("role") == "system"]
                            for m in (system_msgs or candidates):
                                content = m.get("content") if isinstance(m, dict) else None
                                if isinstance(content, str):
                                    return _first_line_from_template(content)
                                if isinstance(content, list):
                                    for part in content:
                                        if isinstance(part, dict):
                                            txt = part.get("text") or part.get("content") or ""
                                            if isinstance(txt, str) and txt.strip():
                                                return txt.strip().splitlines()[0].strip()
                                    # If no dict parts, try stringify
                                    return _first_line_from_template(str(content))
                            return ""
                        # Otherwise stringify
                        return _first_line_from_template(str(t))
                    # Fallback stringify
                    return _first_line_from_template(str(t))

                self._pl_prompt_first_line = _first_line_from_template(tpl)
                self._pl_diag["template_fetched"] = True
                self._pl_diag["prompt_fetch_ok"] = True
            except Exception as e:
                # Non-fatal; leave as None
                self._pl_prompt_first_line = None
                self._pl_template_input_keys = []
                self._pl_diag["prompt_fetch_error"] = str(e)
                self._pl_diag["template_fetched"] = False

    def get_promptlayer_prompt_first_line(self) -> str | None:
        """Return first non-empty line of the active PromptLayer prompt, if available."""
        return getattr(self, "_pl_prompt_first_line", None)

    def get_promptlayer_status(self) -> Dict[str, Any]:
        """Return a diagnostic dict describing PromptLayer activation and issues."""
        try:
            return {
                "active": self.pl_client is not None,
                "first_line": self._pl_prompt_first_line,
                **self._pl_diag,
            }
        except Exception:
            # Safe fallback if anything goes wrong
            return {
                "active": False,
                "first_line": None,
                "import_ok": PromptLayer is not None,
                "key_present": bool(os.getenv("PROMPT_LAYER_API_KEY") or os.getenv("PROMPTLAYER_API_KEY")),
                "client_init": False,
                "client_init_error": "unknown",
                "prompt_fetch_ok": False,
                "prompt_fetch_error": "unknown",
                "prompt_name": getattr(self, "pl_prompt_name", None),
            }

    @execution_profiler
    def generate(self, user_input: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate a response using OpenAI Chat Completions API.

        agent_cfg is one entry from your agents.yaml, with:
          - role           : str (system prompt)
          - max_tokens     : int
          - temperature    : float
          - tools          : list (optional)
          - output_schema  : dict (optional JSON schema)
        """
        # Prompt messages are determined by PromptLayer template when enabled

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
        resp: Optional[Any] = None
        result: Optional[Dict[str, Any]] = None



        promptlayer_disabled = os.getenv("DISABLE_PROMPTLAYER", "").lower() in ("1", "true", "yes")
        if self.pl_client is not None and not promptlayer_disabled:
            overrides: Dict[str, Any] = {
                "model": self.model,
                "tools": tools if tools else None,
                "tool_choice": self.tool_choice,
                "response_format": response_format if response_format else None,
                # Use OpenAI SDK-compatible arg when routing via PromptLayer
                "max_tokens": self.max_completion_tokens,
            }
            overrides = {k: v for k, v in overrides.items() if v is not None}

            input_variables = _build_promptlayer_variables(
                self.prompt_config,
                user_input,
                self._pl_template_input_keys,
            )
            force_system_only = os.getenv("PROMPTLAYER_FORCE_SYSTEM_ONLY", "0").lower() in (
                "1",
                "true",
                "yes",
            )

            try:
                result = self.pl_client.run(
                    prompt_name=self.pl_prompt_name,
                    input_variables=input_variables,
                    model_parameter_overrides=overrides,
                    tags=["repo-assistant", "tool-calls"],
                    metadata={"source": "llm-assistant-for-code-repos"},
                )
            except Exception as exc:
                print(f"[DEBUG] PromptLayer run failed for '{self.pl_prompt_name}': {exc}")
                self._pl_diag["last_run_error"] = str(exc)
            else:
                blueprint = result.get("prompt_blueprint") if isinstance(result, dict) else None
                new_keys = _extract_template_input_keys(blueprint)
                if new_keys:
                    self._pl_template_input_keys = new_keys

                messages_from_template = _build_messages_from_promptlayer(
                    result,
                    self.prompt_config,
                    user_input,
                )

                if messages_from_template:
                    if force_system_only:
                        messages_to_use = _shrink_to_system_plus_user(
                            messages_from_template,
                            user_input,
                            self.prompt_config.get("role", ""),
                        )
                        print(
                            "[DEBUG] PromptLayer replayed with system-only messages due to "
                            "PROMPTLAYER_FORCE_SYSTEM_ONLY"
                        )
                    else:
                        messages_to_use = messages_from_template

                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages_to_use,
                        max_completion_tokens=self.max_completion_tokens,
                        tools=tools,
                        tool_choice=self.tool_choice,
                        response_format=response_format,
                    )
                else:
                    resp = result.get("raw_response") if isinstance(result, dict) else None
                    if resp is None or not getattr(resp, "choices", None):
                        print(
                            f"[DEBUG] PromptLayer raw response unavailable for '{self.pl_prompt_name}', "
                            "falling back to prompt_config"
                        )
                        fallback_messages = [
                            {"role": "system", "content": self.prompt_config.get("role", "")},
                            {"role": "user", "content": user_input},
                        ]
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=fallback_messages,
                            max_completion_tokens=self.max_completion_tokens,
                            tools=tools,
                            tool_choice=self.tool_choice,
                            response_format=response_format,
                        )

        else:
            # --- call OpenAI directly with YAML system prompt ---
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": self.prompt_config["role"]},
                {"role": "user", "content": user_input},
            ]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                tools=tools,
                tool_choice=self.tool_choice,
                response_format=response_format,
            )


        if resp is None:
            fallback_messages = _build_messages_from_promptlayer(result, self.prompt_config, user_input)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=fallback_messages,
                max_completion_tokens=self.max_completion_tokens,
                tools=tools,
                tool_choice=self.tool_choice,
                response_format=response_format,
            )


        if not hasattr(resp, "choices") or not resp.choices or len(resp.choices) == 0:
            raise RuntimeError("OpenAI API response has no choices. Response: {}".format(resp))
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

