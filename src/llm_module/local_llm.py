import os
import json
import time
import signal
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

try:
    from llama_cpp import Llama, LlamaGrammar
except Exception as e:
    Llama = None  # type: ignore
    LlamaGrammar = None  # type: ignore

from src.llm_module.llm_abc import LLMABC
from src.utils.profiler import execution_profiler

# ============================================================
#                FastAPI MCP SERVER (INLINE)
# ============================================================
app = FastAPI()
llm_instance: Optional["Llama"] = None
LAST_ERROR: Optional[str] = None  # keep last traceback text for client to read


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    json_schema: Optional[Dict[str, Any]] = None

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global llm_instance
    if llm_instance is None:
        print("[MCP] Lazy loading model ...")
        llm_instance = _load_llm_from_env()

    data = await request.json()

    # MUST accept model
    model = data.get("model") or os.getenv("LLM_MODEL_ID", "local-llm")
    messages = data.get("messages")
    if not messages:
        return JSONResponse({"error": {"message": "messages is required"}}, status_code=400)

    # Flatten messages → prompt
    prompt = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"

    kwargs = dict(
        prompt=prompt,
        max_tokens=data.get("max_tokens", 512),
        temperature=data.get("temperature", 0.7),
        stop=["</s>"],
    )
    out = llm_instance(**kwargs)
    text = out["choices"][0]["text"]

    return JSONResponse({
        "id": "cmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    })


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    global LAST_ERROR
    LAST_ERROR = f"[{request.method} {request.url.path}] {tb}"
    # Return the traceback text so the client can print it (super handy while wiring up)
    return PlainTextResponse(LAST_ERROR, status_code=500)


def _load_llm_from_env() -> "Llama":
    assert Llama is not None, "llama_cpp is not installed"

    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        raise RuntimeError("LLM_MODEL_PATH not set")

    n_ctx = int(os.getenv("LLM_N_CTX", "4096"))
    n_threads = int(os.getenv("LLM_N_THREADS", "8"))
    n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
    verbose = os.getenv("LLM_VERBOSE", "1") == "1"

    print("[MCP] CWD:", os.getcwd())
    print("[MCP] Init llama.cpp with:")
    print(f"      model_path     = {model_path}")
    print(f"      n_ctx          = {n_ctx}")
    print(f"      n_threads      = {n_threads}")
    print(f"      n_gpu_layers   = {n_gpu_layers}")
    print(f"      verbose        = {verbose}")

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    print("[MCP] Model loaded successfully.")
    return llm


@app.get("/models")
def models():
    return {"data": [{"id": os.getenv("LLM_MODEL_ID", "local-llm")}]}


@app.get("/_info")
def info():
    """Small debug endpoint."""
    return {
        "cwd": os.getcwd(),
        "model_path_env": os.getenv("LLM_MODEL_PATH"),
        "n_ctx": os.getenv("LLM_N_CTX"),
        "n_threads": os.getenv("LLM_N_THREADS"),
        "n_gpu_layers": os.getenv("LLM_N_GPU_LAYERS"),
        "verbose": os.getenv("LLM_VERBOSE"),
    }


@app.get("/_last_error", response_class=PlainTextResponse)
def last_error():
    return LAST_ERROR or ""

from fastapi import Request
from fastapi.responses import JSONResponse

@app.post("/completions")
async def completions(request: Request):
    global llm_instance
    if llm_instance is None:
        print("[MCP] Lazy loading model ...")
        llm_instance = _load_llm_from_env()

    data = await request.json()

    # Case 1: Chat-style (OpenAI format)
    if "messages" in data:
        model = data.get("model", os.getenv("LLM_MODEL_ID", "local-llm"))
        messages = data["messages"]

        # Build a prompt string from messages
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        kwargs = dict(
            prompt=prompt,
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.7),
            stop=["</s>"],
        )

        out = llm_instance(**kwargs)
        text = out["choices"][0]["text"]

        return JSONResponse({
            "id": "cmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }]
        })

    # Case 2: Legacy prompt-style (your old tests)
    else:
        kwargs = dict(
            prompt=data["prompt"],
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.7),
            stop=["</s>"],
        )

        if "json_schema" in data and data["json_schema"]:
            assert LlamaGrammar is not None, "llama_cpp grammar support not available"
            schema_str = json.dumps(data["json_schema"])
            grammar = LlamaGrammar.from_json_schema(schema_str)
            kwargs["grammar"] = grammar

        out = llm_instance(**kwargs)
        return JSONResponse(out)


# ============================================================
#                  CLIENT / SPAWNER WRAPPER
# ============================================================
class LocalLLM(LLMABC):
    """
    Local LLM client that:
      - reads endpoint from config
      - spawns FastAPI+uvicorn MCP server (this file) as a subprocess
      - passes model/runtime via env vars
      - kills any prior server on same port
      - waits until /models responds OK
    """

    @execution_profiler
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Resolve to ABSOLUTE model path (critical)
        self.model = str(Path(config["model"]).expanduser().resolve())
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        self.endpoint = config["endpoint"]  # e.g. "http://127.0.0.1:8000"
        self.model_id = config.get("model_id", "local-llm")

        self.n_ctx = int(config.get("n_ctx", 4096))
        self.n_threads = int(config.get("n_threads", 8))
        self.n_gpu_layers = int(config.get("n_gpu_layers", -1))
        self.verbose = bool(config.get("verbose", True))

        self.server_proc: Optional[subprocess.Popen] = None

        parsed = urlparse(self.endpoint)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = str(parsed.port or 8000)

        self._spawn_server()

    def _kill_server_on_port(self, port: str):
        print(f"[LocalLLM] Killing processes on port {port} ...")
        os.system(f"fuser -k {port}/tcp")

    def _wait_for_server(self, timeout_s: float = 30.0):
        deadline = time.time() + timeout_s
        i = 0
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.endpoint}/models", timeout=1.0)
                if r.status_code == 200:
                    print("[LocalLLM] Server ready.")
                    return
            except Exception as e:
                print(f"[LocalLLM] Waiting for server... ({i}) {e}")
            time.sleep(0.5)
            i += 1

        # dump stderr if uvicorn died early
        if self.server_proc and self.server_proc.poll() is not None:
            try:
                _, err = self.server_proc.communicate(timeout=2)
                print("[LocalLLM] ---- uvicorn stderr ----")
                print(err.decode("utf-8", errors="ignore"))
                print("[LocalLLM] ------------------------")
            except Exception:
                pass

        raise RuntimeError("Local MCP server failed to start")

    def _spawn_server(self):
        # If something already answers, kill it first
        try:
            r = requests.get(f"{self.endpoint}/models", timeout=1.0)
            if r.status_code == 200:
                print("[LocalLLM] Existing server found, killing...")
                self._kill_server_on_port(self.port)
        except Exception:
            pass

        # Make sure uvicorn can import "src.llm_module.local_llm:app"
        project_root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

        # Pass runtime via env
        env["LLM_MODEL_PATH"] = self.model  # ABSOLUTE path (fixes 500s)
        env["LLM_MODEL_ID"] = self.model_id
        env["LLM_N_CTX"] = str(self.n_ctx)
        env["LLM_N_THREADS"] = str(self.n_threads)
        env["LLM_N_GPU_LAYERS"] = str(self.n_gpu_layers)
        env["LLM_VERBOSE"] = "1" if self.verbose else "0"

        cmd = [
            "uvicorn",
            "src.llm_module.local_llm:app",
            "--host", self.host,
            "--port", self.port,
            "--log-level", "debug",
        ]
        print("[LocalLLM] Starting MCP server with command:")
        print("           " + " ".join(cmd))
        print(f"[LocalLLM] PYTHONPATH={env['PYTHONPATH']}")
        print(f"[LocalLLM] LLM_MODEL_PATH={env['LLM_MODEL_PATH']}")
        print(f"[LocalLLM] LLM_N_CTX={env['LLM_N_CTX']} "
              f"LLM_N_THREADS={env['LLM_N_THREADS']} "
              f"LLM_N_GPU_LAYERS={env['LLM_N_GPU_LAYERS']}")

        self.server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            env=env,
        )

        self._wait_for_server(timeout_s=30.0)

    @execution_profiler
    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if "json_schema" in kwargs and kwargs["json_schema"]:
            payload["json_schema"] = kwargs["json_schema"]

        url = f"{self.endpoint}/completions"
        print(f"[LocalLLM] POST {url} payload_keys={list(payload.keys())}")
        r = requests.post(url, json=payload)

        if r.status_code >= 400:
            print("[LocalLLM] --- Server returned error body ---")
            print(r.text)
            try:
                # also try to pull the last traceback from server
                dbg = requests.get(f"{self.endpoint}/_last_error", timeout=1.5)
                if dbg.text.strip():
                    print("[LocalLLM] --- Server traceback ---")
                    print(dbg.text)
            except Exception:
                pass
            print("[LocalLLM] ---------------------------------")
            r.raise_for_status()

        data = r.json()
        choice = data["choices"][0]
        if "text" in choice and choice["text"] is not None:
            return choice["text"].strip()
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"].strip()
        return json.dumps(choice).strip()

    def shutdown(self):
        if getattr(self, "server_proc", None):
            print("[LocalLLM] Shutting down server...")
            try:
                os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
            except Exception as e:
                print("[LocalLLM] Kill failed:", e)
            self.server_proc = None

    def __str__(self):
        # Important: must look like a valid provider/model string
        return "ollama/qwen2.5-7b"
