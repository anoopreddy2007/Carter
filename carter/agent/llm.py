"""
Local LLM client interface for Carter.
Supports:
1. llama.cpp server (local HTTP API bound to 127.0.0.1, FR-02)
2. llama-cpp-python in-process GGUF runner (direct local execution)
3. MockLLMClient for testing and zero-dependency offline unit tests.
Never calls remote services or downloads external models at runtime (FR-01, FR-05).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import urllib.request
import urllib.error

from carter.utils.config import get
from carter.utils.logging import get_logger

logger = get_logger("agent.llm")


class BaseLLMClient:
    """Abstract base interface for local LLM inference backends."""

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """
    Predictable mock client for unit testing agent reasoning, multi-step loops, and fallback parsing.
    """

    def __init__(self, canned_responses: list[str] | None = None) -> None:
        self.responses = list(canned_responses) if canned_responses else []
        self.call_history: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.call_history.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "I have completed the task."


class LlamaServerClient(BaseLLMClient):
    """
    Communicates with a local llama-server instance via HTTP.
    Strictly checks for 127.0.0.1 binding to ensure local air-gapped isolation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: int = 60,
    ) -> None:
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError(
                f"Air-gap safety violation: Carter LLM client can only bind to 127.0.0.1, got {host}"
            )
        self.endpoint = f"http://{host}:{port}/v1/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            logger.error("Failed to connect to local llama-server at %s: %s", self.endpoint, e)
            raise ConnectionError(
                f"Local LLM server unreachable at {self.endpoint}. "
                "Ensure llama-server is running on 127.0.0.1:8080."
            ) from e


class LlamaCppPythonClient(BaseLLMClient):
    """
    Direct in-process GGUF inference via llama-cpp-python.
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF model file not found: {path}")

        try:
            from llama_cpp import Llama  # pyrefly: ignore[missing-import]
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Install the pre-built wheel from your local wheelhouse."
            ) from e

        logger.info("Initializing in-process Llama engine from %s (GPU layers=%d)", path, n_gpu_layers)
        self.llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        temp = kwargs.get("temperature", self.temperature)
        tokens = kwargs.get("max_tokens", self.max_tokens)
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )
        return response["choices"][0]["message"]["content"]


def create_llm_client(cfg: dict[str, Any]) -> BaseLLMClient:
    """Factory to create the appropriate offline LLM client based on carter.yaml config."""
    backend = get(cfg, "llm", "backend", default="server")
    if backend == "mock":
        return MockLLMClient()

    if backend == "in_process":
        model_path = get(cfg, "llm", "model_path", default="")
        n_ctx = get(cfg, "llm", "context_length", default=2048)
        n_gpu_layers = get(cfg, "llm", "n_gpu_layers", default=0)
        return LlamaCppPythonClient(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )

    # Default: local HTTP server
    host = get(cfg, "llm", "host", default="127.0.0.1")
    port = get(cfg, "llm", "port", default=8080)
    temp = get(cfg, "llm", "temperature", default=0.2)
    max_tokens = get(cfg, "llm", "max_tokens", default=512)
    return LlamaServerClient(host=host, port=port, temperature=temp, max_tokens=max_tokens)
