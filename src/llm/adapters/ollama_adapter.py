# src/llm/adapters/ollama_adapter.py — v1
"""Ollama local LLM adapter implementing BaseLLMClient.

Uses the ollama Python SDK. Vision support is model-dependent.
See spec §27.4.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from ayextractor.llm.base_client import BaseLLMClient
from ayextractor.llm.models import ImageInput, LLMResponse, Message

# Models known to support vision
_VISION_MODELS = {"llava", "bakllava", "llava-llama3", "moondream"}


class OllamaAdapter(BaseLLMClient):
    """Ollama local inference adapter."""

    def __init__(
        self, model: str = "llama3", host: str = "http://localhost:11434", **kwargs: Any,
    ):
        self._model = model
        self._host = host

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        import ollama

        client = ollama.AsyncClient(host=self._host)
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})

        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        kwargs: dict[str, Any] = {"model": self._model, "messages": msgs, "options": options}
        if response_format is not None:
            kwargs["format"] = "json"

        t0 = time.monotonic()
        resp = await client.chat(**kwargs)
        latency = int((time.monotonic() - t0) * 1000)

        return LLMResponse(
            content=resp["message"]["content"],
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            model=self._model,
            provider="ollama",
            latency_ms=latency,
            raw_response=resp,
        )

    async def complete_with_vision(
        self,
        messages: list[Message],
        images: list[ImageInput],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        import base64
        import ollama

        client = ollama.AsyncClient(host=self._host)
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})

        # Combine text + images into single user message
        text = " ".join(m.content for m in messages)
        img_data = [base64.b64encode(img.data).decode() for img in images]
        msgs.append({"role": "user", "content": text, "images": img_data})

        t0 = time.monotonic()
        resp = await client.chat(
            model=self._model, messages=msgs,
            options={"num_predict": max_tokens},
        )
        latency = int((time.monotonic() - t0) * 1000)

        return LLMResponse(
            content=resp["message"]["content"],
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            model=self._model,
            provider="ollama",
            latency_ms=latency,
            raw_response=resp,
        )

    @property
    def supports_vision(self) -> bool:
        return any(v in self._model.lower() for v in _VISION_MODELS)

    @property
    def provider_name(self) -> str:
        return "ollama"
