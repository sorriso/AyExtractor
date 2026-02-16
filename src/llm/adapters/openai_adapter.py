# src/llm/adapters/openai_adapter.py — v1
"""OpenAI GPT adapter implementing BaseLLMClient.

Uses the official openai SDK. Supports vision and structured outputs.
See spec §27.4.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel

from ayextractor.llm.base_client import BaseLLMClient
from ayextractor.llm.models import ImageInput, LLMResponse, Message


class OpenAIAdapter(BaseLLMClient):
    """OpenAI GPT adapter."""

    def __init__(self, model: str = "gpt-4o", api_key: str = "", **kwargs: Any):
        self._model = model
        self._api_key = api_key

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)
        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for m in messages:
            oai_messages.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            schema = response_format.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_format.__name__, "schema": schema},
            }

        t0 = time.monotonic()
        resp = await client.chat.completions.create(**kwargs)
        latency = int((time.monotonic() - t0) * 1000)

        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self._model,
            provider="openai",
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
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)
        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        # Build multimodal content
        content_parts: list[dict[str, Any]] = []
        for m in messages:
            content_parts.append({"type": "text", "text": m.content})
        for img in images:
            b64 = base64.b64encode(img.data).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.media_type};base64,{b64}"},
            })
        oai_messages.append({"role": "user", "content": content_parts})

        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=self._model, messages=oai_messages, max_tokens=max_tokens,
        )
        latency = int((time.monotonic() - t0) * 1000)

        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self._model,
            provider="openai",
            latency_ms=latency,
            raw_response=resp,
        )

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "openai"
