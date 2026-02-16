# src/llm/adapters/google_adapter.py — v1
"""Google Gemini adapter implementing BaseLLMClient.

Uses google-generativeai SDK. Supports vision via Gemini Vision.
See spec §27.4.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from ayextractor.llm.base_client import BaseLLMClient
from ayextractor.llm.models import ImageInput, LLMResponse, Message


class GoogleAdapter(BaseLLMClient):
    """Google Gemini adapter."""

    def __init__(self, model: str = "gemini-pro", api_key: str = "", **kwargs: Any):
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
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            self._model,
            system_instruction=system,
        )

        gen_config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = response_format.model_json_schema()

        # Convert messages to Gemini format
        contents = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        t0 = time.monotonic()
        resp = await model.generate_content_async(
            contents, generation_config=gen_config,
        )
        latency = int((time.monotonic() - t0) * 1000)

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            content=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            model=self._model,
            provider="google",
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
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model, system_instruction=system)

        parts: list[dict[str, Any]] = []
        for m in messages:
            parts.append({"text": m.content})
        for img in images:
            parts.append({"inline_data": {"mime_type": img.media_type, "data": img.data}})

        t0 = time.monotonic()
        resp = await model.generate_content_async(
            parts, generation_config={"max_output_tokens": max_tokens},
        )
        latency = int((time.monotonic() - t0) * 1000)

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            content=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            model=self._model,
            provider="google",
            latency_ms=latency,
            raw_response=resp,
        )

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "google"
