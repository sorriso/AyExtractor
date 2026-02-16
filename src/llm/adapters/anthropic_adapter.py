# src/llm/adapters/anthropic_adapter.py — v2
"""Anthropic Claude adapter implementing BaseLLMClient.

Uses the official anthropic SDK. Supports vision and structured outputs
via constrained decoding (output_config with json_schema).
See spec §27.4 for adapter requirements.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from pydantic import BaseModel

from ayextractor.llm.base_client import BaseLLMClient
from ayextractor.llm.models import ImageInput, LLMResponse, Message

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseLLMClient):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens_default: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens_default = max_tokens_default
        self.__client = None  # Lazy initialization

    @property
    def _client(self):
        """Lazy-init Anthropic client (only on first API call)."""
        if self.__client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                ) from e
            self.__client = anthropic.AsyncAnthropic(api_key=self._api_key or "")
        return self.__client

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Text completion via Anthropic Messages API."""
        kwargs = self._build_kwargs(messages, system, max_tokens, temperature)

        # Structured outputs via constrained decoding
        if response_format is not None:
            schema = response_format.model_json_schema()
            kwargs["metadata"] = kwargs.get("metadata", {})
            # Use tool_use with forced tool for guaranteed JSON schema compliance
            kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": "Return structured data matching the schema",
                    "input_schema": schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        start = time.monotonic()
        response = await self._client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)

        content = self._extract_content(response, response_format is not None)

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            model=response.model,
            provider="anthropic",
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def complete_with_vision(
        self,
        messages: list[Message],
        images: list[ImageInput],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Vision-enabled completion with images."""
        # Build multimodal content blocks
        content_blocks: list[dict[str, Any]] = []
        for img in images:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.media_type,
                        "data": base64.b64encode(img.data).decode("ascii"),
                    },
                }
            )

        # Append text from the last user message
        user_text = ""
        non_user_messages: list[Message] = []
        for m in messages:
            if m.role == "user":
                user_text = m.content
            else:
                non_user_messages.append(m)

        content_blocks.append({"type": "text", "text": user_text})

        api_messages = [self._to_api_message(m) for m in non_user_messages]
        api_messages.append({"role": "user", "content": content_blocks})

        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            params["system"] = system

        start = time.monotonic()
        response = await self._client.messages.create(**params)
        latency_ms = int((time.monotonic() - start) * 1000)

        content = self._extract_content(response, structured=False)

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            model=response.model,
            provider="anthropic",
            latency_ms=latency_ms,
            raw_response=response,
        )

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "anthropic"

    # --- Internal helpers ---

    def _build_kwargs(
        self,
        messages: list[Message],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [self._to_api_message(m) for m in messages],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    @staticmethod
    def _to_api_message(m: Message) -> dict[str, Any]:
        return {"role": m.role, "content": m.content}

    @staticmethod
    def _extract_content(response: Any, structured: bool) -> str:
        """Extract text from Anthropic response content blocks."""
        for block in response.content:
            if structured and getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input)
            if getattr(block, "type", None) == "text":
                return block.text
        return ""
