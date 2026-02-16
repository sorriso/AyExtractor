# src/llm/base_client.py — v1
"""Abstract LLM client interface.

See spec §27.2 for full documentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ayextractor.llm.models import ImageInput, LLMResponse, Message


class BaseLLMClient(ABC):
    """Unified interface for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Text completion."""

    @abstractmethod
    async def complete_with_vision(
        self,
        messages: list[Message],
        images: list[ImageInput],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Vision-enabled completion (images + text)."""

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider/model supports image inputs."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (anthropic, openai, google, ollama)."""
