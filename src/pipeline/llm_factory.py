# src/pipeline/llm_factory.py — v1
"""LLM factory — creates per-agent LLM clients using config routing.

Resolves provider:model for each agent via the 3-level cascade
(per-component → per-phase → default → fallback) and instantiates
the appropriate adapter.

See spec §17.3 for routing logic, §27 for adapter interfaces.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from ayextractor.llm.config import LLMAssignment, resolve_llm

if TYPE_CHECKING:
    from ayextractor.config.settings import Settings
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class LLMFactory:
    """Create and cache LLM clients per agent.

    Clients are cached by (provider, model) key so agents sharing
    the same assignment reuse a single client instance.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._clients: dict[str, BaseLLMClient] = {}

    def get_client(self, agent_name: str) -> BaseLLMClient:
        """Get or create LLM client for a specific agent.

        Args:
            agent_name: Agent name for routing resolution.

        Returns:
            BaseLLMClient instance.
        """
        assignment = resolve_llm(agent_name, self._settings)
        cache_key = assignment.key

        if cache_key not in self._clients:
            self._clients[cache_key] = _create_client(
                assignment, self._settings
            )
            logger.info(
                "Created LLM client for '%s': %s (source: %s)",
                agent_name,
                cache_key,
                assignment.source,
            )
        else:
            logger.debug(
                "Reusing cached LLM client for '%s': %s",
                agent_name,
                cache_key,
            )

        return self._clients[cache_key]

    def __call__(self, agent_name: str) -> BaseLLMClient:
        """Callable interface for PipelineRunner.llm_factory."""
        return self.get_client(agent_name)


def _create_client(
    assignment: LLMAssignment,
    settings: Settings,
) -> BaseLLMClient:
    """Instantiate the appropriate LLM adapter.

    Uses lazy imports to avoid loading all adapters at startup.
    """
    provider = assignment.provider.lower()

    if provider == "anthropic":
        from ayextractor.llm.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(
            api_key=settings.anthropic_api_key,
            model=assignment.model,
            max_tokens_default=settings.llm_max_tokens_per_agent,
        )
    elif provider == "openai":
        from ayextractor.llm.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(
            api_key=settings.openai_api_key,
            model=assignment.model,
        )
    elif provider == "google":
        from ayextractor.llm.adapters.google_adapter import GoogleAdapter
        return GoogleAdapter(
            api_key=settings.google_api_key,
            model=assignment.model,
        )
    elif provider == "ollama":
        from ayextractor.llm.adapters.ollama_adapter import OllamaAdapter
        return OllamaAdapter(
            host=settings.ollama_base_url,
            model=assignment.model,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
