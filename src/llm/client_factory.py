# src/llm/client_factory.py — v2
"""Factory: instantiate LLM client from provider name.

Called by the orchestrator to create per-agent clients based on
config resolution (see llm/config.py cascade).
See spec §27.5 for details.
"""

from __future__ import annotations

import logging

from ayextractor.config.settings import Settings
from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Registry of provider name → adapter class path (lazy import).
_PROVIDER_REGISTRY: dict[str, str] = {
    "anthropic": "ayextractor.llm.adapters.anthropic_adapter.AnthropicAdapter",
    "openai": "ayextractor.llm.adapters.openai_adapter.OpenAIAdapter",
    "google": "ayextractor.llm.adapters.google_adapter.GoogleAdapter",
    "ollama": "ayextractor.llm.adapters.ollama_adapter.OllamaAdapter",
}


class UnsupportedProviderError(ValueError):
    """Raised when a provider is not registered."""


def create_llm_client(
    provider: str,
    model: str,
    settings: Settings | None = None,
    **kwargs: object,
) -> BaseLLMClient:
    """Instantiate the correct adapter from provider name.

    Args:
        provider: Provider identifier (anthropic, openai, google, ollama).
        model: Model name (e.g. claude-sonnet-4-20250514).
        settings: Application settings (for API keys).
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured BaseLLMClient instance.

    Raises:
        UnsupportedProviderError: If provider is not registered.
    """
    if provider not in _PROVIDER_REGISTRY:
        raise UnsupportedProviderError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Available: {', '.join(sorted(_PROVIDER_REGISTRY))}"
        )

    class_path = _PROVIDER_REGISTRY[provider]
    adapter_cls = _import_class(class_path)

    # Resolve API key from settings
    init_kwargs = dict(kwargs)
    init_kwargs["model"] = model

    if settings is not None:
        if provider == "anthropic":
            init_kwargs.setdefault("api_key", settings.anthropic_api_key)
        elif provider == "openai":
            init_kwargs.setdefault("api_key", settings.openai_api_key)
        elif provider == "google":
            init_kwargs.setdefault("api_key", settings.google_api_key)
        elif provider == "ollama":
            init_kwargs.setdefault("base_url", settings.ollama_base_url)

    logger.debug("Creating LLM client: provider=%s, model=%s", provider, model)
    return adapter_cls(**init_kwargs)


def register_provider(name: str, class_path: str) -> None:
    """Register a custom provider adapter.

    Args:
        name: Provider identifier.
        class_path: Fully qualified class path implementing BaseLLMClient.
    """
    _PROVIDER_REGISTRY[name] = class_path
    logger.info("Registered LLM provider: %s → %s", name, class_path)


def _import_class(class_path: str) -> type:
    """Dynamically import a class from its fully qualified path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
