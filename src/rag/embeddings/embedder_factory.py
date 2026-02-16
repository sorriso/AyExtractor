# src/rag/embeddings/embedder_factory.py — v1
"""Factory: instantiate embedding provider from configuration.

See spec §29.6.
"""

from __future__ import annotations

import logging

from ayextractor.config.settings import Settings
from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: dict[str, str] = {
    "anthropic": "ayextractor.rag.embeddings.anthropic_embedder.AnthropicEmbedder",
    "openai": "ayextractor.rag.embeddings.openai_embedder.OpenAIEmbedder",
    "sentence_transformers": "ayextractor.rag.embeddings.sentence_tf_embedder.SentenceTransformerEmbedder",
    "ollama": "ayextractor.rag.embeddings.ollama_embedder.OllamaEmbedder",
}


class UnsupportedEmbeddingProviderError(ValueError):
    """Raised when an embedding provider is not registered."""


def create_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Instantiate the configured embedding provider.

    Args:
        settings: Application settings. Uses EMBEDDING_PROVIDER and EMBEDDING_MODEL.

    Returns:
        Configured BaseEmbedder instance.
    """
    if settings is None:
        from ayextractor.rag.embeddings.anthropic_embedder import AnthropicEmbedder
        return AnthropicEmbedder()

    provider = settings.embedding_provider
    if provider not in _PROVIDER_REGISTRY:
        raise UnsupportedEmbeddingProviderError(
            f"Unsupported embedding provider: {provider!r}. "
            f"Available: {', '.join(sorted(_PROVIDER_REGISTRY))}"
        )

    class_path = _PROVIDER_REGISTRY[provider]
    cls = _import_class(class_path)

    kwargs: dict = {"dimensions": settings.embedding_dimensions}

    if provider == "anthropic":
        kwargs["model"] = settings.embedding_model
        kwargs["api_key"] = settings.anthropic_api_key
    elif provider == "openai":
        kwargs["model"] = settings.embedding_model
        kwargs["api_key"] = settings.openai_api_key
    elif provider == "sentence_transformers":
        kwargs["model"] = settings.embedding_st_model
    elif provider == "ollama":
        kwargs["model"] = settings.embedding_ollama_model
        kwargs["base_url"] = settings.ollama_base_url

    logger.debug("Creating embedder: provider=%s", provider)
    return cls(**kwargs)


def register_embedding_provider(name: str, class_path: str) -> None:
    """Register a custom embedding provider."""
    _PROVIDER_REGISTRY[name] = class_path


def _import_class(class_path: str) -> type:
    """Dynamically import a class from its fully qualified path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
