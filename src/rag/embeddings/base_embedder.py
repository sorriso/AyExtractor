# src/rag/embeddings/base_embedder.py — v1
"""Abstract embeddings interface.

See spec §29.2 for full documentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Unified interface for all embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query (may use different instruction than documents)."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensions."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
