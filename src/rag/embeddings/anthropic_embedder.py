# src/rag/embeddings/anthropic_embedder.py — v1
"""Anthropic Voyage embedding adapter.

Uses the voyageai SDK for embedding generation.
Models: voyage-3, voyage-3-lite, voyage-code-3.
See spec §29.3.
"""

from __future__ import annotations

import logging

from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class AnthropicEmbedder(BaseEmbedder):
    """Embeddings via Anthropic Voyage API."""

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        dimensions: int = 1024,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self.__client = None

    @property
    def _client(self):
        if self.__client is None:
            try:
                import voyageai
            except ImportError as e:
                raise ImportError(
                    "voyageai package required: pip install voyageai"
                ) from e
            self.__client = voyageai.Client(api_key=self._api_key or "")
        return self.__client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts via Voyage API."""
        result = self._client.embed(
            texts, model=self._model, input_type="document"
        )
        return result.embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query with query-specific instruction."""
        result = self._client.embed(
            [query], model=self._model, input_type="query"
        )
        return result.embeddings[0]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model
