# src/rag/embeddings/openai_embedder.py — v1
"""OpenAI embedding adapter.

Uses the openai SDK for embedding generation.
Models: text-embedding-3-small, text-embedding-3-large.
See spec §29.3.
"""

from __future__ import annotations

import logging

from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embeddings via OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self.__client = None

    @property
    def _client(self):
        if self.__client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "openai package required: pip install openai"
                ) from e
            self.__client = openai.AsyncOpenAI(api_key=self._api_key or "")
        return self.__client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts via OpenAI API."""
        response = await self._client.embeddings.create(
            input=texts, model=self._model
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query (same endpoint, no special instruction)."""
        response = await self._client.embeddings.create(
            input=[query], model=self._model
        )
        return response.data[0].embedding

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model
