# src/rag/embeddings/ollama_embedder.py — v1
"""Ollama embedding adapter (local inference).

Uses the Ollama REST API for local embedding generation.
Models: nomic-embed-text, mxbai-embed-large, etc.
See spec §29.3.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """Local embeddings via Ollama API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: int = 768,
    ) -> None:
        self._model_name = model
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts via Ollama API (sequential per text)."""
        results: list[list[float]] = []
        for text in texts:
            emb = await self._embed_single(text)
            results.append(emb)
        return results

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query via Ollama API."""
        return await self._embed_single(query)

    async def _embed_single(self, text: str) -> list[float]:
        """Call Ollama embeddings endpoint for a single text."""
        url = f"{self._base_url}/api/embed"
        payload = json.dumps({"model": self._model_name, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        raise RuntimeError(f"Ollama returned no embeddings for model {self._model_name}")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name
