# tests/integration/rag/embeddings/test_int_ollama_embedder.py — v3
"""Integration tests for Ollama embedder via testcontainers.

Coverage target: ollama_embedder.py 55% → 90%+
Marked slow: requires model download.

Source API:
- OllamaEmbedder(model_name, base_url, dimensions).embed_query(text) → list[float]
- OllamaEmbedder.embed_texts(texts) → list[list[float]]
- OllamaEmbedder.provider_name → "ollama"
- OllamaEmbedder.model_name → str
- OllamaEmbedder.dimensions → int

Note: embed_query("") raises RuntimeError("Ollama returned no embeddings for model ...")
because Ollama API returns empty embeddings list for empty input strings.

Changelog:
    v3: Fix test_empty_text — source raises RuntimeError for empty input
        (Ollama API returns no embeddings for empty string). Changed from
        asserting 768 dims to asserting RuntimeError is raised.
    v2: Initial rewrite with correct source API.
"""

from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.ollama, pytest.mark.slow]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class TestOllamaEmbedder:

    @pytest.mark.asyncio
    async def test_single_embedding(self, ollama_embedder):
        emb = await ollama_embedder.embed_query("cybersecurity regulation")
        assert isinstance(emb, list)
        assert len(emb) == 768
        assert all(isinstance(v, float) for v in emb)

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, ollama_embedder):
        texts = ["NIS2 Directive", "ISO 21434", "machine learning"]
        embeddings = await ollama_embedder.embed_texts(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 768

    @pytest.mark.asyncio
    async def test_semantic_similarity(self, ollama_embedder):
        e1 = await ollama_embedder.embed_query("cybersecurity regulation Europe")
        e2 = await ollama_embedder.embed_query("NIS2 network security directive")
        e3 = await ollama_embedder.embed_query("chocolate cake recipe baking")
        assert _cosine(e1, e2) > _cosine(e1, e3)

    @pytest.mark.asyncio
    async def test_deterministic(self, ollama_embedder):
        text = "The EU regulates cybersecurity."
        e1 = await ollama_embedder.embed_query(text)
        e2 = await ollama_embedder.embed_query(text)
        assert _cosine(e1, e2) > 0.999

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, ollama_embedder):
        """Empty string → Ollama returns no embeddings → RuntimeError."""
        with pytest.raises(RuntimeError, match="no embeddings"):
            await ollama_embedder.embed_query("")

    @pytest.mark.asyncio
    async def test_provider_properties(self, ollama_embedder):
        assert ollama_embedder.provider_name == "ollama"
        assert ollama_embedder.model_name == "nomic-embed-text"
        assert ollama_embedder.dimensions == 768