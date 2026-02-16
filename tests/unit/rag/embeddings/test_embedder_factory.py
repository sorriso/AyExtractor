# tests/unit/rag/embeddings/test_embedder_factory.py — v1
"""Tests for rag/embeddings/embedder_factory.py."""

from __future__ import annotations

import pytest

from ayextractor.rag.embeddings.embedder_factory import (
    create_embedder,
    UnsupportedEmbeddingProviderError,
)


class TestCreateEmbedder:
    def test_default_anthropic(self):
        embedder = create_embedder()
        assert embedder.provider_name == "anthropic"

    def test_unsupported_provider(self):
        from ayextractor.config.settings import Settings
        s = Settings(_env_file=None, embedding_provider="nonexistent")
        with pytest.raises(UnsupportedEmbeddingProviderError):
            create_embedder(s)

    def test_anthropic_from_settings(self):
        from ayextractor.config.settings import Settings
        s = Settings(_env_file=None, embedding_provider="anthropic", embedding_model="voyage-3-lite")
        embedder = create_embedder(s)
        assert embedder.provider_name == "anthropic"
        assert embedder.model_name == "voyage-3-lite"

    def test_ollama_from_settings(self):
        from ayextractor.config.settings import Settings
        s = Settings(_env_file=None, embedding_provider="ollama", embedding_ollama_model="mxbai-embed-large")
        embedder = create_embedder(s)
        assert embedder.provider_name == "ollama"
        assert embedder.model_name == "mxbai-embed-large"

    def test_sentence_transformers_from_settings(self):
        from ayextractor.config.settings import Settings
        s = Settings(_env_file=None, embedding_provider="sentence_transformers", embedding_st_model="multilingual-e5-large")
        embedder = create_embedder(s)
        assert embedder.provider_name == "sentence_transformers"
        assert embedder.model_name == "multilingual-e5-large"
