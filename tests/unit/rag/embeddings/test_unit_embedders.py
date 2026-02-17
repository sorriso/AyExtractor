# tests/unit/rag/embeddings/test_embedders.py — v1
"""Tests for all embedding adapters — properties and import error handling."""

from __future__ import annotations

import sys

import pytest


class TestAnthropicEmbedder:
    def test_properties(self):
        from ayextractor.rag.embeddings.anthropic_embedder import AnthropicEmbedder
        e = AnthropicEmbedder(model="voyage-3", dimensions=1024)
        assert e.provider_name == "anthropic"
        assert e.model_name == "voyage-3"
        assert e.dimensions == 1024

    def test_import_error(self):
        mod = sys.modules.get("voyageai")
        sys.modules["voyageai"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.embeddings.anthropic_embedder import AnthropicEmbedder
            e = AnthropicEmbedder()
            with pytest.raises(ImportError, match="voyageai"):
                import asyncio
                asyncio.get_event_loop().run_until_complete(e.embed_query("test"))
        finally:
            if mod is not None:
                sys.modules["voyageai"] = mod
            else:
                sys.modules.pop("voyageai", None)


class TestOpenAIEmbedder:
    def test_properties(self):
        from ayextractor.rag.embeddings.openai_embedder import OpenAIEmbedder
        e = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536)
        assert e.provider_name == "openai"
        assert e.model_name == "text-embedding-3-small"
        assert e.dimensions == 1536

    def test_import_error(self):
        mod = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.embeddings.openai_embedder import OpenAIEmbedder
            e = OpenAIEmbedder()
            with pytest.raises(ImportError, match="openai"):
                import asyncio
                asyncio.get_event_loop().run_until_complete(e.embed_query("test"))
        finally:
            if mod is not None:
                sys.modules["openai"] = mod
            else:
                sys.modules.pop("openai", None)


class TestSentenceTransformerEmbedder:
    def test_properties(self):
        from ayextractor.rag.embeddings.sentence_tf_embedder import SentenceTransformerEmbedder
        e = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2", dimensions=384)
        assert e.provider_name == "sentence_transformers"
        assert e.model_name == "all-MiniLM-L6-v2"
        assert e.dimensions == 384

    def test_import_error(self):
        mod = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.embeddings.sentence_tf_embedder import SentenceTransformerEmbedder
            e = SentenceTransformerEmbedder()
            with pytest.raises(ImportError, match="sentence-transformers"):
                import asyncio
                asyncio.get_event_loop().run_until_complete(e.embed_query("test"))
        finally:
            if mod is not None:
                sys.modules["sentence_transformers"] = mod
            else:
                sys.modules.pop("sentence_transformers", None)


class TestOllamaEmbedder:
    def test_properties(self):
        from ayextractor.rag.embeddings.ollama_embedder import OllamaEmbedder
        e = OllamaEmbedder(model="nomic-embed-text", dimensions=768)
        assert e.provider_name == "ollama"
        assert e.model_name == "nomic-embed-text"
        assert e.dimensions == 768
