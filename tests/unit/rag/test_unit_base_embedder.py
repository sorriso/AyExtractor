# tests/unit/rag/test_base_embedder.py — v1
"""Tests for rag/embeddings/base_embedder.py — BaseEmbedder ABC."""

from __future__ import annotations

import pytest

from ayextractor.rag.embeddings.base_embedder import BaseEmbedder


class TestBaseEmbedder:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore[abstract]

    def test_has_required_methods(self):
        for attr in ["embed_texts", "embed_query", "dimensions", "provider_name", "model_name"]:
            assert hasattr(BaseEmbedder, attr)
