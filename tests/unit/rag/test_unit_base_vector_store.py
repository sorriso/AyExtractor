# tests/unit/rag/test_base_vector_store.py — v1
"""Tests for rag/vector_store/base_vector_store.py — BaseVectorStore ABC."""

from __future__ import annotations

import pytest

from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore


class TestBaseVectorStore:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]

    def test_has_required_methods(self):
        for attr in [
            "upsert", "query", "delete", "create_collection",
            "collection_exists", "count", "provider_name",
        ]:
            assert hasattr(BaseVectorStore, attr)
