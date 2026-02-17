# tests/unit/graph/test_entity_linker.py — v1
"""Tests for graph/entity_linker.py — cross-document entity resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ayextractor.core.models import EntityNormalization
from ayextractor.graph.entity_linker import LinkResult, link_entities


def _ent(name: str, aliases: list[str] | None = None) -> EntityNormalization:
    return EntityNormalization(
        canonical_name=name, aliases=aliases or [], occurrence_count=1,
    )


class TestLinkEntities:
    @pytest.mark.asyncio
    async def test_empty_incoming(self):
        result = await link_entities(existing=[_ent("A")], incoming=[])
        assert result.stats["total_incoming"] == 0

    @pytest.mark.asyncio
    async def test_empty_existing_all_new(self):
        incoming = [_ent("X"), _ent("Y")]
        result = await link_entities(existing=[], incoming=incoming)
        assert set(result.new_entities) == {"X", "Y"}

    @pytest.mark.asyncio
    async def test_exact_canonical_match(self):
        existing = [_ent("European Union")]
        incoming = [_ent("European Union")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert result.matched["European Union"] == "European Union"
        assert result.new_entities == []

    @pytest.mark.asyncio
    async def test_alias_match(self):
        existing = [_ent("European Union", aliases=["EU"])]
        incoming = [_ent("EU")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert result.matched["EU"] == "European Union"

    @pytest.mark.asyncio
    async def test_incoming_alias_matches_existing(self):
        existing = [_ent("European Union")]
        incoming = [_ent("EU", aliases=["European Union"])]
        result = await link_entities(existing=existing, incoming=incoming)
        assert result.matched["EU"] == "European Union"

    @pytest.mark.asyncio
    async def test_no_match_is_new(self):
        existing = [_ent("A")]
        incoming = [_ent("Z")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert "Z" in result.new_entities
        assert result.matched == {}

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self):
        existing = [_ent("European Union")]
        incoming = [_ent("european union")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert result.matched["european union"] == "European Union"

    @pytest.mark.asyncio
    async def test_embedding_match(self):
        existing = [_ent("European Union")]
        incoming = [_ent("EU Parliament")]  # no exact/alias match
        embedder = AsyncMock()
        embedder.embed_texts = AsyncMock(return_value=[
            [1.0, 0.0],  # European Union
            [0.99, 0.01],  # EU Parliament (similar)
        ])
        result = await link_entities(
            existing=existing, incoming=incoming,
            embedder=embedder, threshold=0.9,
        )
        assert "EU Parliament" in result.matched

    @pytest.mark.asyncio
    async def test_embedding_below_threshold_is_new(self):
        existing = [_ent("A")]
        incoming = [_ent("Z")]
        embedder = AsyncMock()
        embedder.embed_texts = AsyncMock(return_value=[
            [1.0, 0.0],  # A
            [0.0, 1.0],  # Z (orthogonal)
        ])
        result = await link_entities(
            existing=existing, incoming=incoming,
            embedder=embedder, threshold=0.9,
        )
        assert "Z" in result.new_entities

    @pytest.mark.asyncio
    async def test_mixed_match_and_new(self):
        existing = [_ent("A"), _ent("B", aliases=["Beta"])]
        incoming = [_ent("A"), _ent("Beta"), _ent("New")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert result.matched["A"] == "A"
        assert result.matched["Beta"] == "B"
        assert "New" in result.new_entities
