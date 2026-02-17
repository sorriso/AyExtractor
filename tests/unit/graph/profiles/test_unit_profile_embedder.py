# tests/unit/graph/profiles/test_profile_embedder.py — v1
"""Tests for graph/profiles/profile_embedder.py — embedding entity/relation profiles."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ayextractor.graph.profiles.models import EntityProfile, RelationProfile
from ayextractor.graph.profiles.profile_embedder import (
    embed_entity_profiles,
    embed_profiles,
    embed_relation_profiles,
)


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(
        side_effect=lambda texts: [[0.1 * (i + 1)] * 3 for i in range(len(texts))]
    )
    return embedder


@pytest.fixture
def entity_profiles():
    return [
        EntityProfile(
            canonical_name="EU", entity_type="organization",
            profile_text="The EU is a political and economic union.",
        ),
        EntityProfile(
            canonical_name="AI Act", entity_type="document",
            profile_text="The AI Act regulates AI systems in Europe.",
        ),
    ]


@pytest.fixture
def relation_profiles():
    return [
        RelationProfile(
            subject="EU", predicate="regulates", object="AI systems",
            profile_text="The EU regulates AI systems through the AI Act.",
        ),
    ]


class TestEmbedEntityProfiles:
    @pytest.mark.asyncio
    async def test_sets_embedding_field(self, entity_profiles, mock_embedder):
        result = await embed_entity_profiles(entity_profiles, mock_embedder)
        assert len(result) == 2
        assert result[0].embedding is not None
        assert len(result[0].embedding) == 3
        assert result[1].embedding is not None

    @pytest.mark.asyncio
    async def test_preserves_profile_data(self, entity_profiles, mock_embedder):
        result = await embed_entity_profiles(entity_profiles, mock_embedder)
        assert result[0].canonical_name == "EU"
        assert result[0].profile_text == entity_profiles[0].profile_text
        assert result[1].entity_type == "document"

    @pytest.mark.asyncio
    async def test_does_not_mutate_originals(self, entity_profiles, mock_embedder):
        await embed_entity_profiles(entity_profiles, mock_embedder)
        assert entity_profiles[0].embedding is None
        assert entity_profiles[1].embedding is None

    @pytest.mark.asyncio
    async def test_empty_input(self, mock_embedder):
        result = await embed_entity_profiles([], mock_embedder)
        assert result == []
        mock_embedder.embed_texts.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_embedder_with_profile_texts(self, entity_profiles, mock_embedder):
        await embed_entity_profiles(entity_profiles, mock_embedder)
        mock_embedder.embed_texts.assert_called_once_with([
            "The EU is a political and economic union.",
            "The AI Act regulates AI systems in Europe.",
        ])


class TestEmbedRelationProfiles:
    @pytest.mark.asyncio
    async def test_sets_embedding_field(self, relation_profiles, mock_embedder):
        result = await embed_relation_profiles(relation_profiles, mock_embedder)
        assert len(result) == 1
        assert result[0].embedding is not None

    @pytest.mark.asyncio
    async def test_preserves_data(self, relation_profiles, mock_embedder):
        result = await embed_relation_profiles(relation_profiles, mock_embedder)
        assert result[0].subject == "EU"
        assert result[0].predicate == "regulates"

    @pytest.mark.asyncio
    async def test_empty_input(self, mock_embedder):
        result = await embed_relation_profiles([], mock_embedder)
        assert result == []


class TestEmbedProfiles:
    @pytest.mark.asyncio
    async def test_both_embedded(self, entity_profiles, relation_profiles, mock_embedder):
        entities, relations = await embed_profiles(
            entity_profiles, relation_profiles, mock_embedder,
        )
        assert len(entities) == 2
        assert len(relations) == 1
        assert all(e.embedding is not None for e in entities)
        assert all(r.embedding is not None for r in relations)

    @pytest.mark.asyncio
    async def test_empty_both(self, mock_embedder):
        entities, relations = await embed_profiles([], [], mock_embedder)
        assert entities == []
        assert relations == []
