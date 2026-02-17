# tests/unit/graph/test_entity_normalizer.py — v2
"""Tests for graph/entity_normalizer.py — entity clustering and normalization."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from ayextractor.core.models import QualifiedTriplet
from ayextractor.graph.entity_normalizer import (
    build_entity_mapping,
    cluster_by_similarity,
    extract_unique_entities,
    normalize_entities,
)


def _make_triplet(subject: str, predicate: str, obj: str, chunk_id: str = "c1"):
    return QualifiedTriplet(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_chunk_id=chunk_id,
        confidence=0.8,
        context_sentence="test",
    )


class TestExtractUniqueEntities:
    def test_extracts_subjects_and_objects(self):
        triplets = [
            _make_triplet("A", "rel", "B"),
            _make_triplet("C", "rel", "A"),
        ]
        entities = extract_unique_entities(triplets)
        assert set(entities.keys()) == {"A", "B", "C"}

    def test_tracks_chunk_ids(self):
        triplets = [
            _make_triplet("A", "rel", "B", "c1"),
            _make_triplet("A", "rel", "C", "c2"),
        ]
        entities = extract_unique_entities(triplets)
        assert "c1" in entities["A"]
        assert "c2" in entities["A"]

    def test_empty_input(self):
        assert extract_unique_entities([]) == {}

    def test_strips_whitespace(self):
        triplets = [_make_triplet("  A  ", "rel", "B")]
        entities = extract_unique_entities(triplets)
        assert "A" in entities


class TestClusterBySimilarity:
    def test_identical_vectors_cluster_together(self):
        names = ["A", "B"]
        emb = np.array([[1.0, 0.0], [1.0, 0.0]])
        clusters = cluster_by_similarity(names, emb, threshold=0.9)
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1}

    def test_orthogonal_vectors_separate_clusters(self):
        names = ["A", "B"]
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        clusters = cluster_by_similarity(names, emb, threshold=0.5)
        assert len(clusters) == 2

    def test_single_entity(self):
        clusters = cluster_by_similarity(["A"], np.array([[1.0, 0.0]]))
        assert clusters == [[0]]

    def test_empty_input(self):
        assert cluster_by_similarity([], np.empty((0, 0))) == []

    def test_threshold_controls_grouping(self):
        names = ["A", "B"]
        emb = np.array([[1.0, 0.1], [1.0, 0.2]])
        # High threshold: should NOT cluster
        clusters_strict = cluster_by_similarity(names, emb, threshold=0.999)
        # Low threshold: should cluster
        clusters_loose = cluster_by_similarity(names, emb, threshold=0.5)
        assert len(clusters_strict) >= len(clusters_loose)


class TestNormalizeEntities:
    @pytest.mark.asyncio
    async def test_without_embedder_each_entity_canonical(self):
        triplets = [
            _make_triplet("EU", "rel", "AI"),
            _make_triplet("European Union", "rel", "AI Act"),
        ]
        result = await normalize_entities(triplets, embedder=None)
        names = {r.canonical_name for r in result}
        assert "EU" in names
        assert "European Union" in names
        assert len(result) == 4  # EU, AI, European Union, AI Act

    @pytest.mark.asyncio
    async def test_with_embedder_clusters_similar(self):
        triplets = [
            _make_triplet("EU", "rel", "AI"),
            _make_triplet("European Union", "rel", "B"),
        ]
        embedder = AsyncMock()
        # EU and European Union get similar embeddings
        embedder.embed_texts = AsyncMock(return_value=[
            [0.9, 0.1],  # AI
            [0.1, 0.9],  # B
            [1.0, 0.0],  # EU
            [0.99, 0.01],  # European Union (similar to EU)
        ])
        result = await normalize_entities(triplets, embedder=embedder, threshold=0.9)
        # EU and European Union should be clustered
        canonical_names = {r.canonical_name for r in result}
        assert len(result) < 4  # fewer than 4 means some clustering happened

    @pytest.mark.asyncio
    async def test_canonical_is_longest_name(self):
        triplets = [
            _make_triplet("EU", "rel", "X"),
            _make_triplet("European Union", "rel", "Y"),
        ]
        embedder = AsyncMock()
        embedder.embed_texts = AsyncMock(return_value=[
            [1.0, 0.0],  # EU
            [0.99, 0.01],  # European Union
            [0.0, 1.0],  # X
            [0.1, 0.9],  # Y
        ])
        result = await normalize_entities(triplets, embedder=embedder, threshold=0.9)
        # Find the cluster containing EU
        for r in result:
            if "EU" in [r.canonical_name] + r.aliases:
                assert r.canonical_name == "European Union"
                break

    @pytest.mark.asyncio
    async def test_empty_triplets(self):
        result = await normalize_entities([], embedder=None)
        assert result == []


class TestBuildEntityMapping:
    def test_maps_aliases_to_canonical(self):
        from ayextractor.core.models import EntityNormalization
        norms = [
            EntityNormalization(
                canonical_name="European Union",
                aliases=["EU", "l'UE"],
                occurrence_count=5,
            ),
        ]
        mapping = build_entity_mapping(norms)
        assert mapping["EU"] == "European Union"
        assert mapping["l'UE"] == "European Union"
        assert mapping["European Union"] == "European Union"

    def test_empty_normalizations(self):
        assert build_entity_mapping([]) == {}
