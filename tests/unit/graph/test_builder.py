# tests/unit/graph/test_builder.py — v1
"""Tests for graph/builder.py — document graph construction from consolidated triplets."""

from __future__ import annotations

import pytest

from ayextractor.core.models import (
    ConsolidatedTriplet,
    EntityNormalization,
    TemporalScope,
)
from ayextractor.graph.builder import build_document_graph, _is_literal


@pytest.fixture
def sample_entities():
    return [
        EntityNormalization(
            canonical_name="European Union",
            aliases=["EU"],
            entity_type="organization",
            occurrence_count=5,
            source_chunk_ids=["c1", "c2"],
        ),
        EntityNormalization(
            canonical_name="AI Act",
            aliases=[],
            entity_type="document",
            occurrence_count=3,
            source_chunk_ids=["c1"],
        ),
    ]


@pytest.fixture
def sample_triplets():
    return [
        ConsolidatedTriplet(
            subject="European Union",
            predicate="regulates",
            object="AI systems",
            source_chunk_ids=["c1", "c2"],
            occurrence_count=2,
            confidence=0.92,
            original_forms=["regulates"],
            context_sentences=["The EU regulates AI systems."],
        ),
        ConsolidatedTriplet(
            subject="AI Act",
            predicate="requires",
            object="risk assessment",
            source_chunk_ids=["c1"],
            occurrence_count=1,
            confidence=0.85,
            original_forms=["requires"],
        ),
    ]


class TestBuildDocumentGraph:
    def test_creates_nodes_for_all_entities(self, sample_triplets, sample_entities):
        g = build_document_graph(sample_triplets, sample_entities)
        assert "European Union" in g.nodes
        assert "AI systems" in g.nodes
        assert "AI Act" in g.nodes
        assert "risk assessment" in g.nodes

    def test_creates_edges_for_all_triplets(self, sample_triplets, sample_entities):
        g = build_document_graph(sample_triplets, sample_entities)
        assert g.has_edge("European Union", "AI systems")
        assert g.has_edge("AI Act", "risk assessment")

    def test_edge_attributes(self, sample_triplets, sample_entities):
        g = build_document_graph(sample_triplets, sample_entities)
        edge = g["European Union"]["AI systems"]
        assert edge["relation_type"] == "regulates"
        assert edge["confidence"] == 0.92
        assert edge["occurrence_count"] == 2

    def test_node_uses_entity_metadata(self, sample_triplets, sample_entities):
        g = build_document_graph(sample_triplets, sample_entities)
        node = g.nodes["European Union"]
        assert node["entity_type"] == "organization"
        assert "EU" in node["aliases"]

    def test_node_default_type_concept(self, sample_triplets, sample_entities):
        """Entities not in normalization table get default 'concept' type."""
        g = build_document_graph(sample_triplets, sample_entities)
        node = g.nodes["AI systems"]
        assert node["entity_type"] == "concept"

    def test_node_layer_assignment(self, sample_triplets, sample_entities):
        g = build_document_graph(sample_triplets, sample_entities)
        assert g.nodes["European Union"]["layer"] == 2
        assert g.nodes["AI Act"]["layer"] == 2

    def test_literal_node_gets_l3(self):
        triplets = [
            ConsolidatedTriplet(
                subject="Acme Corp",
                predicate="has_attribute",
                object="500",
                source_chunk_ids=["c1"],
                occurrence_count=1,
                confidence=0.8,
            ),
        ]
        g = build_document_graph(triplets, [])
        assert g.nodes["500"]["layer"] == 3
        assert g.nodes["500"]["is_literal"] is True

    def test_node_confidence_from_max_edge(self, sample_triplets, sample_entities):
        g = build_document_graph(sample_triplets, sample_entities)
        assert g.nodes["European Union"]["confidence"] == 0.92

    def test_empty_input(self):
        g = build_document_graph([], [])
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_temporal_scope_on_edge(self, sample_entities):
        triplets = [
            ConsolidatedTriplet(
                subject="European Union",
                predicate="regulates",
                object="AI",
                source_chunk_ids=["c1"],
                occurrence_count=1,
                confidence=0.9,
                temporal_scope=TemporalScope(
                    type="point", start="2025", granularity="year",
                    raw_expression="since 2025",
                ),
            ),
        ]
        g = build_document_graph(triplets, sample_entities)
        edge = g["European Union"]["AI"]
        assert edge["temporal_scope"]["type"] == "point"

    def test_multiple_edges_same_pair(self, sample_entities):
        """Multiple relations between same nodes keep highest confidence."""
        triplets = [
            ConsolidatedTriplet(
                subject="European Union", predicate="regulates", object="AI",
                source_chunk_ids=["c1"], occurrence_count=1, confidence=0.7,
            ),
            ConsolidatedTriplet(
                subject="European Union", predicate="monitors", object="AI",
                source_chunk_ids=["c2"], occurrence_count=1, confidence=0.9,
            ),
        ]
        g = build_document_graph(triplets, sample_entities)
        edge = g["European Union"]["AI"]
        assert edge["confidence"] == 0.9


class TestIsLiteral:
    def test_pure_number(self):
        assert _is_literal("500") is True

    def test_percentage(self):
        assert _is_literal("15%") is True

    def test_currency(self):
        assert _is_literal("42€") is True

    def test_measurement(self):
        assert _is_literal("500 employees") is True

    def test_date_pattern(self):
        assert _is_literal("2025-01-15") is True

    def test_named_entity_not_literal(self):
        assert _is_literal("European Union") is False

    def test_empty_string(self):
        assert _is_literal("") is False

    def test_concept_not_literal(self):
        assert _is_literal("cybersecurity") is False
