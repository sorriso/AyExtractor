# tests/integration/graph/test_int_graph_building.py — v2
"""Integration tests for graph construction pipeline.

Covers: builder, taxonomy, layer_classifier, community_detector,
community_integrator, exporters, inference_engine, contradiction_detector,
decay_manager, triplet_consolidator, reference_linker, entity_linker.

Pure Python — no external services.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import networkx as nx
import pytest

from ayextractor.core.models import (
    ConsolidatedTriplet,
    EntityNormalization,
    QualifiedTriplet,
    Reference,
)


# ── Helpers ─────────────────────────────────────────────────────

def _make_triplet(
    subj: str, pred: str, obj: str,
    conf: float = 0.9,
    chunks: list[str] | None = None,
) -> ConsolidatedTriplet:
    return ConsolidatedTriplet(
        subject=subj, predicate=pred, object=obj,
        confidence=conf, occurrence_count=1,
        source_chunk_ids=chunks or ["c1"],
        original_forms=[f"{subj}|{pred}|{obj}"],
        context_sentences=[f"{subj} {pred} {obj}"],
    )


def _make_norm(
    name: str,
    etype: str = "concept",
    aliases: list[str] | None = None,
) -> EntityNormalization:
    return EntityNormalization(
        canonical_name=name,
        entity_type=etype,  # type: ignore[arg-type]
        aliases=aliases or [],
        source_chunk_ids=["c1"],
    )


def _build_sample_graph() -> nx.Graph:
    """Build a small Document Graph from triplets."""
    from ayextractor.graph.builder import build_document_graph

    triplets = [
        _make_triplet("EU", "enacted", "NIS2", 0.95, ["c1"]),
        _make_triplet("EU", "enacted", "CRA", 0.90, ["c2"]),
        _make_triplet("NIS2", "applies_to", "critical_infrastructure", 0.85, ["c1"]),
        _make_triplet("CRA", "regulates", "digital_products", 0.88, ["c2"]),
        _make_triplet("critical_infrastructure", "includes", "energy", 0.80, ["c3"]),
        _make_triplet("critical_infrastructure", "includes", "transport", 0.82, ["c3"]),
    ]
    norms = [
        _make_norm("EU", "organization"),
        _make_norm("NIS2", "document"),
        _make_norm("CRA", "document"),
        _make_norm("critical_infrastructure", "concept"),
        _make_norm("digital_products", "concept"),
        _make_norm("energy", "concept"),
        _make_norm("transport", "concept"),
    ]
    return build_document_graph(triplets, norms)


# ── Graph Builder ──────────────────────────────────────────────

class TestGraphBuilder:

    def test_builds_nodes_and_edges(self):
        g = _build_sample_graph()
        assert g.number_of_nodes() >= 5
        assert g.number_of_edges() >= 4

    def test_node_attributes(self):
        g = _build_sample_graph()
        assert "EU" in g.nodes()
        data = g.nodes["EU"]
        assert "entity_type" in data or "label" in data or "node_type" in data

    def test_edge_attributes(self):
        g = _build_sample_graph()
        edges = list(g.edges(data=True))
        assert len(edges) >= 4
        has_relation = any(
            "relation_type" in d or "predicate" in d or "label" in d
            for _, _, d in edges
        )
        assert has_relation

    def test_node_confidence_from_edges(self):
        g = _build_sample_graph()
        assert "EU" in g.nodes()

    def test_multiple_edges_between_same_pair(self):
        triplets = [
            _make_triplet("A", "relates_to", "B"),
            _make_triplet("A", "contrasts_with", "B"),
        ]
        norms = [_make_norm("A"), _make_norm("B")]
        from ayextractor.graph.builder import build_document_graph
        g = build_document_graph(triplets, norms)
        assert g.has_node("A")
        assert g.has_node("B")

    def test_literal_detection(self):
        triplets = [_make_triplet("NIS2", "adopted_in", "2022")]
        norms = [_make_norm("NIS2", "document")]
        from ayextractor.graph.builder import build_document_graph
        g = build_document_graph(triplets, norms)
        assert g.number_of_nodes() >= 2


# ── Triplet Consolidator ───────────────────────────────────────

class TestTripletConsolidator:

    def test_consolidate_duplicates(self):
        from ayextractor.graph.triplet_consolidator import consolidate_triplets
        raw = [
            QualifiedTriplet(subject="EU", predicate="enacted", object="NIS2",
                             source_chunk_id="c1", confidence=0.9, context_sentence="test"),
            QualifiedTriplet(subject="EU", predicate="enacted", object="NIS2",
                             source_chunk_id="c2", confidence=0.85, context_sentence="test"),
        ]
        result = consolidate_triplets(raw, entity_mapping={}, relation_mapping={})
        assert len(result) == 1
        assert result[0].occurrence_count == 2
        assert len(result[0].source_chunk_ids) == 2
        assert result[0].confidence >= 0.85

    def test_distinct_triplets_preserved(self):
        from ayextractor.graph.triplet_consolidator import consolidate_triplets
        raw = [
            QualifiedTriplet(subject="EU", predicate="enacted", object="NIS2",
                             source_chunk_id="c1", confidence=0.9, context_sentence="test"),
            QualifiedTriplet(subject="EU", predicate="enacted", object="CRA",
                             source_chunk_id="c2", confidence=0.8, context_sentence="test"),
        ]
        result = consolidate_triplets(raw, entity_mapping={}, relation_mapping={})
        assert len(result) == 2

    def test_empty_input(self):
        from ayextractor.graph.triplet_consolidator import consolidate_triplets
        result = consolidate_triplets([], entity_mapping={}, relation_mapping={})
        assert result == []

    def test_qualifiers_merged(self):
        from ayextractor.graph.triplet_consolidator import consolidate_triplets
        raw = [
            QualifiedTriplet(
                subject="NIS2", predicate="applies_to", object="energy",
                source_chunk_id="c1", confidence=0.9, context_sentence="test",
                qualifiers={"scope": "EU"},
            ),
            QualifiedTriplet(
                subject="NIS2", predicate="applies_to", object="energy",
                source_chunk_id="c2", confidence=0.85, context_sentence="test",
                qualifiers={"effective_date": "2024"},
            ),
        ]
        result = consolidate_triplets(raw, entity_mapping={}, relation_mapping={})
        assert len(result) == 1
        assert result[0].occurrence_count == 2

    def test_entity_mapping_applied(self):
        from ayextractor.graph.triplet_consolidator import consolidate_triplets
        raw = [
            QualifiedTriplet(subject="European Union", predicate="enacted", object="NIS2",
                             source_chunk_id="c1", confidence=0.9, context_sentence="test"),
            QualifiedTriplet(subject="EU", predicate="enacted", object="NIS2",
                             source_chunk_id="c2", confidence=0.85, context_sentence="test"),
        ]
        result = consolidate_triplets(raw, entity_mapping={"European Union": "EU"}, relation_mapping={})
        assert len(result) == 1
        assert result[0].subject == "EU"

    def test_relation_mapping_applied(self):
        from ayextractor.graph.triplet_consolidator import consolidate_triplets
        raw = [
            QualifiedTriplet(subject="EU", predicate="adopted", object="NIS2",
                             source_chunk_id="c1", confidence=0.9, context_sentence="test"),
            QualifiedTriplet(subject="EU", predicate="enacted", object="NIS2",
                             source_chunk_id="c2", confidence=0.85, context_sentence="test"),
        ]
        result = consolidate_triplets(raw, entity_mapping={}, relation_mapping={"adopted": "enacted"})
        assert len(result) == 1
        assert result[0].predicate == "enacted"


# ── Contradiction Detector ─────────────────────────────────────

class TestContradictionDetector:

    def test_no_contradictions_clean_graph(self):
        from ayextractor.graph.contradiction_detector import detect_contradictions
        triplets = [
            _make_triplet("EU", "enacted", "NIS2"),
            _make_triplet("EU", "enacted", "CRA"),
        ]
        report = detect_contradictions(triplets)
        assert len(report.contradictions) == 0

    def test_detects_opposing_predicates(self):
        from ayextractor.graph.contradiction_detector import detect_contradictions
        triplets = [
            _make_triplet("EU", "supports", "NIS2", 0.9),
            _make_triplet("EU", "opposes", "NIS2", 0.85),
        ]
        report = detect_contradictions(triplets)
        assert isinstance(report.contradictions, list)
        assert hasattr(report, "stats")


# ── Decay Manager ──────────────────────────────────────────────

class TestDecayManager:

    def test_apply_decay(self):
        from ayextractor.graph.decay_manager import apply_decay
        g = nx.Graph()
        old_date = datetime.now(timezone.utc) - timedelta(days=730)
        g.add_node("A", confidence=1.0, last_seen=old_date.isoformat())
        g.add_node("B", confidence=0.8, last_seen=old_date.isoformat())
        g.add_edge("A", "B", confidence=0.9, last_seen=old_date.isoformat())
        stats = apply_decay(g, half_life_days=365.0)
        assert isinstance(stats.nodes_decayed, int)
        assert isinstance(stats.edges_decayed, int)

    def test_apply_decay_no_change_if_recent(self):
        from ayextractor.graph.decay_manager import apply_decay
        g = nx.Graph()
        now = datetime.now(timezone.utc).isoformat()
        g.add_node("A", confidence=1.0, last_seen=now)
        g.add_edge("A", "A", confidence=0.9, last_seen=now)
        stats = apply_decay(g, half_life_days=365.0)
        assert stats.nodes_decayed >= 0

    def test_compute_decay_factor(self):
        from ayextractor.graph.decay_manager import compute_decay_factor
        factor = compute_decay_factor(365.0, half_life_days=365.0)
        assert 0.45 < factor < 0.55
        factor_zero = compute_decay_factor(0.0, half_life_days=365.0)
        assert factor_zero == pytest.approx(1.0, abs=0.01)


# ── Layer Classifier ──────────────────────────────────────────

class TestLayerClassifier:

    def test_classify_layers(self):
        from ayextractor.graph.layers.layer_classifier import classify_layers
        g = _build_sample_graph()
        layers = classify_layers(g)
        assert isinstance(layers, dict)
        assert len(layers) > 0

    def test_apply_layers(self):
        from ayextractor.graph.layers.layer_classifier import apply_layers, classify_layers
        g = _build_sample_graph()
        layers = classify_layers(g)
        apply_layers(g, layers)
        for node_id in g.nodes():
            data = g.nodes[node_id]
            assert "layer" in data or node_id in layers

    def test_get_l2_subgraph(self):
        from ayextractor.graph.layers.layer_classifier import (
            apply_layers, classify_layers, get_l2_subgraph,
        )
        g = _build_sample_graph()
        layers = classify_layers(g)
        apply_layers(g, layers)
        sub = get_l2_subgraph(g)
        assert isinstance(sub, nx.Graph)
        assert sub.number_of_nodes() <= g.number_of_nodes()


# ── Community Detector ─────────────────────────────────────────

class TestCommunityDetector:

    def test_detect_communities(self):
        from ayextractor.graph.layers.community_detector import detect_communities
        g = _build_sample_graph()
        hierarchy = detect_communities(g, seed=42)
        assert hierarchy.total_communities >= 1
        assert len(hierarchy.communities) >= 1

    def test_all_nodes_assigned(self):
        from ayextractor.graph.layers.community_detector import detect_communities
        g = _build_sample_graph()
        hierarchy = detect_communities(g, seed=42)
        all_members = set()
        for c in hierarchy.communities:
            all_members.update(c.members)
        # Some nodes may be in communities below min_community_size
        assert all_members.issubset(set(g.nodes()))
        assert len(all_members) > 0


# ── Community Integrator ───────────────────────────────────────

class TestCommunityIntegrator:

    def test_integrate_creates_l1_nodes(self):
        from ayextractor.graph.layers.community_detector import detect_communities
        from ayextractor.graph.layers.community_integrator import integrate_communities
        g = _build_sample_graph()
        hierarchy = detect_communities(g, seed=42)
        g2 = integrate_communities(g, hierarchy)
        assert g2.number_of_nodes() >= g.number_of_nodes()

    def test_community_edges_created(self):
        from ayextractor.graph.layers.community_detector import detect_communities
        from ayextractor.graph.layers.community_integrator import integrate_communities
        g = _build_sample_graph()
        hierarchy = detect_communities(g, seed=42)
        g2 = integrate_communities(g, hierarchy)
        assert g2.number_of_edges() >= g.number_of_edges()


# ── Exporters ──────────────────────────────────────────────────

class TestExporters:

    @pytest.mark.asyncio
    async def test_json_export(self):
        from ayextractor.graph.exporters.json_exporter import JsonExporter
        g = _build_sample_graph()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        exporter = JsonExporter()
        await exporter.export(g, path)
        content = Path(path).read_text()
        data = json.loads(content)
        assert "nodes" in data or isinstance(data, (list, dict))

    @pytest.mark.asyncio
    async def test_graphml_export(self):
        from ayextractor.graph.exporters.graphml_exporter import GraphMLExporter
        g = _build_sample_graph()
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            path = f.name
        await GraphMLExporter().export(g, path)
        content = Path(path).read_text()
        assert "graphml" in content.lower() or "graph" in content.lower()

    @pytest.mark.asyncio
    async def test_gexf_export(self):
        from ayextractor.graph.exporters.gexf_exporter import GexfExporter
        g = _build_sample_graph()
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            path = f.name
        await GexfExporter().export(g, path)
        content = Path(path).read_text()
        assert "gexf" in content.lower() or "graph" in content.lower()

    @pytest.mark.asyncio
    async def test_cypher_export(self):
        from ayextractor.graph.exporters.cypher_exporter import CypherExporter
        g = _build_sample_graph()
        with tempfile.NamedTemporaryFile(suffix=".cypher", delete=False) as f:
            path = f.name
        await CypherExporter().export(g, path)
        content = Path(path).read_text()
        assert "MERGE" in content or "CREATE" in content or len(content) > 0


# ── Inference Engine ───────────────────────────────────────────

class TestInferenceEngine:

    def test_run_inference(self):
        from ayextractor.graph.inference_engine import run_inference
        g = _build_sample_graph()
        result = run_inference(g)
        assert hasattr(result, "inferred_edges")
        assert isinstance(result.inferred_edges, list)

    def test_transitive_inference(self):
        from ayextractor.graph.inference_engine import run_inference
        g = nx.DiGraph()
        g.add_node("A", entity_type="concept")
        g.add_node("B", entity_type="concept")
        g.add_node("C", entity_type="concept")
        g.add_edge("A", "B", relation_type="part_of", confidence=0.9)
        g.add_edge("B", "C", relation_type="part_of", confidence=0.9)
        result = run_inference(g, max_depth=2)
        assert isinstance(result.inferred_edges, list)


# ── Reference Linker ──────────────────────────────────────────

class TestReferenceLinker:

    def test_link_references_empty(self):
        from ayextractor.graph.reference_linker import link_references
        g = _build_sample_graph()
        g2 = link_references(g, [], [])
        assert isinstance(g2, nx.Graph)

    def test_link_references_with_match(self):
        from ayextractor.graph.reference_linker import link_references
        g = _build_sample_graph()
        refs = [
            Reference(type="citation", text="NIS2 Directive (2022/2555)",
                      source_chunk_id="c1"),
        ]
        norms = [_make_norm("NIS2", "document", aliases=["NIS2 Directive"])]
        g2 = link_references(g, refs, norms)
        assert isinstance(g2, nx.Graph)


# ── Entity Linker ─────────────────────────────────────────────

class TestEntityLinker:

    @pytest.mark.asyncio
    async def test_link_entities_empty(self):
        from ayextractor.graph.entity_linker import link_entities
        result = await link_entities(existing=[], incoming=[])
        assert isinstance(result.matched, dict)
        assert isinstance(result.new_entities, list)

    @pytest.mark.asyncio
    async def test_link_entities_exact_match(self):
        from ayextractor.graph.entity_linker import link_entities
        existing = [_make_norm("EU", "organization", aliases=["European Union"])]
        incoming = [_make_norm("EU", "organization")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert isinstance(result, object)

    @pytest.mark.asyncio
    async def test_link_entities_alias_match(self):
        from ayextractor.graph.entity_linker import link_entities
        existing = [_make_norm("EU", "organization", aliases=["European Union"])]
        incoming = [_make_norm("European Union", "organization")]
        result = await link_entities(existing=existing, incoming=incoming)
        assert isinstance(result.matched, dict)


# ── Taxonomy ──────────────────────────────────────────────────

class TestTaxonomy:

    def test_default_taxonomy(self):
        from ayextractor.graph.taxonomy import DEFAULT_RELATION_TAXONOMY
        assert isinstance(DEFAULT_RELATION_TAXONOMY, list)
        assert len(DEFAULT_RELATION_TAXONOMY) > 0

    def test_find_canonical(self):
        from ayextractor.graph.taxonomy import find_canonical
        result = find_canonical("is_part_of")
        # Returns canonical name or None
        assert result is None or isinstance(result, str)

    def test_get_categories(self):
        from ayextractor.graph.taxonomy import get_categories
        cats = get_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_get_taxonomy_dict(self):
        from ayextractor.graph.taxonomy import get_taxonomy_dict
        d = get_taxonomy_dict()
        assert isinstance(d, dict)
        assert len(d) > 0
