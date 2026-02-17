# tests/unit/consolidator/test_models.py — v1
"""Tests for consolidator/models.py — Corpus Graph and consolidation report models."""

from __future__ import annotations

from datetime import datetime, timezone

from ayextractor.consolidator.models import (
    CNode,
    ClusteringReport,
    ConsolidationReport,
    Contradiction,
    ContradictionReport,
    DecayReport,
    InferenceReport,
    LinkingReport,
    PassResult,
    TNode,
    XEdge,
)


class TestTNode:
    def test_create(self):
        t = TNode(
            canonical_name="Cybersecurity", level="domain",
            created_by="manual", created_at=datetime.now(timezone.utc),
        )
        assert t.parent is None
        assert t.children == []


class TestCNode:
    def test_create(self, sample_provenance):
        c = CNode(
            canonical_name="ISO 21434", entity_type="document",
            aliases=["ISO/SAE 21434", "21434"],
            source_documents=[sample_provenance], corroboration=1,
            confidence=0.9, salience=0.8,
            first_seen_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
        assert c.corroboration == 1
        assert c.taxonomy_path is None


class TestXEdge:
    def test_create(self):
        x = XEdge(
            source="ISO 21434", relation_type="regulates", target="automotive cybersecurity",
            corroboration=2, confidence=0.85,
            first_seen_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
        assert x.inferred is False


class TestContradiction:
    def test_create(self):
        c = Contradiction(
            contradiction_id="contr_001",
            edge_a_subject="A", edge_a_predicate="is", edge_a_object="B",
            edge_b_subject="A", edge_b_predicate="is_not", edge_b_object="B",
            conflict_type="negation",
        )
        assert c.conflict_type == "negation"


class TestConsolidationReport:
    def test_create(self):
        r = ConsolidationReport(
            consolidation_id="cons_001",
            timestamp=datetime.now(timezone.utc),
            trigger="on_ingestion",
            passes_executed=["linking", "clustering"],
            results={
                "linking": PassResult(
                    pass_name="linking", duration_ms=5000,
                    items_processed=100, items_modified=25,
                ),
            },
        )
        assert len(r.passes_executed) == 2
