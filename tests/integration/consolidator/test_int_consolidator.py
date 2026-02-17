# tests/integration/consolidator/test_int_consolidator.py — v3
"""Integration tests for consolidator subsystem.

Covers: consolidator/community_clusterer.py, consolidator/orchestrator.py,
        consolidator/models.py
No Docker required.

Source API notes:
- TNode fields: canonical_name, level, parent, children, classified_cnodes,
  created_by, created_at
- cluster_corpus: internally calls _build_cnode_graph which uses
  corpus_store.to_networkx() if available, else returns empty graph
- _build_cnode_graph result goes through graph.number_of_nodes() < min_cluster_size
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import networkx as nx
import pytest


# =====================================================================
#  CONSOLIDATOR MODELS
# =====================================================================

class TestConsolidatorModels:

    def test_clustering_report(self):
        from ayextractor.consolidator.models import ClusteringReport
        report = ClusteringReport(new_tnodes=3, updated_tnodes=1, clusters_found=5)
        assert report.new_tnodes == 3
        assert report.clusters_found == 5

    def test_tnode_model(self):
        """TNode requires canonical_name, level, created_by, created_at."""
        from ayextractor.consolidator.models import TNode
        tnode = TNode(
            canonical_name="Machine Learning",
            level="concept",
            classified_cnodes=["c1", "c2", "c3"],
            created_by="consolidator_clustering",
            created_at=datetime.now(timezone.utc),
        )
        assert tnode.canonical_name == "Machine Learning"
        assert tnode.level == "concept"
        assert len(tnode.classified_cnodes) == 3

    def test_tnode_hierarchy(self):
        from ayextractor.consolidator.models import TNode
        parent = TNode(
            canonical_name="AI",
            level="domain",
            children=["Machine Learning"],
            created_by="manual",
            created_at=datetime.now(timezone.utc),
        )
        child = TNode(
            canonical_name="Machine Learning",
            level="subdomain",
            parent="AI",
            created_by="manual",
            created_at=datetime.now(timezone.utc),
        )
        assert child.parent == "AI"
        assert "Machine Learning" in parent.children

    def test_cnode_model(self):
        from ayextractor.consolidator.models import CNode
        cnode = CNode(
            canonical_name="GPT-4",
            entity_type="technology",
            aliases=["GPT4", "gpt-4o"],
            corroboration=3,
            confidence=0.85,
            first_seen_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
        assert cnode.canonical_name == "GPT-4"
        assert len(cnode.aliases) == 2

    def test_xedge_model(self):
        from ayextractor.consolidator.models import XEdge
        xedge = XEdge(
            source="EU",
            relation_type="enacted",
            target="NIS2",
            first_seen_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
        assert xedge.source == "EU"
        assert xedge.relation_type == "enacted"

    def test_pass_result(self):
        from ayextractor.consolidator.models import PassResult
        result = PassResult(
            pass_name="linking",
            duration_ms=1500,
            items_processed=100,
            items_modified=42,
        )
        assert result.pass_name == "linking"
        assert result.items_modified == 42


# =====================================================================
#  COMMUNITY CLUSTERER
# =====================================================================

class TestCommunityClusters:

    def test_cluster_corpus_too_small(self):
        """Graph with fewer nodes than min_cluster_size → skip."""
        from ayextractor.consolidator.community_clusterer import cluster_corpus

        # Build a mock store whose to_networkx returns a tiny graph
        small_graph = nx.Graph()
        small_graph.add_node("c1", node_type="cnode")
        mock_store = MagicMock()
        mock_store.to_networkx.return_value = small_graph

        report = cluster_corpus(mock_store, min_cluster_size=3)
        assert report.clusters_found == 0
        assert report.new_tnodes == 0

    def test_cluster_corpus_no_to_networkx(self):
        """Store without to_networkx → empty graph → 0 clusters."""
        from ayextractor.consolidator.community_clusterer import cluster_corpus

        mock_store = MagicMock(spec=[])  # spec=[] → no attributes
        report = cluster_corpus(mock_store, min_cluster_size=3)
        assert report.clusters_found == 0

    def test_clustering_report_fields(self):
        from ayextractor.consolidator.models import ClusteringReport
        r = ClusteringReport(new_tnodes=0, updated_tnodes=0, clusters_found=0)
        assert r.new_tnodes == 0


# =====================================================================
#  CONSOLIDATOR ORCHESTRATOR
# =====================================================================

class TestConsolidatorOrchestrator:

    def test_orchestrator_creation(self):
        from ayextractor.consolidator.orchestrator import ConsolidatorOrchestrator
        mock_store = MagicMock()
        orch = ConsolidatorOrchestrator(corpus_store=mock_store)
        assert orch is not None

    def test_orchestrator_has_pass_methods(self):
        from ayextractor.consolidator.orchestrator import ConsolidatorOrchestrator
        mock_store = MagicMock()
        orch = ConsolidatorOrchestrator(corpus_store=mock_store)
        assert hasattr(orch, "run_linking") or hasattr(orch, "run_pass")