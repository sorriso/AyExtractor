# tests/unit/consolidator/test_orchestrator.py — v1
"""Tests for consolidator/orchestrator.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ayextractor.consolidator.models import ConsolidationReport, PassResult
from ayextractor.consolidator.orchestrator import ConsolidatorOrchestrator


class _FakeStore:
    """Minimal fake graph store for testing."""

    def get_all_entities(self):
        return []

    def to_networkx(self):
        import networkx as nx
        return nx.Graph()

    def count_nodes(self):
        return 0

    def count_edges(self):
        return 0


class TestConsolidatorOrchestrator:
    def test_init_default_passes(self):
        orch = ConsolidatorOrchestrator(corpus_store=_FakeStore())
        assert "linking" in orch._active_passes
        assert "contradiction" in orch._active_passes
        assert len(orch._active_passes) == 5

    def test_init_with_settings_filter(self):
        settings = MagicMock()
        settings.CONSOLIDATOR_PASSES = "linking,decay"
        orch = ConsolidatorOrchestrator(
            corpus_store=_FakeStore(), settings=settings
        )
        assert orch._active_passes == ["linking", "decay"]

    def test_run_all_returns_report(self):
        orch = ConsolidatorOrchestrator(corpus_store=_FakeStore())
        # Patch all passes to avoid real execution
        with (
            patch.object(orch, "_execute_linking", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_clustering", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_inference", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_decay", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_contradiction", return_value={"items_processed": 0, "items_modified": 0}),
        ):
            report = orch.run_all(trigger="manual")

        assert isinstance(report, ConsolidationReport)
        assert report.trigger == "manual"
        assert len(report.passes_executed) == 5

    def test_run_all_handles_pass_failure(self):
        orch = ConsolidatorOrchestrator(corpus_store=_FakeStore())
        with (
            patch.object(orch, "_execute_linking", side_effect=RuntimeError("boom")),
            patch.object(orch, "_execute_clustering", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_inference", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_decay", return_value={"items_processed": 0, "items_modified": 0}),
            patch.object(orch, "_execute_contradiction", return_value={"items_processed": 0, "items_modified": 0}),
        ):
            report = orch.run_all()

        # linking failed but others should still execute
        assert "linking" not in report.passes_executed
        assert report.results["linking"].details.get("error") is True

    def test_run_linking_standalone(self):
        import networkx as nx

        g = nx.Graph()
        g.add_node("A", layer="L2", canonical_name="A", entity_type="organization", confidence=0.9)

        orch = ConsolidatorOrchestrator(corpus_store=_FakeStore())
        with patch(
            "ayextractor.graph.entity_linker.link_entities",
        ) as mock_link:
            from dataclasses import dataclass, field as dfield

            @dataclass
            class FakeResult:
                matched: dict = dfield(default_factory=dict)
                new_entities: list = dfield(default_factory=lambda: ["A"])
                stats: dict = dfield(default_factory=dict)

            import asyncio
            mock_link.return_value = FakeResult()

            result = orch.run_linking(g)

        assert isinstance(result, PassResult)
        assert result.pass_name == "linking"

    def test_run_linking_no_graph_skips(self):
        orch = ConsolidatorOrchestrator(corpus_store=_FakeStore())
        result = orch.run_linking(document_graph=None)  # type: ignore
        # Should return PassResult with skipped
        assert result.pass_name == "linking"

    def test_gather_corpus_stats(self):
        store = _FakeStore()
        orch = ConsolidatorOrchestrator(corpus_store=store)
        stats = orch._gather_corpus_stats()
        assert stats.get("total_cnodes") == 0


class TestPassOrdering:
    def test_passes_execute_in_order(self):
        orch = ConsolidatorOrchestrator(corpus_store=_FakeStore())
        expected = ["linking", "clustering", "inference", "decay", "contradiction"]
        assert orch._active_passes == expected
