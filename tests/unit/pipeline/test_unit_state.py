# tests/unit/pipeline/test_state.py — v1
"""Tests for pipeline/state.py — PipelineState."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
from ayextractor.pipeline.state import PipelineState


class TestPipelineState:
    def test_default_creation(self):
        state = PipelineState()
        assert state.run_id != ""
        assert state.document_id == ""
        assert state.chunks == []
        assert state.raw_triplets == []
        assert state.graph is None

    def test_run_id_unique(self):
        s1 = PipelineState()
        s2 = PipelineState()
        assert s1.run_id != s2.run_id

    def test_record_agent_output(self):
        state = PipelineState()
        output = AgentOutput(
            data={"key": "value"},
            confidence=0.9,
            metadata=AgentMetadata(
                agent_name="test_agent",
                agent_version="1.0.0",
                execution_time_ms=100,
                llm_calls=2,
                tokens_used=500,
            ),
        )
        state.record_agent_output("test_agent", output)

        assert "test_agent" in state.agent_outputs
        assert state.total_llm_calls == 2
        assert state.total_tokens_used == 500

    def test_record_multiple_agents(self):
        state = PipelineState()
        for i in range(3):
            output = AgentOutput(
                data={},
                confidence=0.8,
                metadata=AgentMetadata(
                    agent_name=f"agent_{i}",
                    agent_version="1.0.0",
                    execution_time_ms=50,
                    llm_calls=1,
                    tokens_used=100,
                ),
            )
            state.record_agent_output(f"agent_{i}", output)

        assert state.total_llm_calls == 3
        assert state.total_tokens_used == 300
        assert len(state.agent_outputs) == 3

    def test_get_graph_stats_no_graph(self):
        state = PipelineState()
        stats = state.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_get_graph_stats_with_graph(self):
        state = PipelineState()
        g = nx.Graph()
        g.add_node("A", layer=1)
        g.add_node("B", layer=2)
        g.add_node("C", layer=2)
        g.add_node("42", layer=3)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        state.graph = g

        stats = state.get_graph_stats()
        assert stats["nodes"] == 4
        assert stats["edges"] == 2
        assert stats["l1_nodes"] == 1
        assert stats["l2_nodes"] == 2
        assert stats["l3_nodes"] == 1

    def test_errors_accumulation(self):
        state = PipelineState()
        state.errors.append("Error 1")
        state.errors.append("Error 2")
        assert len(state.errors) == 2

    def test_arbitrary_types_allowed(self):
        """NetworkX Graph should be accepted by Pydantic."""
        state = PipelineState()
        state.graph = nx.Graph()
        assert state.graph is not None
