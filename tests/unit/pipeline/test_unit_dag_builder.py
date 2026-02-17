# tests/unit/pipeline/test_dag_builder.py — v1
"""Tests for pipeline/dag_builder.py — DAG construction and topological sort."""

from __future__ import annotations

import pytest

from ayextractor.pipeline.dag_builder import DAGError, ExecutionPlan, build_dag


class TestBuildDAG:
    def test_empty_map(self):
        plan = build_dag({})
        assert plan.total_agents == 0
        assert plan.stages == []
        assert plan.flat_order == []

    def test_single_agent_no_deps(self):
        plan = build_dag({"summarizer": []})
        assert plan.total_agents == 1
        assert plan.flat_order == ["summarizer"]
        assert len(plan.stages) == 1

    def test_linear_chain(self):
        dep_map = {
            "a": [],
            "b": ["a"],
            "c": ["b"],
        }
        plan = build_dag(dep_map)
        assert plan.total_agents == 3
        order = plan.flat_order
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_parallel_agents_same_stage(self):
        dep_map = {
            "root": [],
            "branch_a": ["root"],
            "branch_b": ["root"],
        }
        plan = build_dag(dep_map)
        assert plan.total_agents == 3
        # root in stage 0, branches in stage 1
        assert plan.stages[0] == ["root"]
        assert set(plan.stages[1]) == {"branch_a", "branch_b"}

    def test_diamond_dependency(self):
        dep_map = {
            "a": [],
            "b": ["a"],
            "c": ["a"],
            "d": ["b", "c"],
        }
        plan = build_dag(dep_map)
        assert plan.total_agents == 4
        order = plan.flat_order
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_raises(self):
        dep_map = {
            "a": ["b"],
            "b": ["a"],
        }
        with pytest.raises(DAGError, match="Cycle"):
            build_dag(dep_map)

    def test_missing_dependency_raises(self):
        dep_map = {
            "a": ["nonexistent"],
        }
        with pytest.raises(DAGError, match="not registered"):
            build_dag(dep_map)

    def test_complex_graph(self):
        dep_map = {
            "summarizer": [],
            "densifier": ["summarizer"],
            "concept_extractor": [],
            "community_summarizer": ["concept_extractor"],
            "profile_generator": ["concept_extractor"],
            "synthesizer": ["densifier", "community_summarizer", "profile_generator"],
            "critic": ["synthesizer"],
        }
        plan = build_dag(dep_map)
        assert plan.total_agents == 7
        order = plan.flat_order
        assert order.index("summarizer") < order.index("densifier")
        assert order.index("densifier") < order.index("synthesizer")
        assert order.index("concept_extractor") < order.index("community_summarizer")
        assert order.index("synthesizer") < order.index("critic")

    def test_stages_reflect_concurrency(self):
        dep_map = {
            "summarizer": [],
            "concept_extractor": [],
            "densifier": ["summarizer"],
            "critic": ["densifier", "concept_extractor"],
        }
        plan = build_dag(dep_map)
        # Stage 0: summarizer, concept_extractor (no deps)
        assert set(plan.stages[0]) == {"concept_extractor", "summarizer"}
        # Stage 1: densifier
        assert plan.stages[1] == ["densifier"]
        # Stage 2: critic
        assert plan.stages[2] == ["critic"]


class TestExecutionPlan:
    def test_flat_order(self):
        plan = ExecutionPlan(
            stages=[["a", "b"], ["c"], ["d"]],
            total_agents=4,
        )
        assert plan.flat_order == ["a", "b", "c", "d"]

    def test_empty_flat_order(self):
        plan = ExecutionPlan()
        assert plan.flat_order == []
