# tests/unit/rag/retriever/test_query_classifier.py — v1
"""Tests for rag/retriever/query_classifier.py."""

from __future__ import annotations

import pytest

from ayextractor.rag.retriever.query_classifier import classify_query


class TestClassifyQuery:
    def test_empty_query_exploratory(self):
        plan = classify_query("")
        assert plan.query_type == "exploratory"
        assert "community" in plan.levels_to_query

    def test_factual_query(self):
        plan = classify_query("What is ISO 21434?")
        assert plan.query_type == "factual"
        assert "chunk" in plan.levels_to_query
        assert "entity" in plan.levels_to_query

    def test_who_query_factual(self):
        plan = classify_query("Who regulates cybersecurity?")
        assert plan.query_type == "factual"

    def test_how_many_factual(self):
        plan = classify_query("How many standards are referenced?")
        assert plan.query_type == "factual"

    def test_relational_query(self):
        plan = classify_query("What is the relationship between EU and OEMs?")
        assert plan.query_type == "relational"
        assert "entity" in plan.levels_to_query

    def test_impact_relational(self):
        plan = classify_query("How does NIS2 impact supply chains?")
        assert plan.query_type == "relational"

    def test_between_relational(self):
        plan = classify_query("What connects ISO 21434 and UN R155?")
        assert plan.query_type == "relational"

    def test_exploratory_query(self):
        plan = classify_query("Give me a summary of the document")
        assert plan.query_type == "exploratory"
        assert "community" in plan.levels_to_query

    def test_overview_exploratory(self):
        plan = classify_query("Provide an overview of the main themes")
        assert plan.query_type == "exploratory"

    def test_conceptual_fallback(self):
        """No strong signal → conceptual."""
        plan = classify_query("cybersecurity frameworks automotive")
        assert plan.query_type == "conceptual"
        assert "entity" in plan.levels_to_query
        assert "chunk" in plan.levels_to_query

    def test_estimated_token_cost_positive(self):
        plan = classify_query("What is ISO 21434?")
        assert plan.estimated_token_cost > 0

    def test_levels_always_non_empty(self):
        for q in ["", "test", "who what where", "summary overview"]:
            plan = classify_query(q)
            assert len(plan.levels_to_query) >= 1

    def test_case_insensitive(self):
        p1 = classify_query("WHAT IS ISO 21434?")
        p2 = classify_query("what is iso 21434?")
        assert p1.query_type == p2.query_type
