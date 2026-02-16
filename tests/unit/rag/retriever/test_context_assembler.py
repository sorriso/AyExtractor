# tests/unit/rag/retriever/test_context_assembler.py — v1
"""Tests for rag/retriever/context_assembler.py."""

from __future__ import annotations

import pytest

from ayextractor.rag.models import SearchResult
from ayextractor.rag.retriever.context_assembler import (
    CHARS_PER_TOKEN,
    ContextAssembler,
)


def _sr(sid: str, score: float, stype: str, content: str = "") -> SearchResult:
    return SearchResult(
        source_type=stype, source_id=sid,
        content=content or f"Content of {sid} " * 10,
        score=score,
    )


class TestContextAssembler:
    def test_empty_results(self):
        asm = ContextAssembler()
        ctx = asm.assemble([])
        assert ctx.assembled_text == ""
        assert ctx.total_token_count == 0
        assert ctx.search_results == []

    def test_routes_chunks(self):
        asm = ContextAssembler()
        ctx = asm.assemble([_sr("c1", 0.9, "chunk")])
        assert len(ctx.chunk_excerpts) == 1
        assert ctx.community_summaries == []
        assert ctx.entity_profiles == []

    def test_routes_entity_profiles(self):
        asm = ContextAssembler()
        ctx = asm.assemble([_sr("e1", 0.9, "entity_profile")])
        assert len(ctx.entity_profiles) == 1

    def test_routes_relation_profiles(self):
        asm = ContextAssembler()
        ctx = asm.assemble([_sr("r1", 0.9, "relation_profile")])
        assert len(ctx.entity_profiles) == 1  # routed to entity_profiles

    def test_routes_community_summaries(self):
        asm = ContextAssembler()
        ctx = asm.assemble([_sr("m1", 0.9, "community_summary")])
        assert len(ctx.community_summaries) == 1

    def test_mixed_sources(self):
        asm = ContextAssembler()
        results = [
            _sr("c1", 0.95, "chunk"),
            _sr("e1", 0.90, "entity_profile"),
            _sr("m1", 0.85, "community_summary"),
        ]
        ctx = asm.assemble(results)
        assert len(ctx.chunk_excerpts) == 1
        assert len(ctx.entity_profiles) == 1
        assert len(ctx.community_summaries) == 1
        assert len(ctx.search_results) == 3

    def test_assembled_text_sections(self):
        asm = ContextAssembler()
        results = [
            _sr("m1", 0.95, "community_summary", "Theme about AI"),
            _sr("e1", 0.90, "entity_profile", "Entity EU info"),
            _sr("c1", 0.85, "chunk", "Source text excerpt"),
        ]
        ctx = asm.assemble(results)
        assert "## Document Context" in ctx.assembled_text
        assert "## Relevant Entities" in ctx.assembled_text
        assert "## Evidence" in ctx.assembled_text

    def test_token_budget_respected(self):
        asm = ContextAssembler(max_tokens=50)
        # Each result ~150 chars = ~37 tokens, budget=50 → fits ~1.3
        results = [
            _sr("c1", 0.9, "chunk", "A" * 150),
            _sr("c2", 0.8, "chunk", "B" * 150),
            _sr("c3", 0.7, "chunk", "C" * 150),
        ]
        ctx = asm.assemble(results)
        assert ctx.total_token_count <= 55  # some tolerance for truncation

    def test_truncation_with_ellipsis(self):
        asm = ContextAssembler(max_tokens=30)
        results = [
            _sr("c1", 0.9, "chunk", "Short"),
            _sr("c2", 0.8, "chunk", "X" * 500),
        ]
        ctx = asm.assemble(results)
        # Second result should be truncated
        found_truncated = any("..." in excerpt for excerpt in ctx.chunk_excerpts)
        assert found_truncated or len(ctx.chunk_excerpts) <= 1

    def test_override_max_tokens(self):
        asm = ContextAssembler(max_tokens=10000)
        # Override with small budget
        results = [_sr("c1", 0.9, "chunk", "X" * 500)]
        ctx = asm.assemble(results, max_tokens=20)
        assert ctx.total_token_count <= 25

    def test_token_count_approximate(self):
        asm = ContextAssembler()
        content = "A" * 400  # 400 chars ≈ 100 tokens
        ctx = asm.assemble([_sr("c1", 0.9, "chunk", content)])
        assert ctx.total_token_count == 400 // CHARS_PER_TOKEN

    def test_preserves_result_order(self):
        asm = ContextAssembler()
        results = [
            _sr("c1", 0.9, "chunk", "First"),
            _sr("c2", 0.8, "chunk", "Second"),
        ]
        ctx = asm.assemble(results)
        assert ctx.search_results[0].source_id == "c1"
        assert ctx.search_results[1].source_id == "c2"
