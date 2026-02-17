# tests/integration/rag/retriever/test_int_rag_retrievers.py — v4
"""Integration tests for RAG retrieval pipeline.

Covers: rag/retriever/ppr_scorer.py, rag/retriever/context_assembler.py,
        rag/retriever/query_classifier.py, rag/retriever/chunk_retriever.py,
        rag/enricher.py, rag/models.py,
        rag/embeddings/base_embedder.py, rag/embeddings/embedder_factory.py

No Docker required — uses in-memory stores + mock embedder.

Source API reference (verified against deployed src/):
- ppr_score(graph, seed_entities, alpha=0.15) → dict[str, float]
  alpha is passed as damping factor to nx.pagerank:
    higher alpha = more link following = more spread to distant nodes
- ContextAssembler(max_tokens).assemble(results, ...) → RAGContext
- RAGContext.assembled_text (required), .chunk_excerpts, .entity_profiles, ...
- build_context(agent_name, state, vector_store=None, ...) → RAGContext | None
- SearchResult.source_type: Literal["chunk","entity_profile","relation_profile","community_summary"]

Changelog:
    v4: Fix PPR alpha test — alpha IS damping factor (link-follow probability)
        passed directly to nx.pagerank. Higher alpha → more spread → distant
        node gets HIGHER normalized score.
    v3: Fix SearchResult source_type Literal.
    v2: Fix all API names.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from ayextractor.config.settings import Settings
from ayextractor.rag.models import RAGContext, SearchResult


DIMS = 128


def _rand_vec(dims: int = DIMS, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


# =====================================================================
#  QUERY CLASSIFIER
# =====================================================================

class TestQueryClassifier:

    def test_classify_entity_query(self):
        from ayextractor.rag.retriever.query_classifier import classify_query
        plan = classify_query("What is the role of GPT-4 in education?")
        assert plan is not None
        assert hasattr(plan, "query_type") or hasattr(plan, "retrieval_levels")

    def test_classify_factual_query(self):
        from ayextractor.rag.retriever.query_classifier import classify_query
        plan = classify_query("When was Python first released?")
        assert plan is not None

    def test_classify_broad_query(self):
        from ayextractor.rag.retriever.query_classifier import classify_query
        plan = classify_query("Summarize the main themes of the document")
        assert plan is not None


# =====================================================================
#  PPR SCORER
# =====================================================================

class TestPPRScorer:

    def test_ppr_simple_graph(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.Graph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")])
        scores = ppr_score(g, seed_entities=["A"])
        assert isinstance(scores, dict)
        assert "A" in scores
        assert scores["A"] > 0

    def test_ppr_seed_has_high_score(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.Graph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        scores = ppr_score(g, seed_entities=["A"])
        assert scores["A"] >= scores.get("D", 0)

    def test_ppr_multiple_seeds(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.Graph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("D", "E")])
        scores = ppr_score(g, seed_entities=["A", "D"])
        assert scores["A"] > 0
        assert scores["D"] > 0

    def test_ppr_empty_graph(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.Graph()
        scores = ppr_score(g, seed_entities=[])
        assert scores == {} or isinstance(scores, dict)

    def test_ppr_disconnected_components(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.Graph()
        g.add_edges_from([("A", "B"), ("C", "D")])
        scores = ppr_score(g, seed_entities=["A"])
        assert scores.get("A", 0) > scores.get("C", 0)

    def test_ppr_single_node(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.Graph()
        g.add_node("solo")
        scores = ppr_score(g, seed_entities=["solo"])
        assert isinstance(scores, dict)

    def test_ppr_large_graph(self):
        from ayextractor.rag.retriever.ppr_scorer import ppr_score
        g = nx.barabasi_albert_graph(200, 3, seed=42)
        scores = ppr_score(g, seed_entities=[0, 1])
        assert len(scores) == 200

    def test_ppr_alpha_parameter(self):
        """Alpha is the damping factor passed to nx.pagerank.

        Higher alpha = higher probability of following a link (vs teleporting).
        This means MORE score spread to distant nodes.
        After normalization (max=1.0), distant node 9 gets a HIGHER
        normalized score with high alpha than with low alpha.
        """
        from ayextractor.rag.retriever.ppr_scorer import ppr_score

        g = nx.path_graph(10)  # 0-1-2-...-9
        scores_low = ppr_score(g, seed_entities=[0], alpha=0.05)
        scores_high = ppr_score(g, seed_entities=[0], alpha=0.5)

        # Higher damping factor → more spread → distant node gets MORE score
        assert scores_high[9] > scores_low[9]

    def test_ppr_combined_score(self):
        from ayextractor.rag.retriever.ppr_scorer import combined_score
        result = combined_score(vector_score=1.0, ppr=0.0)
        assert abs(result - 0.3) < 0.01
        result = combined_score(vector_score=0.0, ppr=1.0)
        assert abs(result - 0.7) < 0.01


# =====================================================================
#  CONTEXT ASSEMBLER
# =====================================================================

class TestContextAssembler:

    def test_assemble_empty_results(self):
        from ayextractor.rag.retriever.context_assembler import ContextAssembler
        assembler = ContextAssembler(max_tokens=4000)
        ctx = assembler.assemble(results=[])
        assert isinstance(ctx, RAGContext)
        assert ctx.assembled_text is not None

    def test_assemble_with_results(self):
        from ayextractor.rag.retriever.context_assembler import ContextAssembler
        results = [
            SearchResult(
                source_type="chunk", source_id="c1",
                content="chunk 1 text", score=0.9, metadata={},
            ),
            SearchResult(
                source_type="entity_profile", source_id="e1",
                content="entity profile text", score=0.8, metadata={},
            ),
            SearchResult(
                source_type="chunk", source_id="c2",
                content="chunk 2 text", score=0.7, metadata={},
            ),
        ]
        assembler = ContextAssembler(max_tokens=4000)
        ctx = assembler.assemble(results=results)
        assert isinstance(ctx, RAGContext)
        assert len(ctx.assembled_text) > 0

    def test_assemble_respects_token_budget(self):
        from ayextractor.rag.retriever.context_assembler import ContextAssembler
        results = [
            SearchResult(
                source_type="chunk", source_id=f"c{i}",
                content="x " * 500, score=0.5, metadata={},
            )
            for i in range(20)
        ]
        assembler = ContextAssembler(max_tokens=100)
        ctx = assembler.assemble(results=results, max_tokens=100)
        assert isinstance(ctx, RAGContext)
        assert ctx.total_token_count <= 200


# =====================================================================
#  CHUNK RETRIEVAL via vector store (direct test)
# =====================================================================

class TestChunkRetrieval:

    @pytest.mark.asyncio
    async def test_retrieve_similar_chunks(self, qdrant_memory_store, mock_embedder, unique_collection):
        store = qdrant_memory_store
        dims = mock_embedder.dimensions
        await store.create_collection(unique_collection, dimensions=dims)
        texts = ["AI safety research", "Machine learning models", "Cooking recipes"]
        embeddings = await mock_embedder.embed_texts(texts)
        await store.upsert(
            collection=unique_collection,
            ids=["c1", "c2", "c3"],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source_type": "chunk"}] * 3,
        )
        query_vec = await mock_embedder.embed_query("AI safety")
        results = await store.query(unique_collection, query_vec, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)


# =====================================================================
#  RAG MODELS
# =====================================================================

class TestRAGModels:

    def test_search_result_fields(self):
        sr = SearchResult(
            source_type="chunk", source_id="test",
            content="test content", score=0.85, metadata={"key": "value"},
        )
        assert sr.score == 0.85
        assert sr.source_type == "chunk"

    def test_search_result_valid_types(self):
        for st in ("chunk", "entity_profile", "relation_profile", "community_summary"):
            sr = SearchResult(source_type=st, source_id="x", content="y", score=0.5)
            assert sr.source_type == st

    def test_rag_context_creation(self):
        ctx = RAGContext(assembled_text="relevant context here")
        assert ctx.assembled_text == "relevant context here"
        assert ctx.chunk_excerpts == []
        assert ctx.entity_profiles == []
        assert ctx.community_summaries == []

    def test_rag_context_full(self):
        ctx = RAGContext(
            assembled_text="full context",
            chunk_excerpts=["chunk 1", "chunk 2"],
            entity_profiles=["entity A profile"],
            community_summaries=["community summary"],
            contradictions=["contradiction 1"],
            total_token_count=500,
        )
        assert len(ctx.chunk_excerpts) == 2
        assert ctx.total_token_count == 500


# =====================================================================
#  EMBEDDER FACTORY
# =====================================================================

class TestEmbedderFactory:

    def test_create_ollama_embedder(self):
        from ayextractor.rag.embeddings.embedder_factory import create_embedder
        settings = Settings(embedding_provider="ollama", embedding_model="nomic-embed-text")
        embedder = create_embedder(settings)
        assert embedder.provider_name == "ollama"

    def test_unsupported_provider_raises(self):
        from ayextractor.rag.embeddings.embedder_factory import (
            UnsupportedEmbeddingProviderError, create_embedder,
        )
        settings = Settings(embedding_provider="nonexistent_provider")
        with pytest.raises((ValueError, UnsupportedEmbeddingProviderError)):
            create_embedder(settings)


# =====================================================================
#  ENRICHER
# =====================================================================

class TestRAGEnricher:

    def test_should_enrich_disabled(self):
        from ayextractor.rag.enricher import should_enrich
        assert should_enrich("summarizer", rag_enabled=False) is False

    def test_should_enrich_enabled_matching_agent(self):
        from ayextractor.rag.enricher import should_enrich
        assert should_enrich(
            "concept_extractor", rag_enabled=True,
            enrich_agents={"concept_extractor", "synthesizer"},
        ) is True

    def test_should_enrich_enabled_non_matching_agent(self):
        from ayextractor.rag.enricher import should_enrich
        assert should_enrich(
            "summarizer", rag_enabled=True,
            enrich_agents={"concept_extractor"},
        ) is False

    @pytest.mark.asyncio
    async def test_build_context_no_stores(self):
        from ayextractor.rag.enricher import build_context
        from ayextractor.pipeline.state import PipelineState
        state = PipelineState(document_title="Test Doc")
        ctx = await build_context(
            agent_name="summarizer", state=state,
            vector_store=None, graph_store=None, embedder=None,
        )
        assert ctx is None or isinstance(ctx, RAGContext)