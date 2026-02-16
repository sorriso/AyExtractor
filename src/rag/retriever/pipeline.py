# src/rag/retriever/pipeline.py — v1
"""Retrieval pipeline — orchestrate hierarchical retrieval.

Coordinates the multi-level retrieval flow:
  1. Query classification → RetrievalPlan
  2. Level 1: Community retrieval
  3. Level 2: Entity retrieval (with PPR)
  4. Level 3: Chunk retrieval (fallback if Level 2 low confidence)
  5. Cross-doc: Corpus retrieval (if CONSOLIDATOR_ENABLED)
  6. Context assembly → RAGContext

See spec §26.6 for full documentation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ayextractor.rag.models import CorpusContext, RAGContext, SearchResult
from ayextractor.rag.retriever.context_assembler import ContextAssembler
from ayextractor.rag.retriever.query_classifier import classify_query

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Hierarchical retrieval pipeline: community → entity → chunk → corpus.

    Args:
        vector_store: Vector store for similarity search.
        embedder: Embedding model for query vectorization.
        graph_store: Graph store for corpus retrieval (optional).
        knowledge_graph: nx.Graph for PPR scoring (optional).
        config: Retrieval configuration overrides.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        embedder: BaseEmbedder | None = None,
        graph_store: BaseGraphStore | None = None,
        knowledge_graph: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._graph_store = graph_store
        self._knowledge_graph = knowledge_graph
        self._config = config or {}

        self._assembler = ContextAssembler(
            max_tokens=self._config.get("context_token_budget", 4000)
        )

    async def retrieve(
        self,
        query: str,
        consolidator_enabled: bool = False,
    ) -> RAGContext:
        """Execute full hierarchical retrieval pipeline.

        Args:
            query: Search query derived from agent context.
            consolidator_enabled: Whether to include Corpus Graph retrieval.

        Returns:
            Assembled RAGContext ready for prompt injection.
        """
        # Step 1: Classify query
        plan = classify_query(query)
        logger.info("Query classified as '%s', levels: %s", plan.query_type, plan.levels_to_query)

        all_results: list[SearchResult] = []
        corpus_context: CorpusContext | None = None
        contradictions: list[str] = []

        # Step 2: Execute retrieval levels
        if "community" in plan.levels_to_query:
            results = await self._retrieve_communities(query)
            all_results.extend(results)

        entity_results: list[SearchResult] = []
        if "entity" in plan.levels_to_query:
            entity_results = await self._retrieve_entities(query)
            all_results.extend(entity_results)

        if "chunk" in plan.levels_to_query:
            # Only retrieve chunks if entity confidence is low
            from ayextractor.rag.retriever.chunk_retriever import should_retrieve_chunks

            threshold = self._config.get("chunk_fallback_threshold", 0.6)
            if should_retrieve_chunks(entity_results, threshold):
                chunks = await self._retrieve_chunks(query)
                all_results.extend(chunks)

        if "corpus" in plan.levels_to_query and consolidator_enabled:
            corpus_context = await self._retrieve_corpus(query, entity_results)
            # Extract contradictions if available
            if self._graph_store and hasattr(self._graph_store, "get_contradictions"):
                raw_contradictions = self._graph_store.get_contradictions()
                contradictions = [str(c) for c in raw_contradictions]

        # Step 3: Assemble context
        rag_context = self._assembler.assemble(
            results=all_results,
            corpus_context=corpus_context,
            contradictions=contradictions,
        )

        logger.info(
            "Retrieval complete: %d total results, ~%d tokens",
            len(all_results), rag_context.total_token_count,
        )
        return rag_context

    # ------------------------------------------------------------------
    # Level-specific retrieval
    # ------------------------------------------------------------------

    async def _retrieve_communities(self, query: str) -> list[SearchResult]:
        """Level 1: Community summary retrieval."""
        if not self._vector_store or not self._embedder:
            return []

        from ayextractor.rag.retriever.community_retriever import retrieve_communities

        top_k = self._config.get("top_k_communities", 5)
        return await retrieve_communities(
            query=query,
            vector_store=self._vector_store,
            embedder=self._embedder,
            top_k=top_k,
        )

    async def _retrieve_entities(self, query: str) -> list[SearchResult]:
        """Level 2: Entity/relation profile retrieval with PPR."""
        if not self._vector_store or not self._embedder:
            return []

        from ayextractor.rag.retriever.entity_retriever import retrieve_entities

        top_k = self._config.get("top_k_entities", 20)
        composite_weight = self._config.get("composite_weight", 0.3)
        return await retrieve_entities(
            query=query,
            vector_store=self._vector_store,
            embedder=self._embedder,
            knowledge_graph=self._knowledge_graph,
            top_k=top_k,
            composite_weight=composite_weight,
        )

    async def _retrieve_chunks(self, query: str) -> list[SearchResult]:
        """Level 3: Source chunk retrieval (fallback evidence)."""
        if not self._vector_store or not self._embedder:
            return []

        from ayextractor.rag.retriever.chunk_retriever import retrieve_chunks

        top_k = self._config.get("top_k_chunks", 10)
        return await retrieve_chunks(
            query=query,
            vector_store=self._vector_store,
            embedder=self._embedder,
            top_k=top_k,
        )

    async def _retrieve_corpus(
        self,
        query: str,
        entity_results: list[SearchResult],
    ) -> CorpusContext | None:
        """Cross-doc: Corpus Graph retrieval."""
        if not self._graph_store:
            return None

        from ayextractor.rag.retriever.corpus_retriever import retrieve_corpus_context

        seed_entities = [r.source_id for r in entity_results[:5]]
        return await retrieve_corpus_context(
            query=query,
            graph_store=self._graph_store,
            seed_entities=seed_entities,
        )
