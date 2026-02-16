# src/rag/enricher.py — v1
"""RAG enricher — query stores and inject context into agent prompts.

Orchestrator calls this module before each enrichable agent to build
a RAGContext from available vector/graph stores. The enricher internally
uses the hierarchical retrieval pipeline (§26.6).

See spec §26.4.1 for enrichment injection mechanism.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ayextractor.rag.models import RAGContext

if TYPE_CHECKING:
    from ayextractor.config.settings import Settings
    from ayextractor.pipeline.state import PipelineState
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Default agents eligible for RAG enrichment (spec §26.4)
DEFAULT_ENRICH_AGENTS = {
    "decontextualizer",
    "concept_extractor",
    "reference_extractor",
    "synthesizer",
}


async def build_context(
    agent_name: str,
    state: PipelineState,
    vector_store: BaseVectorStore | None = None,
    graph_store: BaseGraphStore | None = None,
    embedder: BaseEmbedder | None = None,
    knowledge_graph: Any | None = None,
    settings: Settings | None = None,
) -> RAGContext | None:
    """Build RAGContext for a specific agent.

    Derives a search query from the current pipeline state and agent type,
    then runs the hierarchical retrieval pipeline.

    Args:
        agent_name: Name of the agent being enriched.
        state: Current PipelineState with extraction context.
        vector_store: Vector store for similarity search (may be None).
        graph_store: Graph store for corpus retrieval (may be None).
        embedder: Embedder for query vectorization (may be None).
        knowledge_graph: nx.Graph for PPR scoring (may be None).
        settings: Application settings.

    Returns:
        RAGContext if enrichment succeeds, None otherwise.
    """
    if not vector_store and not graph_store:
        logger.debug("No stores available for enrichment of '%s'", agent_name)
        return None

    if not embedder and vector_store:
        logger.warning("Embedder required for vector store enrichment of '%s'", agent_name)
        return None

    # Derive query from state + agent type
    query = _derive_query(agent_name, state)
    if not query:
        logger.debug("No query derived for agent '%s'", agent_name)
        return None

    # Determine retrieval config
    consolidator_enabled = False
    config: dict[str, Any] = {}
    if settings:
        consolidator_enabled = getattr(settings, "CONSOLIDATOR_ENABLED", False)
        config = {
            "top_k_communities": getattr(settings, "RAG_RETRIEVAL_TOP_K_COMMUNITIES", 5),
            "top_k_entities": getattr(settings, "RAG_RETRIEVAL_TOP_K_ENTITIES", 20),
            "top_k_chunks": getattr(settings, "RAG_RETRIEVAL_TOP_K_CHUNKS", 10),
            "chunk_fallback_threshold": getattr(settings, "RAG_CHUNK_FALLBACK_THRESHOLD", 0.6),
            "composite_weight": getattr(settings, "RAG_COMPOSITE_WEIGHT", 0.3),
            "context_token_budget": getattr(settings, "RAG_CONTEXT_TOKEN_BUDGET", 4000),
        }

    # Use retrieval pipeline
    from ayextractor.rag.retriever.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        embedder=embedder,
        graph_store=graph_store,
        knowledge_graph=knowledge_graph,
        config=config,
    )

    try:
        rag_context = await pipeline.retrieve(
            query=query,
            consolidator_enabled=consolidator_enabled,
        )
        logger.info(
            "RAG enrichment for '%s': ~%d tokens, %d results",
            agent_name,
            rag_context.total_token_count,
            len(rag_context.search_results),
        )
        return rag_context
    except Exception:
        logger.exception("RAG enrichment failed for agent '%s'", agent_name)
        return None


def should_enrich(
    agent_name: str,
    rag_enabled: bool = False,
    enrich_agents: set[str] | None = None,
) -> bool:
    """Check whether an agent should receive RAG enrichment.

    Args:
        agent_name: Agent name to check.
        rag_enabled: Whether RAG is enabled globally.
        enrich_agents: Set of agent names eligible for enrichment.

    Returns:
        True if the agent should be enriched.
    """
    if not rag_enabled:
        return False
    agents = enrich_agents or DEFAULT_ENRICH_AGENTS
    return agent_name in agents


def _derive_query(agent_name: str, state: Any) -> str:
    """Derive a search query from pipeline state based on agent type.

    Each agent needs different context:
    - decontextualizer: entity definitions, acronym resolutions
    - concept_extractor: canonical names, domain knowledge
    - reference_extractor: document citations, bibliography
    - synthesizer: cross-document connections, themes
    """
    parts: list[str] = []

    # Use dense_summary as base context
    if hasattr(state, "dense_summary") and state.dense_summary:
        parts.append(state.dense_summary[:500])

    # Agent-specific query augmentation
    if agent_name == "decontextualizer":
        # Focus on entities and references for disambiguation
        if hasattr(state, "references") and state.references:
            ref_names = [r.title for r in state.references[:5] if hasattr(r, "title")]
            if ref_names:
                parts.append("references: " + ", ".join(ref_names))

    elif agent_name == "concept_extractor":
        # Focus on entities already found
        if hasattr(state, "qualified_triplets") and state.qualified_triplets:
            entities = set()
            for t in state.qualified_triplets[:20]:
                if hasattr(t, "subject"):
                    entities.add(t.subject)
                if hasattr(t, "object"):
                    entities.add(t.object)
            if entities:
                parts.append("entities: " + ", ".join(list(entities)[:10]))

    elif agent_name == "synthesizer":
        # Broad query for thematic context
        if hasattr(state, "refine_summary") and state.refine_summary:
            parts.append(state.refine_summary[:300])

    elif agent_name == "reference_extractor":
        # Focus on citations
        if hasattr(state, "references") and state.references:
            parts.append("citations bibliography references")

    return " ".join(parts).strip()
