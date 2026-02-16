# src/rag/indexer.py — v1
"""RAG indexer — post-analysis indexing into stores.

After each successful analysis, indexes:
- Vector DB: chunks, entity_profiles, relation_profiles, community_summaries
- Graph DB: Document Graph (L1/L2/L3 + qualified edges)

If CONSOLIDATOR_ENABLED: triggers Pass 1 (Linking) after graph import.

See spec §26.5 for full documentation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ayextractor.config.settings import Settings
    from ayextractor.pipeline.state import PipelineState
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Collection names matching spec §26.5
COLLECTION_CHUNKS = "chunks"
COLLECTION_ENTITY_PROFILES = "entity_profiles"
COLLECTION_RELATION_PROFILES = "relation_profiles"
COLLECTION_COMMUNITY_SUMMARIES = "community_summaries"


async def index_analysis_results(
    state: PipelineState,
    vector_store: BaseVectorStore | None = None,
    graph_store: BaseGraphStore | None = None,
    embedder: BaseEmbedder | None = None,
    settings: Settings | None = None,
    document_graph: Any | None = None,
) -> IndexingReport:
    """Index analysis results into configured stores.

    Args:
        state: Completed PipelineState with all analysis outputs.
        vector_store: Vector store for similarity indexing (may be None).
        graph_store: Graph store for graph import (may be None).
        embedder: Embedder for generating vectors.
        settings: Application settings.
        document_graph: nx.Graph of the Document Graph.

    Returns:
        IndexingReport with counts and status.
    """
    report = IndexingReport()

    # Vector DB indexing
    if vector_store and embedder:
        await _index_vector_db(state, vector_store, embedder, report)

    # Graph DB indexing
    if graph_store and document_graph:
        await _index_graph_db(document_graph, graph_store, report)

        # Trigger consolidator Pass 1 if enabled
        consolidator_enabled = False
        if settings:
            consolidator_enabled = getattr(settings, "CONSOLIDATOR_ENABLED", False)

        if consolidator_enabled:
            _run_linking_pass(document_graph, graph_store, settings)
            report.consolidator_linked = True

    logger.info(
        "Indexing complete: %d chunks, %d entities, %d relations, %d communities indexed",
        report.chunks_indexed,
        report.entity_profiles_indexed,
        report.relation_profiles_indexed,
        report.community_summaries_indexed,
    )
    return report


class IndexingReport:
    """Report of indexing operations."""

    def __init__(self) -> None:
        self.chunks_indexed: int = 0
        self.entity_profiles_indexed: int = 0
        self.relation_profiles_indexed: int = 0
        self.community_summaries_indexed: int = 0
        self.graph_nodes_imported: int = 0
        self.graph_edges_imported: int = 0
        self.consolidator_linked: bool = False
        self.errors: list[str] = []


# ------------------------------------------------------------------
# Vector DB indexing
# ------------------------------------------------------------------


async def _index_vector_db(
    state: Any,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    report: IndexingReport,
) -> None:
    """Index all 4 collections into vector store."""
    # 1. Chunks
    if hasattr(state, "chunks") and state.chunks:
        for chunk in state.chunks:
            try:
                content = _get_chunk_content(chunk)
                if not content:
                    continue
                embedding = await embedder.embed(content)
                await vector_store.upsert(
                    collection=COLLECTION_CHUNKS,
                    id=chunk.chunk_id if hasattr(chunk, "chunk_id") else str(id(chunk)),
                    content=content,
                    vector=embedding,
                    metadata=_chunk_metadata(chunk),
                )
                report.chunks_indexed += 1
            except Exception as e:
                report.errors.append(f"chunk: {e}")

    # 2. Entity profiles
    if hasattr(state, "entity_profiles") and state.entity_profiles:
        for profile in state.entity_profiles:
            try:
                text = profile.profile_text if hasattr(profile, "profile_text") else str(profile)
                embedding = await embedder.embed(text)
                name = profile.canonical_name if hasattr(profile, "canonical_name") else str(id(profile))
                await vector_store.upsert(
                    collection=COLLECTION_ENTITY_PROFILES,
                    id=name,
                    content=text,
                    vector=embedding,
                    metadata={"entity_type": getattr(profile, "entity_type", "")},
                )
                report.entity_profiles_indexed += 1
            except Exception as e:
                report.errors.append(f"entity_profile: {e}")

    # 3. Relation profiles
    if hasattr(state, "relation_profiles") and state.relation_profiles:
        for profile in state.relation_profiles:
            try:
                text = profile.profile_text if hasattr(profile, "profile_text") else str(profile)
                embedding = await embedder.embed(text)
                rid = getattr(profile, "relation_id", str(id(profile)))
                await vector_store.upsert(
                    collection=COLLECTION_RELATION_PROFILES,
                    id=rid,
                    content=text,
                    vector=embedding,
                    metadata={},
                )
                report.relation_profiles_indexed += 1
            except Exception as e:
                report.errors.append(f"relation_profile: {e}")

    # 4. Community summaries
    if hasattr(state, "community_summaries") and state.community_summaries:
        for cs in state.community_summaries:
            try:
                text = cs.summary if hasattr(cs, "summary") else str(cs)
                embedding = await embedder.embed(text)
                cid = cs.community_id if hasattr(cs, "community_id") else str(id(cs))
                await vector_store.upsert(
                    collection=COLLECTION_COMMUNITY_SUMMARIES,
                    id=cid,
                    content=text,
                    vector=embedding,
                    metadata={"level": getattr(cs, "level", 0)},
                )
                report.community_summaries_indexed += 1
            except Exception as e:
                report.errors.append(f"community_summary: {e}")


def _get_chunk_content(chunk: Any) -> str:
    """Extract indexable content from a chunk."""
    parts: list[str] = []
    if hasattr(chunk, "content") and chunk.content:
        parts.append(chunk.content)
    if hasattr(chunk, "context_summary") and chunk.context_summary:
        parts.append(chunk.context_summary)
    if hasattr(chunk, "global_summary") and chunk.global_summary:
        parts.append(chunk.global_summary)
    return " ".join(parts)


def _chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Extract metadata from a chunk for vector store."""
    meta: dict[str, Any] = {}
    for attr in ("document_id", "chunk_index", "section_title", "page_number"):
        if hasattr(chunk, attr):
            meta[attr] = getattr(chunk, attr)
    return meta


# ------------------------------------------------------------------
# Graph DB indexing
# ------------------------------------------------------------------


async def _index_graph_db(
    document_graph: Any,
    graph_store: BaseGraphStore,
    report: IndexingReport,
) -> None:
    """Import Document Graph into graph store."""
    try:
        # Import nodes
        for node_id, data in document_graph.nodes(data=True):
            if hasattr(graph_store, "upsert_node"):
                graph_store.upsert_node(str(node_id), data)
                report.graph_nodes_imported += 1

        # Import edges
        for u, v, data in document_graph.edges(data=True):
            if hasattr(graph_store, "upsert_edge"):
                graph_store.upsert_edge(str(u), str(v), data)
                report.graph_edges_imported += 1

    except Exception as e:
        report.errors.append(f"graph_import: {e}")
        logger.exception("Graph DB import failed")


def _run_linking_pass(
    document_graph: Any,
    graph_store: BaseGraphStore,
    settings: Any,
) -> None:
    """Trigger consolidator Pass 1 (Linking) synchronously."""
    try:
        from ayextractor.consolidator.orchestrator import ConsolidatorOrchestrator

        orchestrator = ConsolidatorOrchestrator(
            corpus_store=graph_store, settings=settings
        )
        result = orchestrator.run_linking(document_graph)
        logger.info(
            "Consolidator linking: %d items processed, %d modified",
            result.items_processed, result.items_modified,
        )
    except Exception:
        logger.exception("Consolidator linking pass failed")
