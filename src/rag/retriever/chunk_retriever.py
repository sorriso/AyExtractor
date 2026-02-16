# src/rag/retriever/chunk_retriever.py — v1
"""Chunk retriever — Level 3 fallback of hierarchical retrieval pipeline.

Vector search on chunks collection. Only activated when Level 2 entity
confidence falls below CHUNK_FALLBACK_THRESHOLD.

See spec §26.6.1 for architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ayextractor.rag.models import SearchResult

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.2


async def retrieve_chunks(
    query: str,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[SearchResult]:
    """Retrieve source chunks by vector similarity (fallback evidence).

    Args:
        query: Search query text.
        vector_store: Vector store with chunks collection.
        embedder: Embedder to vectorize the query.
        top_k: Maximum results to return.
        min_score: Minimum similarity score threshold.

    Returns:
        Ranked list of SearchResult with source_type="chunk".
    """
    query_embedding = await embedder.embed(query)

    raw_results = await vector_store.search(
        collection="chunks",
        query_vector=query_embedding,
        top_k=top_k,
    )

    results: list[SearchResult] = []
    for item in raw_results:
        if isinstance(item, dict):
            score = item.get("score", 0.0)
            content = item.get("content", "")
            source_id = item.get("id", "")
            metadata = item.get("metadata", {})
        else:
            score = getattr(item, "score", 0.0)
            content = getattr(item, "content", "")
            source_id = getattr(item, "id", "")
            metadata = getattr(item, "metadata", {})

        if float(score) < min_score:
            continue

        results.append(
            SearchResult(
                source_type="chunk",
                source_id=str(source_id),
                content=str(content),
                score=float(score),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "Chunk retrieval: %d results (top_k=%d, min_score=%.2f)",
        len(results), top_k, min_score,
    )
    return results[:top_k]


def should_retrieve_chunks(
    entity_results: list[SearchResult],
    fallback_threshold: float = 0.6,
) -> bool:
    """Determine if chunk retrieval is needed based on entity confidence.

    Returns True if average entity score is below fallback_threshold,
    meaning Level 2 results are not confident enough.
    """
    if not entity_results:
        return True

    avg_score = sum(r.score for r in entity_results) / len(entity_results)
    return avg_score < fallback_threshold
