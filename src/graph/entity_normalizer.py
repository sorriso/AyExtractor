# src/graph/entity_normalizer.py — v2
"""Entity normalization — Pass 1 of the triplet consolidation pipeline.

Groups entity variants under canonical names using embedding similarity
clustering, optionally validated by LLM. Singletons skip LLM validation.

See spec §13.3 for full documentation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ayextractor.core.models import EntityNormalization, QualifiedTriplet
from ayextractor.core.similarity import cosine_similarity_matrix

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.85


def extract_unique_entities(
    triplets: list[QualifiedTriplet],
) -> dict[str, list[str]]:
    """Extract all unique entity names and their source chunk IDs.

    Returns:
        Mapping entity_name -> list of source_chunk_ids.
    """
    entities: dict[str, list[str]] = {}
    for t in triplets:
        for name in (t.subject, t.object):
            clean = name.strip()
            if clean:
                entities.setdefault(clean, []).append(t.source_chunk_id)
    return entities


def cluster_by_similarity(
    entity_names: list[str],
    embeddings: np.ndarray,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[list[int]]:
    """Cluster entities by cosine similarity above threshold.

    Simple agglomerative single-linkage clustering. Returns list of clusters,
    each cluster being a list of indices into entity_names.
    """
    n = len(entity_names)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    sim_matrix = cosine_similarity_matrix(embeddings)
    visited = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i, j] >= threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    return clusters


async def normalize_entities(
    triplets: list[QualifiedTriplet],
    embedder: BaseEmbedder | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[EntityNormalization]:
    """Run Pass 1: extract, cluster, and normalize entities.

    If no embedder is provided, each entity becomes its own canonical form
    (no clustering). Singletons always skip LLM validation.

    Args:
        triplets: Raw triplets from all chunks.
        embedder: Optional embedding provider for similarity clustering.
        threshold: Cosine similarity threshold for clustering.

    Returns:
        List of EntityNormalization entries.
    """
    entity_chunks = extract_unique_entities(triplets)
    names = sorted(entity_chunks.keys())

    if not names:
        return []

    # Without embedder: each entity is its own canonical form
    if embedder is None:
        logger.info("No embedder provided, skipping similarity clustering")
        return [
            EntityNormalization(
                canonical_name=name,
                aliases=[],
                occurrence_count=len(entity_chunks[name]),
                source_chunk_ids=sorted(set(entity_chunks[name])),
            )
            for name in names
        ]

    # Compute embeddings
    embeddings = await embedder.embed_texts(names)
    emb_array = np.array(embeddings, dtype=np.float64)

    # Cluster
    clusters = cluster_by_similarity(names, emb_array, threshold)

    # Build normalizations
    results: list[EntityNormalization] = []
    for cluster_indices in clusters:
        cluster_names = [names[i] for i in cluster_indices]
        # Choose canonical: longest name (most complete form)
        canonical = max(cluster_names, key=len)
        aliases = [n for n in cluster_names if n != canonical]

        all_chunks: list[str] = []
        for n in cluster_names:
            all_chunks.extend(entity_chunks[n])

        results.append(
            EntityNormalization(
                canonical_name=canonical,
                aliases=aliases,
                occurrence_count=len(all_chunks),
                source_chunk_ids=sorted(set(all_chunks)),
            )
        )

    logger.info(
        "Entity normalization: %d unique -> %d canonical (%d clusters with aliases)",
        len(names),
        len(results),
        sum(1 for r in results if r.aliases),
    )
    return results


def build_entity_mapping(
    normalizations: list[EntityNormalization],
) -> dict[str, str]:
    """Build alias -> canonical_name mapping for triplet substitution.

    Returns:
        Dict mapping every known form (canonical + aliases) to canonical_name.
    """
    mapping: dict[str, str] = {}
    for norm in normalizations:
        mapping[norm.canonical_name] = norm.canonical_name
        for alias in norm.aliases:
            mapping[alias] = norm.canonical_name
    return mapping
