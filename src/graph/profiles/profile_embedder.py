# src/graph/profiles/profile_embedder.py — v1
"""Profile embedder — compute embeddings for entity and relation profiles.

Pure function (no LLM, just embedding provider). Returns new profile
objects with the embedding field set. Does NOT modify originals.

See spec §13.11.3 for full documentation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ayextractor.graph.profiles.models import EntityProfile, RelationProfile

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


async def embed_entity_profiles(
    profiles: list[EntityProfile],
    embedder: BaseEmbedder,
) -> list[EntityProfile]:
    """Compute embeddings for entity profiles.

    Args:
        profiles: Entity profiles with profile_text set.
        embedder: Embedding provider.

    Returns:
        New EntityProfile list with embedding field populated.
    """
    if not profiles:
        return []

    texts = [p.profile_text for p in profiles]
    embeddings = await embedder.embed_texts(texts)

    result: list[EntityProfile] = []
    for profile, embedding in zip(profiles, embeddings):
        result.append(profile.model_copy(update={"embedding": embedding}))

    logger.info("Embedded %d entity profiles", len(result))
    return result


async def embed_relation_profiles(
    profiles: list[RelationProfile],
    embedder: BaseEmbedder,
) -> list[RelationProfile]:
    """Compute embeddings for relation profiles.

    Args:
        profiles: Relation profiles with profile_text set.
        embedder: Embedding provider.

    Returns:
        New RelationProfile list with embedding field populated.
    """
    if not profiles:
        return []

    texts = [p.profile_text for p in profiles]
    embeddings = await embedder.embed_texts(texts)

    result: list[RelationProfile] = []
    for profile, embedding in zip(profiles, embeddings):
        result.append(profile.model_copy(update={"embedding": embedding}))

    logger.info("Embedded %d relation profiles", len(result))
    return result


async def embed_profiles(
    entity_profiles: list[EntityProfile],
    relation_profiles: list[RelationProfile],
    embedder: BaseEmbedder,
) -> tuple[list[EntityProfile], list[RelationProfile]]:
    """Convenience: embed both entity and relation profiles in one call.

    Returns:
        Tuple of (embedded_entity_profiles, embedded_relation_profiles).
    """
    entities = await embed_entity_profiles(entity_profiles, embedder)
    relations = await embed_relation_profiles(relation_profiles, embedder)
    return entities, relations
