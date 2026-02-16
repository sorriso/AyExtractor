# src/graph/entity_linker.py — v1
"""Cross-document entity linker — resolve entities across multiple runs.

Matches entities from a new extraction run against an existing knowledge
graph. Uses embedding similarity + LLM validation for ambiguous matches.

See spec §13.3.2 for cross-document linking rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ayextractor.core.models import EntityNormalization
from ayextractor.core.similarity import cosine_similarity_matrix

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_LINK_THRESHOLD = 0.88


@dataclass
class LinkResult:
    """Result of entity linking between existing and new entities."""

    matched: dict[str, str] = field(default_factory=dict)  # new -> existing canonical
    new_entities: list[str] = field(default_factory=list)  # genuinely new
    stats: dict[str, int] = field(default_factory=dict)


async def link_entities(
    existing: list[EntityNormalization],
    incoming: list[EntityNormalization],
    embedder: BaseEmbedder | None = None,
    threshold: float = DEFAULT_LINK_THRESHOLD,
) -> LinkResult:
    """Link incoming entities to existing knowledge graph entities.

    Strategy:
      1. Exact canonical name match -> immediate link.
      2. Alias overlap -> immediate link.
      3. Embedding similarity above threshold -> candidate link.
      4. No match -> mark as new entity.

    Args:
        existing: Entities already in the knowledge graph.
        incoming: Entities from a new extraction run.
        embedder: Embedding provider for similarity check.
        threshold: Cosine threshold for embedding-based matching.

    Returns:
        LinkResult with matched and new entities.
    """
    if not incoming:
        return LinkResult(stats={"total_incoming": 0})
    if not existing:
        return LinkResult(
            new_entities=[e.canonical_name for e in incoming],
            stats={"total_incoming": len(incoming), "new": len(incoming)},
        )

    # Build lookup indexes
    canonical_index: dict[str, str] = {}
    alias_index: dict[str, str] = {}
    for e in existing:
        canonical_index[e.canonical_name.lower()] = e.canonical_name
        for alias in e.aliases:
            alias_index[alias.lower()] = e.canonical_name

    matched: dict[str, str] = {}
    unresolved: list[EntityNormalization] = []

    # Pass 1: exact + alias match
    for inc in incoming:
        inc_lower = inc.canonical_name.lower()
        if inc_lower in canonical_index:
            matched[inc.canonical_name] = canonical_index[inc_lower]
            continue
        if inc_lower in alias_index:
            matched[inc.canonical_name] = alias_index[inc_lower]
            continue
        # Check incoming aliases against existing canonical/alias
        found = False
        for alias in inc.aliases:
            alias_lower = alias.lower()
            if alias_lower in canonical_index:
                matched[inc.canonical_name] = canonical_index[alias_lower]
                found = True
                break
            if alias_lower in alias_index:
                matched[inc.canonical_name] = alias_index[alias_lower]
                found = True
                break
        if not found:
            unresolved.append(inc)

    # Pass 2: embedding similarity for unresolved
    if unresolved and embedder is not None:
        existing_names = [e.canonical_name for e in existing]
        unresolved_names = [e.canonical_name for e in unresolved]

        all_names = existing_names + unresolved_names
        embeddings = await embedder.embed_texts(all_names)
        emb_array = np.array(embeddings, dtype=np.float64)

        n_existing = len(existing_names)
        existing_emb = emb_array[:n_existing]
        unresolved_emb = emb_array[n_existing:]

        sim = cosine_similarity_matrix(
            np.vstack([existing_emb, unresolved_emb])
        )

        still_new: list[str] = []
        for i, ent in enumerate(unresolved):
            row_idx = n_existing + i
            best_j = -1
            best_sim = 0.0
            for j in range(n_existing):
                if sim[row_idx, j] > best_sim:
                    best_sim = sim[row_idx, j]
                    best_j = j
            if best_sim >= threshold and best_j >= 0:
                matched[ent.canonical_name] = existing_names[best_j]
            else:
                still_new.append(ent.canonical_name)

        new_entities = still_new
    else:
        new_entities = [e.canonical_name for e in unresolved]

    stats = {
        "total_incoming": len(incoming),
        "matched_exact": len(matched) - len([m for m in matched if m in [e.canonical_name for e in unresolved]]),
        "matched_embedding": len(matched) - (len(incoming) - len(unresolved)),
        "new": len(new_entities),
    }

    logger.info(
        "Entity linking: %d incoming -> %d matched, %d new",
        len(incoming), len(matched), len(new_entities),
    )

    return LinkResult(matched=matched, new_entities=new_entities, stats=stats)
