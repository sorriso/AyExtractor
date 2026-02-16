# src/rag/retriever/context_assembler.py — v2
"""Context assembler — build RAGContext from search results with token budgeting.

Takes search results from hierarchical retriever pipeline and assembles
a structured RAGContext ready for injection into agent prompts.

Supports CorpusContext (cross-document knowledge) and contradiction signaling.

See spec §26.6.4 for assembly logic and token budget rules.
"""

from __future__ import annotations

import logging
from typing import Any

from ayextractor.rag.models import CorpusContext, RAGContext, SearchResult

logger = logging.getLogger(__name__)

# Rough chars-to-tokens ratio
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 4000


class ContextAssembler:
    """Assemble RAGContext from search results.

    Args:
        max_tokens: Maximum token budget for assembled context.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._max_tokens = max_tokens

    def assemble(
        self,
        results: list[SearchResult],
        max_tokens: int | None = None,
        corpus_context: CorpusContext | None = None,
        contradictions: list[str] | None = None,
    ) -> RAGContext:
        """Assemble RAGContext from ranked search results.

        Results are already sorted by score (highest first).
        We greedily add content until the token budget is exhausted.

        Args:
            results: Ranked SearchResult list from retrieval pipeline.
            max_tokens: Override default token budget.
            corpus_context: Cross-document knowledge (if CONSOLIDATOR_ENABLED).
            contradictions: Contradiction descriptions from Corpus Graph.

        Returns:
            Populated RAGContext.
        """
        budget = max_tokens or self._max_tokens
        budget_chars = budget * CHARS_PER_TOKEN

        community_summaries: list[str] = []
        entity_profiles: list[str] = []
        chunk_excerpts: list[str] = []
        used_chars = 0
        included_results: list[SearchResult] = []

        for result in results:
            content_len = len(result.content)
            if used_chars + content_len > budget_chars:
                remaining = budget_chars - used_chars
                if remaining > 100:
                    truncated = result.content[:remaining] + "..."
                    _route_result(
                        result, truncated,
                        community_summaries, entity_profiles, chunk_excerpts,
                    )
                    included_results.append(result)
                    used_chars += len(truncated)
                break

            _route_result(
                result, result.content,
                community_summaries, entity_profiles, chunk_excerpts,
            )
            included_results.append(result)
            used_chars += content_len

        # Build assembled text per spec §26.6.4 structure
        parts: list[str] = []
        if community_summaries:
            parts.append("## Document Context\n" + "\n\n".join(community_summaries))

        if corpus_context and (corpus_context.cnodes or corpus_context.tnodes or corpus_context.xedges):
            corpus_text = _format_corpus_context(corpus_context)
            corpus_len = len(corpus_text)
            if used_chars + corpus_len <= budget_chars:
                parts.append("## Cross-Document Knowledge\n" + corpus_text)
                used_chars += corpus_len

        if entity_profiles:
            parts.append("## Relevant Entities\n" + "\n\n".join(entity_profiles))

        if chunk_excerpts:
            parts.append("## Evidence\n" + "\n\n".join(chunk_excerpts))

        if contradictions:
            contradiction_text = "\n".join(f"- {c}" for c in contradictions)
            contra_len = len(contradiction_text)
            if used_chars + contra_len <= budget_chars:
                parts.append("## Contradictions\n" + contradiction_text)
                used_chars += contra_len

        assembled = "\n\n".join(parts) if parts else ""
        total_tokens = used_chars // CHARS_PER_TOKEN

        logger.info(
            "Context assembled: %d results, ~%d tokens (budget=%d)",
            len(included_results), total_tokens, budget,
        )

        return RAGContext(
            assembled_text=assembled,
            community_summaries=community_summaries,
            entity_profiles=entity_profiles,
            chunk_excerpts=chunk_excerpts,
            corpus_context=corpus_context,
            contradictions=contradictions or [],
            total_token_count=total_tokens,
            search_results=included_results,
        )


def _route_result(
    result: SearchResult,
    content: str,
    community_summaries: list[str],
    entity_profiles: list[str],
    chunk_excerpts: list[str],
) -> None:
    """Route content to the appropriate bucket based on source_type."""
    if result.source_type == "community_summary":
        community_summaries.append(content)
    elif result.source_type in ("entity_profile", "relation_profile"):
        entity_profiles.append(content)
    else:
        chunk_excerpts.append(content)


def _format_corpus_context(ctx: CorpusContext) -> str:
    """Format CorpusContext into readable text for prompt injection."""
    lines: list[str] = []

    if ctx.cnodes:
        lines.append(f"Canonical entities from {ctx.source_document_count} documents:")
        for c in ctx.cnodes[:10]:
            attrs = f" — {c.consolidated_attributes}" if c.consolidated_attributes else ""
            lines.append(f"  - {c.canonical_name} ({c.entity_type}, corroboration={c.corroboration}){attrs}")

    if ctx.tnodes:
        lines.append("Taxonomy categories:")
        for t in ctx.tnodes[:5]:
            members = ", ".join(t.classified_cnodes[:5])
            lines.append(f"  - {t.canonical_name}: {members}")

    if ctx.xedges:
        lines.append("Cross-document relations:")
        for e in ctx.xedges[:10]:
            lines.append(f"  - {e.source} —[{e.relation_type}]→ {e.target} (conf={e.confidence:.2f})")

    return "\n".join(lines)
