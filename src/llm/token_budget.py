# src/llm/token_budget.py — v1
"""Token budget estimation and allocation per agent.

See spec §15.1 for estimation formula and §15.2 for allocation strategy.
"""

from __future__ import annotations

import logging

from ayextractor.config.settings import Settings
from ayextractor.core.models import TokenBudget

logger = logging.getLogger(__name__)

# Rough cost multipliers per agent (tokens per input token)
_AGENT_COST_RATIOS: dict[str, float] = {
    "image_analyzer": 0.15,
    "reference_extractor": 0.10,
    "decontextualizer": 0.25,
    "summarizer": 0.20,
    "densifier": 0.05,
    "concept_extractor": 0.15,
    "entity_normalizer": 0.02,
    "relation_normalizer": 0.02,
    "community_summarizer": 0.02,
    "profile_generator": 0.02,
    "synthesizer": 0.02,
    "critic": 0.00,
}


def estimate_budget(
    text_tokens: int,
    n_chunks: int,
    n_images: int = 0,
    settings: Settings | None = None,
    critic_enabled: bool = False,
) -> TokenBudget:
    """Estimate total token budget for a document analysis.

    Args:
        text_tokens: Estimated tokens in the full extracted text.
        n_chunks: Number of chunks after chunking.
        n_images: Number of embedded images requiring Vision analysis.
        settings: Application settings (for density_iterations etc.).
        critic_enabled: Whether the critic agent is enabled.

    Returns:
        TokenBudget with per-agent allocations.
    """
    density_iter = 5 if settings is None else settings.density_iterations
    max_per_agent = 4096 if settings is None else settings.llm_max_tokens_per_agent

    per_agent: dict[str, int] = {}

    # Image analyzer: ~1000 tokens per image (vision)
    per_agent["image_analyzer"] = n_images * 1000

    # Reference extractor: process full text once
    per_agent["reference_extractor"] = min(text_tokens, max_per_agent)

    # Decontextualizer: 1 call per chunk with context window
    per_agent["decontextualizer"] = n_chunks * max_per_agent

    # Summarizer: 1 call per chunk (refine)
    per_agent["summarizer"] = n_chunks * max_per_agent

    # Densifier: density_iterations passes on summary
    per_agent["densifier"] = density_iter * max_per_agent

    # Concept extractor: 1 call per chunk
    per_agent["concept_extractor"] = n_chunks * max_per_agent

    # Normalizers: batch calls (~50 items per call)
    per_agent["entity_normalizer"] = max(1, n_chunks // 5) * max_per_agent
    per_agent["relation_normalizer"] = max(1, n_chunks // 10) * max_per_agent

    # Community + profile: proportional to graph size
    est_communities = max(1, n_chunks // 3)
    per_agent["community_summarizer"] = est_communities * max_per_agent
    per_agent["profile_generator"] = est_communities * 2 * max_per_agent

    # Synthesizer: single call
    per_agent["synthesizer"] = max_per_agent * 2

    # Critic: optional
    per_agent["critic"] = max_per_agent * 2 if critic_enabled else 0

    total = sum(per_agent.values())

    return TokenBudget(
        total_estimated=total,
        per_agent=per_agent,
    )


def check_budget(budget: TokenBudget, agent: str) -> bool:
    """Check if an agent has remaining budget.

    Returns True if within budget, False if over.
    """
    allocated = budget.per_agent.get(agent, 0)
    consumed = budget.consumed.get(agent, 0)
    if consumed > allocated:
        logger.warning(
            "Agent '%s' over budget: %d/%d tokens consumed",
            agent, consumed, allocated,
        )
        return False
    return True


def record_usage(budget: TokenBudget, agent: str, tokens: int) -> None:
    """Record token usage for an agent (mutates budget in place)."""
    budget.consumed[agent] = budget.consumed.get(agent, 0) + tokens
