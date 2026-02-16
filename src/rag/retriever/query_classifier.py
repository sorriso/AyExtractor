# src/rag/retriever/query_classifier.py — v1
"""Query classifier — determine retrieval strategy from user query.

Classifies query type (conceptual, factual, relational, exploratory)
and selects which retrieval levels to engage.

See spec §26.2 for classification rules.
"""

from __future__ import annotations

import re
from typing import Literal

from ayextractor.rag.models import RetrievalPlan

QueryType = Literal["conceptual", "factual", "relational", "exploratory"]

# Keyword patterns for heuristic classification
_FACTUAL_PATTERNS = [
    r"\bwho\b", r"\bwhat is\b", r"\bwhen\b", r"\bhow many\b",
    r"\bhow much\b", r"\bdefine\b", r"\bname\b",
]
_RELATIONAL_PATTERNS = [
    r"\brelat\w+\b", r"\bconnect\w+\b", r"\blink\w+\b",
    r"\bbetween\b", r"\bcause\b", r"\bimpact\b", r"\baffect\b",
    r"\bdepend\b", r"\binfluence\b",
]
_EXPLORATORY_PATTERNS = [
    r"\bsummar\w+\b", r"\boverview\b", r"\bexplain\b",
    r"\bmain\s+theme\b", r"\bkey\s+topic\b", r"\btell\s+me\s+about\b",
    r"\bwhat\s+are\s+the\b",
]


def classify_query(query: str) -> RetrievalPlan:
    """Classify query and produce a retrieval plan.

    Args:
        query: User query string.

    Returns:
        RetrievalPlan with query_type and levels_to_query.
    """
    q = query.lower().strip()

    if not q:
        return RetrievalPlan(
            query_type="exploratory",
            levels_to_query=["community"],
            estimated_token_cost=500,
        )

    query_type = _classify_type(q)
    levels = _select_levels(query_type)
    cost = _estimate_cost(levels)

    return RetrievalPlan(
        query_type=query_type,
        levels_to_query=levels,
        estimated_token_cost=cost,
    )


def _classify_type(q: str) -> QueryType:
    """Heuristic classification based on keyword patterns."""
    factual_score = sum(1 for p in _FACTUAL_PATTERNS if re.search(p, q))
    relational_score = sum(1 for p in _RELATIONAL_PATTERNS if re.search(p, q))
    exploratory_score = sum(1 for p in _EXPLORATORY_PATTERNS if re.search(p, q))

    scores = {
        "factual": factual_score,
        "relational": relational_score,
        "exploratory": exploratory_score,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]

    if scores[best] == 0:
        # No strong signal → conceptual (entity/chunk search)
        return "conceptual"
    return best  # type: ignore[return-value]


def _select_levels(
    query_type: QueryType,
) -> list[Literal["community", "entity", "chunk", "corpus"]]:
    """Select retrieval levels based on query type."""
    if query_type == "factual":
        return ["chunk", "entity"]
    elif query_type == "relational":
        return ["entity", "chunk", "corpus"]
    elif query_type == "exploratory":
        return ["community", "entity"]
    else:  # conceptual
        return ["entity", "chunk"]


def _estimate_cost(
    levels: list[str],
) -> int:
    """Rough token cost estimate per level."""
    cost_map = {
        "community": 500,
        "entity": 800,
        "chunk": 1200,
        "corpus": 600,
    }
    return sum(cost_map.get(l, 500) for l in levels)
