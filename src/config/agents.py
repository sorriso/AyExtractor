# src/config/agents.py — v1
"""Declarative agent registry configuration.

Lists all agents participating in the LangGraph DAG (Phase 3).
The decontextualizer is NOT listed here — it runs in Phase 2 pre-processing.
See spec §25.4 for details.
"""

from __future__ import annotations

# Fully qualified class paths for dynamic import by plugin_kit/registry.py.
AGENT_REGISTRY: list[str] = [
    "ayextractor.pipeline.agents.summarizer.SummarizerAgent",
    "ayextractor.pipeline.agents.densifier.DensifierAgent",
    "ayextractor.pipeline.agents.concept_extractor.ConceptExtractorAgent",
    "ayextractor.pipeline.agents.reference_extractor.ReferenceExtractorAgent",
    "ayextractor.pipeline.agents.community_summarizer.CommunitySummarizerAgent",
    "ayextractor.pipeline.agents.profile_generator.ProfileGeneratorAgent",
    "ayextractor.pipeline.agents.synthesizer.SynthesizerAgent",
    # Optional agents (loaded only if enabled in settings)
    "ayextractor.pipeline.agents.critic.CriticAgent",
]

# Phase-to-component mapping for LLM routing (spec §17.3).
PHASE_COMPONENT_MAP: dict[str, list[str]] = {
    "extraction": ["image_analyzer", "reference_extractor"],
    "chunking": ["summarizer", "densifier", "decontextualizer"],
    "analysis": [
        "concept_extractor",
        "community_summarizer",
        "profile_generator",
        "synthesizer",
        "critic",
    ],
    "normalization": ["entity_normalizer", "relation_normalizer"],
}
