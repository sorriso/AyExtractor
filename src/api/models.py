# src/api/models.py — v1
"""API-level models: DocumentInput, Metadata, ConfigOverrides, AnalysisResult.

See spec §2.2 for full documentation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ayextractor.core.models import Concept, Relation, Theme


class ConfigOverrides(BaseModel):
    """Per-document overrides — validated subset of Settings (spec §2.2)."""

    llm_assignments: dict[str, str] | None = None
    chunking_strategy: str | None = None
    chunk_target_size: int | None = None
    chunk_overlap: int | None = None
    density_iterations: int | None = None
    decontextualization_enabled: bool | None = None
    critic_agent_enabled: bool | None = None
    output_format: str | None = None
    entity_similarity_threshold: float | None = None
    relation_taxonomy_extensible: bool | None = None
    community_detection_resolution: float | None = None
    community_detection_seed: int | None = None
    community_summary_enabled: bool | None = None
    profile_generation_enabled: bool | None = None
    consolidator_enabled: bool | None = None


class DocumentInput(BaseModel):
    """Input document for analysis."""

    content: bytes | str | Path | list[Path]
    format: str
    filename: str


class Metadata(BaseModel):
    """Execution metadata provided by the caller."""

    document_id: str | None = None
    document_type: str = "report"
    output_path: Path = Path("./output")
    language: str | None = None
    resume_from_run: str | None = None
    resume_from_step: int | None = None
    config_overrides: ConfigOverrides | None = None


class AnalysisResult(BaseModel):
    """Return value of facade.analyze() — API-level result."""

    document_id: str
    run_id: str
    summary: str
    themes: list[Theme] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    community_count: int = 0
    graph_path: Path
    communities_path: Path
    profiles_path: Path
    output_dir: Path
    run_dir: Path
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    fingerprint: object = None  # DocumentFingerprint — resolved at runtime
    usage_stats: object = None  # SessionStats — resolved at runtime
