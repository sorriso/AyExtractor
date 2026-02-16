# src/config/settings.py — v1
"""Typed configuration loaded from .env via pydantic-settings.

Single source of truth for all deployment-specific settings.
See spec §17.1 for full env var documentation and §17.6 for validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(Exception):
    """Raised when configuration is internally inconsistent (spec §17.6)."""


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === LLM PROVIDERS ===
    llm_default_provider: str = "anthropic"
    llm_default_model: str = "claude-sonnet-4-20250514"
    llm_default_temperature: float = 0.2
    llm_max_tokens_per_agent: int = 4096

    # Provider API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Per-phase LLM assignment
    llm_phase_extraction: str = ""
    llm_phase_chunking: str = ""
    llm_phase_analysis: str = ""
    llm_phase_normalization: str = ""

    # Per-component LLM assignment (highest priority)
    llm_image_analyzer: str = ""
    llm_reference_extractor: str = ""
    llm_summarizer: str = ""
    llm_densifier: str = ""
    llm_decontextualizer: str = ""
    llm_concept_extractor: str = ""
    llm_community_summarizer: str = ""
    llm_profile_generator: str = ""
    llm_synthesizer: str = ""
    llm_critic: str = ""
    llm_entity_normalizer: str = ""
    llm_relation_normalizer: str = ""

    # === EMBEDDINGS ===
    embedding_provider: str = "anthropic"
    embedding_model: str = "voyage-3"
    embedding_dimensions: int = 1024
    embedding_ollama_model: str = "nomic-embed-text"
    embedding_st_model: str = "all-MiniLM-L6-v2"

    # === Document limits ===
    max_document_size_mb: int = 50
    max_document_pages: int = 2000
    max_document_tokens: int = 500_000
    oversize_strategy: Literal["reject", "truncate", "sample"] = "reject"

    # === Chunking ===
    chunking_strategy: Literal["structural", "semantic"] = "structural"
    chunk_target_size: int = 2000
    chunk_overlap: int = 0
    decontextualization_enabled: bool = True
    decontextualizer_tool_use: Literal["auto", "always", "never"] = "auto"
    decontextualizer_tool_confidence_threshold: float = 0.7

    # === Pipeline ===
    density_iterations: int = 5
    confidence_threshold: float = 0.6
    critic_agent_enabled: bool = False
    critic_strictness: Literal["low", "medium", "high"] = "medium"

    # === Triplet consolidation ===
    entity_similarity_threshold: float = 0.85
    relation_taxonomy_extensible: bool = True
    triplet_confidence_boost: bool = True

    # === Community detection ===
    community_detection_resolution: float = 1.0
    community_detection_seed: int | None = 42
    community_min_size: int = 3
    community_summary_enabled: bool = True

    # === Entity/Relation profiles ===
    profile_generation_enabled: bool = True
    profile_min_relations: int = 2

    # === Scoring weights (RAG composite) ===
    scoring_w_confidence: float = 0.3
    scoring_w_salience: float = 0.3
    scoring_w_freshness: float = 0.2
    scoring_w_corroboration: float = 0.2
    scoring_corroboration_cap: int = 5

    # === Chunk output mode ===
    chunk_output_mode: Literal[
        "files_only", "files_and_vectordb", "files_and_graphdb", "files_and_both_db"
    ] = "files_only"

    # === Cache ===
    cache_enabled: bool = True
    cache_backend: Literal["json", "sqlite", "redis", "arangodb"] = "json"
    cache_root: Path = Path("~/.ayextractor/cache")
    cache_redis_url: str = ""
    cache_arangodb_url: str = ""
    cache_arangodb_database: str = "ayextractor_cache"
    simhash_threshold: int = 3
    minhash_threshold: float = 0.8
    constellation_threshold: float = 0.7

    # === Output storage ===
    output_writer: Literal["local", "s3"] = "local"
    output_s3_bucket: str = ""
    output_s3_prefix: str = "ayextractor/"
    output_s3_region: str = ""

    # === Graph export ===
    graph_export_formats: str = "graphml"

    # === Batch scan ===
    batch_scan_enabled: bool = False
    batch_scan_root: str = ""
    batch_scan_recursive: bool = True
    batch_scan_formats: str = "pdf,epub,docx,md,txt,png,jpg,jpeg,webp"

    # === Output ===
    output_format: Literal["markdown", "json", "both"] = "both"

    # === Logging ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: Path = Path("~/.ayextractor/logs/analyzer.log")
    log_rotation: str = "10MB"
    log_retention: int = 30

    # === Vector database ===
    vector_db_type: Literal["none", "chromadb", "qdrant", "arangodb"] = "none"
    vector_db_path: Path = Path("~/.ayextractor/vectordb")
    vector_db_url: str = ""
    vector_db_api_key: str = ""
    vector_db_collection: str = "ayextractor"

    # === Graph database ===
    graph_db_type: Literal["none", "neo4j", "arangodb"] = "none"
    graph_db_uri: str = "bolt://localhost:7687"
    graph_db_database: str = "ayextractor"
    graph_db_user: str = ""
    graph_db_password: str = ""
    graph_db_merge_strategy: Literal["incremental", "replace"] = "incremental"

    # === RAG ===
    rag_enabled: bool = False
    rag_enrich_agents: str = (
        "decontextualizer,concept_extractor,reference_extractor,synthesizer"
    )
    rag_retrieval_top_k_communities: int = 5
    rag_retrieval_top_k_entities: int = 20
    rag_retrieval_top_k_chunks: int = 10
    rag_chunk_fallback_threshold: float = 0.6
    rag_composite_weight: float = 0.3
    rag_ppr_alpha: float = 0.15
    rag_context_token_budget: int = 4000
    rag_include_corpus_graph: bool = True

    # === Consolidator ===
    consolidator_enabled: bool = False
    consolidator_trigger: Literal["on_ingestion", "scheduled", "manual"] = (
        "on_ingestion"
    )
    consolidator_schedule: str = "0 3 * * 0"
    consolidator_passes: str = "linking,clustering,inference,decay,contradiction"
    consolidator_decay_halflife_days: int = 90
    consolidator_inference_min_confidence: float = 0.6
    consolidator_inference_discount: float = 0.8
    consolidator_prune_threshold: float = 0.2
    consolidator_cluster_min_size: int = 3

    # === GPU Acceleration (optional) ===
    gpu_clustering_backend: Literal["sklearn", "cuml"] = "sklearn"
    gpu_similarity_backend: Literal["sklearn", "cupy", "torch"] = "sklearn"

    # --- Validators ---

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:  # noqa: N805
        """V-05: CHUNK_OVERLAP must be non-negative."""
        if v < 0:
            raise ValueError("chunk_overlap must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_config_consistency(self) -> Settings:
        """Validate cross-field consistency rules (spec §17.6 V-01 to V-05)."""
        errors: list[str] = []

        # V-01
        if "vectordb" in self.chunk_output_mode and self.vector_db_type == "none":
            errors.append(
                "CHUNK_OUTPUT_MODE requires vectordb but VECTOR_DB_TYPE is none"
            )

        # V-02
        if "graphdb" in self.chunk_output_mode and self.graph_db_type == "none":
            errors.append(
                "CHUNK_OUTPUT_MODE requires graphdb but GRAPH_DB_TYPE is none"
            )

        # V-03
        if (
            self.rag_enabled
            and self.vector_db_type == "none"
            and self.graph_db_type == "none"
        ):
            errors.append("RAG_ENABLED requires at least one DB configured")

        # V-04
        if self.consolidator_enabled and self.graph_db_type == "none":
            errors.append("Consolidator requires a graph DB")

        # V-05
        if self.chunk_overlap >= self.chunk_target_size:
            errors.append("CHUNK_OVERLAP must be < CHUNK_TARGET_SIZE")

        if errors:
            raise ConfigurationError("; ".join(errors))

        return self

    # --- Helpers ---

    @property
    def graph_export_formats_list(self) -> list[str]:
        """Parse comma-separated graph export formats."""
        return [f.strip() for f in self.graph_export_formats.split(",") if f.strip()]

    @property
    def batch_scan_formats_list(self) -> list[str]:
        """Parse comma-separated batch scan formats."""
        return [f.strip() for f in self.batch_scan_formats.split(",") if f.strip()]

    @property
    def rag_enrich_agents_list(self) -> list[str]:
        """Parse comma-separated RAG agent list."""
        return [a.strip() for a in self.rag_enrich_agents.split(",") if a.strip()]

    @property
    def consolidator_passes_list(self) -> list[str]:
        """Parse comma-separated consolidator passes."""
        return [p.strip() for p in self.consolidator_passes.split(",") if p.strip()]


def load_settings(**overrides: object) -> Settings:
    """Load settings from .env with optional overrides.

    Args:
        **overrides: Field-level overrides (for testing or per-document config).

    Returns:
        Validated Settings instance.

    Raises:
        ConfigurationError: If configuration is internally inconsistent.
    """
    return Settings(**overrides)  # type: ignore[arg-type]
