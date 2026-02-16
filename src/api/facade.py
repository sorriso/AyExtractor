# src/api/facade.py — v1
"""Public API facade — single entry point for document analysis.

Usage:
    from ayextractor.api.facade import analyze
    result = await analyze(document, metadata)

See spec §2.1 for full documentation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ayextractor.api.models import AnalysisResult, DocumentInput, Metadata
from ayextractor.cache.fingerprint import compute_fingerprint
from ayextractor.config.settings import Settings

if TYPE_CHECKING:
    from ayextractor.cache.base_cache_store import BaseCacheStore
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


async def analyze(
    document: DocumentInput,
    metadata: Metadata,
    settings: Settings | None = None,
    cache_store: BaseCacheStore | None = None,
    vector_store: BaseVectorStore | None = None,
    graph_store: BaseGraphStore | None = None,
    embedder: BaseEmbedder | None = None,
) -> AnalysisResult:
    """Analyze a document end-to-end and return structured results.

    This is the main public API. It orchestrates the full pipeline:
      1. Resolve settings and apply per-document overrides
      2. Generate document_id and run_id
      3. Check cache for duplicates (fingerprint lookup)
      4. Run extraction → chunking → analysis pipeline
      5. Export results and index in stores if configured
      6. Return AnalysisResult

    Args:
        document: Input document (content + format + filename).
        metadata: Execution metadata (output path, language hint, etc.).
        settings: Global settings. Loaded from .env if None.
        cache_store: Cache backend for dedup. None = no caching.
        vector_store: Vector DB for RAG indexing. None = skip indexing.
        graph_store: Graph DB for RAG indexing. None = skip indexing.
        embedder: Embedding provider for vector indexing. None = skip.

    Returns:
        AnalysisResult with all outputs, paths, and stats.

    Raises:
        ValueError: If document format is unsupported or content is empty.
    """
    settings = settings or Settings()
    settings = _apply_overrides(settings, metadata)

    document_id = metadata.document_id or _generate_document_id()
    run_id = _generate_run_id(document_id)

    logger.info(
        "Starting analysis: document_id=%s, run_id=%s, format=%s",
        document_id, run_id, document.format,
    )

    # --- Phase 0: Fingerprint + cache check ---
    raw_bytes = _resolve_content_bytes(document)
    fingerprint = None
    if cache_store is not None:
        fingerprint = compute_fingerprint(
            raw_bytes=raw_bytes,
            extracted_text="",  # Pre-extraction; content hash computed later
            source_format=document.format,
        )
        lookup = await cache_store.lookup_fingerprint(fingerprint)
        if lookup.hit:
            logger.info("Cache hit (%s) for %s", lookup.match_level, document_id)

    # --- Phase 1-3: Pipeline ---
    from ayextractor.pipeline.document_pipeline import DocumentPipeline

    pipeline = DocumentPipeline(settings=settings)
    state = await pipeline.process(document_id=document_id)

    # --- Phase 4: Export + indexing ---
    output_dir = Path(metadata.output_path) / document_id
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    graph_path = run_dir / "graph.json"
    communities_path = run_dir / "communities.json"
    profiles_path = run_dir / "profiles.json"

    if settings.rag_enabled and vector_store and embedder:
        await _index_results(state, vector_store, embedder, settings)

    if settings.consolidator_enabled and graph_store and state.graph:
        await _link_to_corpus(state.graph, graph_store, settings)

    # --- Build result ---
    result = AnalysisResult(
        document_id=document_id,
        run_id=run_id,
        summary=state.synthesis or state.dense_summary,
        themes=[],
        concepts=[],
        relations=[],
        community_count=len(state.community_summaries),
        graph_path=graph_path,
        communities_path=communities_path,
        profiles_path=profiles_path,
        output_dir=output_dir,
        run_dir=run_dir,
        confidence_scores={
            name: out.confidence
            for name, out in state.agent_outputs.items()
        },
        fingerprint=fingerprint,
    )

    logger.info(
        "Analysis complete: document_id=%s, communities=%d, llm_calls=%d",
        document_id, result.community_count, state.total_llm_calls,
    )

    return result


def _apply_overrides(settings: Settings, metadata: Metadata) -> Settings:
    """Apply per-document config overrides if provided."""
    if metadata.config_overrides is None:
        return settings
    overrides = metadata.config_overrides.model_dump(exclude_none=True)
    if not overrides:
        return settings
    # Create new settings with overrides applied
    current = settings.model_dump()
    current.update(overrides)
    return Settings(**current)


def _generate_document_id() -> str:
    """Generate a unique document ID: yyyymmdd_hhmmss_{uuid4_short}."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


def _generate_run_id(document_id: str) -> str:
    """Generate a run ID: yyyymmdd_hhmm_{uuid5}."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    ns = uuid.NAMESPACE_DNS
    run_uuid = uuid.uuid5(ns, f"{document_id}_{ts}")
    return f"{ts}_{run_uuid.hex[:12]}"


def _resolve_content_bytes(document: DocumentInput) -> bytes:
    """Extract raw bytes from DocumentInput.content."""
    content = document.content
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, Path):
        return content.read_bytes()
    if isinstance(content, list):
        # Multi-image: concatenate bytes for fingerprinting
        return b"".join(p.read_bytes() for p in content)
    msg = f"Unsupported content type: {type(content)}"
    raise ValueError(msg)


async def _index_results(
    state: Any,
    vector_store: Any,
    embedder: Any,
    settings: Settings,
) -> None:
    """Post-analysis indexing into vector store (spec §26.5)."""
    try:
        from ayextractor.rag.indexer import index_analysis_results

        await index_analysis_results(
            state=state,
            vector_store=vector_store,
            embedder=embedder,
            settings=settings,
        )
    except Exception:
        logger.exception("Vector indexing failed (non-fatal)")


async def _link_to_corpus(
    graph: Any,
    graph_store: Any,
    settings: Settings,
) -> None:
    """Link Document Graph to Corpus Graph via Pass 1 (spec §13.15.3)."""
    try:
        from ayextractor.consolidator.orchestrator import ConsolidatorOrchestrator

        orchestrator = ConsolidatorOrchestrator(
            corpus_store=graph_store,
            settings=settings,
        )
        await orchestrator.run_linking(graph)
    except Exception:
        logger.exception("Corpus linking failed (non-fatal)")
