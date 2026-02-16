# src/consolidator/orchestrator.py — v1
"""Consolidator orchestrator — schedule and run consolidation passes.

Manages the execution of 5 independent consolidation passes on the
Corpus Graph:
  Pass 1: Linking (entity_linker) — always synchronous post-ingestion
  Pass 2: Clustering (community_clusterer) — periodic
  Pass 3: Inference (inference_engine) — periodic
  Pass 4: Decay (decay_manager) — periodic
  Pass 5: Contradiction (contradiction_detector) — periodic

See spec §13.15 for full documentation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ayextractor.consolidator.models import (
    ConsolidationReport,
    PassResult,
)

if TYPE_CHECKING:
    from ayextractor.config.settings import Settings
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore

logger = logging.getLogger(__name__)

# Ordered pass registry: name → (module_path, function_name)
PASS_REGISTRY: list[tuple[str, str]] = [
    ("linking", "consolidator.entity_linker"),
    ("clustering", "consolidator.community_clusterer"),
    ("inference", "consolidator.inference_engine"),
    ("decay", "consolidator.decay_manager"),
    ("contradiction", "consolidator.contradiction_detector"),
]

# Pass 1 import — always available for synchronous post-ingestion
_LINKING_PASS = "linking"


class ConsolidatorOrchestrator:
    """Orchestrate consolidation passes on the Corpus Graph.

    Args:
        corpus_store: Graph store backing the Corpus Graph.
        settings: Application settings (for pass config).
    """

    def __init__(
        self,
        corpus_store: BaseGraphStore,
        settings: Settings | None = None,
    ) -> None:
        self._store = corpus_store
        self._settings = settings
        self._active_passes = self._resolve_active_passes(settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        trigger: str = "manual",
        document_graph: Any | None = None,
    ) -> ConsolidationReport:
        """Run all active passes in order.

        Args:
            trigger: Trigger source (on_ingestion, scheduled, manual).
            document_graph: nx.Graph for linking pass (required if linking active).

        Returns:
            ConsolidationReport with per-pass results.
        """
        ts = datetime.now(timezone.utc)
        consolidation_id = f"cons_{ts.strftime('%Y%m%d_%H%M%S')}"

        results: dict[str, PassResult] = {}
        executed: list[str] = []

        for pass_name in self._active_passes:
            logger.info("Running consolidation pass: %s", pass_name)
            start = time.monotonic()
            try:
                detail = self._run_pass(pass_name, document_graph)
                duration_ms = int((time.monotonic() - start) * 1000)
                results[pass_name] = PassResult(
                    pass_name=pass_name,
                    duration_ms=duration_ms,
                    items_processed=detail.get("items_processed", 0),
                    items_modified=detail.get("items_modified", 0),
                    details=detail,
                )
                executed.append(pass_name)
                logger.info(
                    "Pass %s completed in %dms: %s",
                    pass_name, duration_ms, detail,
                )
            except Exception:
                logger.exception("Pass %s failed", pass_name)
                duration_ms = int((time.monotonic() - start) * 1000)
                results[pass_name] = PassResult(
                    pass_name=pass_name,
                    duration_ms=duration_ms,
                    items_processed=0,
                    items_modified=0,
                    details={"error": True},
                )

        return ConsolidationReport(
            consolidation_id=consolidation_id,
            timestamp=ts,
            trigger=trigger,  # type: ignore[arg-type]
            passes_executed=executed,
            results=results,
            corpus_stats=self._gather_corpus_stats(),
        )

    def run_linking(self, document_graph: Any) -> PassResult:
        """Run Pass 1 (Linking) synchronously after document ingestion.

        Always executed regardless of CONSOLIDATOR_TRIGGER setting.

        Args:
            document_graph: nx.Graph of the Document Graph to link.

        Returns:
            PassResult for the linking pass.
        """
        start = time.monotonic()
        detail = self._execute_linking(document_graph)
        duration_ms = int((time.monotonic() - start) * 1000)
        return PassResult(
            pass_name="linking",
            duration_ms=duration_ms,
            items_processed=detail.get("items_processed", 0),
            items_modified=detail.get("items_modified", 0),
            details=detail,
        )

    # ------------------------------------------------------------------
    # Internal pass dispatch
    # ------------------------------------------------------------------

    def _run_pass(
        self,
        pass_name: str,
        document_graph: Any | None = None,
    ) -> dict[str, Any]:
        """Dispatch a single pass by name."""
        if pass_name == "linking":
            return self._execute_linking(document_graph)
        elif pass_name == "clustering":
            return self._execute_clustering()
        elif pass_name == "inference":
            return self._execute_inference()
        elif pass_name == "decay":
            return self._execute_decay()
        elif pass_name == "contradiction":
            return self._execute_contradiction()
        else:
            raise ValueError(f"Unknown pass: {pass_name}")

    def _execute_linking(self, document_graph: Any | None) -> dict[str, Any]:
        """Execute Pass 1 — entity linking."""
        from ayextractor.graph.entity_linker import link_entities

        if document_graph is None:
            logger.warning("Linking pass skipped: no document_graph provided")
            return {"skipped": True, "items_processed": 0, "items_modified": 0}

        # Extract entities from document graph nodes
        incoming = _extract_entities_from_graph(document_graph)
        existing = self._store.get_all_entities() if hasattr(self._store, "get_all_entities") else []

        # Perform linking (sync wrapper around async)
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                link_entities(existing=existing, incoming=incoming)
            )
        finally:
            loop.close()

        items_modified = len(result.matched) + len(result.new_entities)
        return {
            "items_processed": len(incoming),
            "items_modified": items_modified,
            "matched": len(result.matched),
            "new_entities": len(result.new_entities),
        }

    def _execute_clustering(self) -> dict[str, Any]:
        """Execute Pass 2 — community clustering on Corpus Graph."""
        from ayextractor.consolidator.community_clusterer import cluster_corpus

        min_size = 3
        seed = 42
        if self._settings:
            min_size = getattr(self._settings, "CONSOLIDATOR_CLUSTER_MIN_SIZE", 3)

        report = cluster_corpus(
            corpus_store=self._store,
            min_cluster_size=min_size,
            seed=seed,
        )
        return {
            "items_processed": report.clusters_found,
            "items_modified": report.new_tnodes + report.updated_tnodes,
            "new_tnodes": report.new_tnodes,
            "updated_tnodes": report.updated_tnodes,
            "clusters_found": report.clusters_found,
        }

    def _execute_inference(self) -> dict[str, Any]:
        """Execute Pass 3 — transitive inference."""
        from ayextractor.graph.inference_engine import infer_transitive

        min_conf = 0.6
        if self._settings:
            min_conf = getattr(
                self._settings, "CONSOLIDATOR_INFERENCE_MIN_CONFIDENCE", 0.6
            )

        graph = self._store.to_networkx() if hasattr(self._store, "to_networkx") else None
        if graph is None:
            return {"skipped": True, "items_processed": 0, "items_modified": 0}

        result = infer_transitive(graph, min_confidence=min_conf)
        return {
            "items_processed": graph.number_of_edges(),
            "items_modified": len(result.inferred),
            "proposed": len(result.inferred),
        }

    def _execute_decay(self) -> dict[str, Any]:
        """Execute Pass 4 — staleness decay."""
        from ayextractor.graph.decay_manager import apply_decay

        halflife = 90
        if self._settings:
            halflife = getattr(
                self._settings, "CONSOLIDATOR_DECAY_HALFLIFE_DAYS", 90
            )

        graph = self._store.to_networkx() if hasattr(self._store, "to_networkx") else None
        if graph is None:
            return {"skipped": True, "items_processed": 0, "items_modified": 0}

        result = apply_decay(graph, halflife_days=halflife)
        return {
            "items_processed": result.total_processed,
            "items_modified": result.decayed + result.pruned,
            "decayed": result.decayed,
            "pruned": result.pruned,
        }

    def _execute_contradiction(self) -> dict[str, Any]:
        """Execute Pass 5 — contradiction detection."""
        from ayextractor.graph.contradiction_detector import detect_contradictions

        graph = self._store.to_networkx() if hasattr(self._store, "to_networkx") else None
        if graph is None:
            return {"skipped": True, "items_processed": 0, "items_modified": 0}

        result = detect_contradictions(graph)
        return {
            "items_processed": graph.number_of_edges(),
            "items_modified": len(result.contradictions),
            "contradictions_found": len(result.contradictions),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_active_passes(self, settings: Any | None) -> list[str]:
        """Determine which passes to execute from config."""
        all_passes = [name for name, _ in PASS_REGISTRY]

        if settings is None:
            return all_passes

        raw = getattr(settings, "CONSOLIDATOR_PASSES", None)
        if raw is None:
            return all_passes

        if isinstance(raw, str):
            requested = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            requested = list(raw)

        # Preserve ordering from PASS_REGISTRY
        return [p for p in all_passes if p in requested]

    def _gather_corpus_stats(self) -> dict[str, int]:
        """Collect basic stats from the corpus store."""
        stats: dict[str, int] = {}
        try:
            if hasattr(self._store, "count_nodes"):
                stats["total_cnodes"] = self._store.count_nodes()
            if hasattr(self._store, "count_edges"):
                stats["total_xedges"] = self._store.count_edges()
        except Exception:
            logger.debug("Could not gather corpus stats", exc_info=True)
        return stats


def _extract_entities_from_graph(graph: Any) -> list:
    """Extract entity normalizations from an nx.Graph for linking."""
    from ayextractor.core.models import EntityNormalization

    entities = []
    for node_id, data in graph.nodes(data=True):
        if data.get("layer") == "L2":
            entities.append(
                EntityNormalization(
                    canonical_name=data.get("canonical_name", node_id),
                    entity_type=data.get("entity_type"),
                    confidence=data.get("confidence", 0.5),
                )
            )
    return entities
