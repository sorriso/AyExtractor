# src/graph/base_graph_exporter.py — v1
"""Abstract graph export interface.

See spec §30.5 for full documentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


class BaseGraphExporter(ABC):
    """Unified interface for knowledge graph export formats."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Export format identifier (e.g., 'json', 'graphml')."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Output file extension (e.g., '.json', '.graphml')."""

    @abstractmethod
    async def export(self, graph: nx.Graph, output_path: str) -> str:
        """Export graph to file, return path to exported file."""
