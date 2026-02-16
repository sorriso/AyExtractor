# src/graph/exporters/json_exporter.py — v1
"""JSON graph exporter using NetworkX node_link format.

Always generated regardless of GRAPH_EXPORT_FORMATS setting.
See spec §30.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from ayextractor.graph.base_graph_exporter import BaseGraphExporter


class JsonExporter(BaseGraphExporter):
    """Export graph to NetworkX JSON node_link format."""

    @property
    def format_name(self) -> str:
        return "json"

    @property
    def file_extension(self) -> str:
        return ".json"

    async def export(self, graph: nx.Graph, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(graph)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return str(path)
