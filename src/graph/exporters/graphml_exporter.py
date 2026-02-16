# src/graph/exporters/graphml_exporter.py — v1
"""GraphML graph exporter for standard interchange.

See spec §30.5.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from ayextractor.graph.base_graph_exporter import BaseGraphExporter


class GraphMLExporter(BaseGraphExporter):
    """Export graph to GraphML format."""

    @property
    def format_name(self) -> str:
        return "graphml"

    @property
    def file_extension(self) -> str:
        return ".graphml"

    async def export(self, graph: nx.Graph, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # GraphML requires string attribute values
        g = graph.copy()
        for _, data in g.nodes(data=True):
            for k, v in list(data.items()):
                if not isinstance(v, (str, int, float, bool)):
                    data[k] = str(v)
        for _, _, data in g.edges(data=True):
            for k, v in list(data.items()):
                if not isinstance(v, (str, int, float, bool)):
                    data[k] = str(v)

        nx.write_graphml(g, str(path))
        return str(path)
