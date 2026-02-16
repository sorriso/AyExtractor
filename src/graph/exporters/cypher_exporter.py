# src/graph/exporters/cypher_exporter.py — v1
"""Cypher script exporter for Neo4j direct import.

Generates a .cypher file with CREATE statements.
See spec §30.5.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from ayextractor.graph.base_graph_exporter import BaseGraphExporter


def _escape(value: str) -> str:
    """Escape a string for Cypher."""
    return value.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')


class CypherExporter(BaseGraphExporter):
    """Export graph to Cypher CREATE statements."""

    @property
    def format_name(self) -> str:
        return "cypher"

    @property
    def file_extension(self) -> str:
        return ".cypher"

    async def export(self, graph: nx.Graph, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append("// Auto-generated Cypher import script")
        lines.append("// ayextractor graph export\n")

        # Nodes
        for node_id, data in graph.nodes(data=True):
            label = data.get("entity_type", "Entity")
            props = {k: str(v) for k, v in data.items() if k != "entity_type"}
            props["name"] = str(node_id)
            props_str = ", ".join(f'{k}: "{_escape(v)}"' for k, v in props.items())
            lines.append(f"CREATE (:`{label}` {{{props_str}}});")

        lines.append("")

        # Edges
        for src, tgt, data in graph.edges(data=True):
            rel_type = data.get("relation_type", "RELATED_TO").upper().replace(" ", "_")
            props = {k: str(v) for k, v in data.items() if k != "relation_type"}
            props_str = ", ".join(f'{k}: "{_escape(v)}"' for k, v in props.items())
            lines.append(
                f'MATCH (a {{name: "{_escape(str(src))}"}}),'
                f' (b {{name: "{_escape(str(tgt))}"}}) '
                f"CREATE (a)-[:`{rel_type}` {{{props_str}}}]->(b);"
            )

        path.write_text("\n".join(lines))
        return str(path)
