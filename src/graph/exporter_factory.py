# src/graph/exporter_factory.py — v1
"""Factory for graph exporter instantiation.

JSON exporter is always included regardless of configuration.
See spec §30.5 and §30.8.
"""

from __future__ import annotations

from ayextractor.config.settings import Settings
from ayextractor.graph.base_graph_exporter import BaseGraphExporter

_EXPORTERS: dict[str, str] = {
    "json": "ayextractor.graph.exporters.json_exporter.JsonExporter",
    "graphml": "ayextractor.graph.exporters.graphml_exporter.GraphMLExporter",
    "gexf": "ayextractor.graph.exporters.gexf_exporter.GexfExporter",
    "cypher": "ayextractor.graph.exporters.cypher_exporter.CypherExporter",
}


def create_exporters(settings: Settings | None = None) -> list[BaseGraphExporter]:
    """Create all configured graph exporters.

    JSON is always included. Additional formats come from settings.

    Returns:
        List of exporter instances.
    """
    formats: set[str] = {"json"}  # Always present

    if settings is not None:
        formats.update(settings.graph_export_formats_list)

    exporters: list[BaseGraphExporter] = []
    for fmt in sorted(formats):
        fqcn = _EXPORTERS.get(fmt)
        if fqcn is None:
            continue
        module_path, class_name = fqcn.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        exporters.append(cls())

    return exporters
