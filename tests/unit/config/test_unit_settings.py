# tests/unit/config/test_settings.py — v1
"""Tests for config/settings.py — typed Settings and validation rules."""

from __future__ import annotations

import pytest

from ayextractor.config.settings import ConfigurationError, Settings, load_settings


class TestSettingsDefaults:
    def test_default_provider(self):
        s = Settings(_env_file=None)
        assert s.llm_default_provider == "anthropic"

    def test_default_chunking(self):
        s = Settings(_env_file=None)
        assert s.chunking_strategy == "structural"
        assert s.chunk_target_size == 2000
        assert s.chunk_overlap == 0

    def test_default_cache(self):
        s = Settings(_env_file=None)
        assert s.cache_enabled is True
        assert s.cache_backend == "json"

    def test_default_rag_disabled(self):
        s = Settings(_env_file=None)
        assert s.rag_enabled is False

    def test_default_consolidator_disabled(self):
        s = Settings(_env_file=None)
        assert s.consolidator_enabled is False


class TestSettingsValidation:
    def test_v01_vectordb_mode_without_vector_db(self):
        with pytest.raises(ConfigurationError, match="vectordb"):
            Settings(
                _env_file=None,
                chunk_output_mode="files_and_vectordb",
                vector_db_type="none",
            )

    def test_v02_graphdb_mode_without_graph_db(self):
        with pytest.raises(ConfigurationError, match="graphdb"):
            Settings(
                _env_file=None,
                chunk_output_mode="files_and_graphdb",
                graph_db_type="none",
            )

    def test_v03_rag_without_any_db(self):
        with pytest.raises(ConfigurationError, match="RAG_ENABLED"):
            Settings(
                _env_file=None,
                rag_enabled=True,
                vector_db_type="none",
                graph_db_type="none",
            )

    def test_v04_consolidator_without_graph_db(self):
        with pytest.raises(ConfigurationError, match="Consolidator"):
            Settings(
                _env_file=None,
                consolidator_enabled=True,
                graph_db_type="none",
            )

    def test_v05_overlap_gte_target(self):
        with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
            Settings(
                _env_file=None,
                chunk_target_size=2000,
                chunk_overlap=2000,
            )

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            Settings(_env_file=None, chunk_overlap=-1)

    def test_valid_vectordb_mode(self):
        s = Settings(
            _env_file=None,
            chunk_output_mode="files_and_vectordb",
            vector_db_type="chromadb",
        )
        assert s.chunk_output_mode == "files_and_vectordb"

    def test_valid_rag_with_vector_db(self):
        s = Settings(
            _env_file=None,
            rag_enabled=True,
            vector_db_type="chromadb",
        )
        assert s.rag_enabled is True


class TestSettingsHelpers:
    def test_graph_export_formats_list(self):
        s = Settings(_env_file=None, graph_export_formats="graphml,gexf,cypher")
        assert s.graph_export_formats_list == ["graphml", "gexf", "cypher"]

    def test_batch_scan_formats_list(self):
        s = Settings(_env_file=None)
        assert "pdf" in s.batch_scan_formats_list

    def test_rag_enrich_agents_list(self):
        s = Settings(_env_file=None)
        agents = s.rag_enrich_agents_list
        assert "summarizer" not in agents
        assert "decontextualizer" in agents

    def test_consolidator_passes_list(self):
        s = Settings(_env_file=None)
        passes = s.consolidator_passes_list
        assert "linking" in passes
        assert len(passes) == 5


class TestLoadSettings:
    def test_with_overrides(self):
        s = load_settings(log_level="DEBUG", chunk_target_size=4000)
        assert s.log_level == "DEBUG"
        assert s.chunk_target_size == 4000
