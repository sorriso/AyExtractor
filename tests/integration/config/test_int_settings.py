# tests/integration/config/test_int_settings.py — v1
"""Integration tests for configuration loading.

Tests Settings with real .env files, validation rules, and cross-field consistency.
No external services required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayextractor.config.settings import ConfigurationError, Settings


class TestSettingsLoading:

    def test_defaults(self):
        settings = Settings(_env_file=None)
        assert settings.llm_default_provider == "anthropic"
        assert settings.chunking_strategy == "structural"
        assert settings.cache_enabled is True

    def test_from_env_file(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_DEFAULT_PROVIDER=ollama\n"
            "LLM_DEFAULT_MODEL=qwen2.5:1.5b\n"
            "OLLAMA_BASE_URL=http://localhost:11434\n"
            "EMBEDDING_PROVIDER=ollama\n"
            "EMBEDDING_OLLAMA_MODEL=nomic-embed-text\n"
            "VECTOR_DB_TYPE=qdrant\n"
            "VECTOR_DB_URL=http://localhost:6333\n"
            "GRAPH_DB_TYPE=arangodb\n"
            "GRAPH_DB_URI=http://localhost:8529\n"
            "GRAPH_DB_USER=root\n"
            "GRAPH_DB_PASSWORD=testpwd\n"
        )
        settings = Settings(_env_file=str(env_file))
        assert settings.llm_default_provider == "ollama"
        assert settings.vector_db_type == "qdrant"
        assert settings.graph_db_type == "arangodb"

    def test_rag_enrich_agents_list(self):
        settings = Settings(_env_file=None)
        agents = settings.rag_enrich_agents_list
        assert isinstance(agents, list)

    def test_oversize_strategy_options(self):
        for strategy in ["reject", "truncate", "sample"]:
            s = Settings(_env_file=None, oversize_strategy=strategy)
            assert s.oversize_strategy == strategy

    def test_chunking_strategies(self):
        for strat in ["structural", "semantic"]:
            s = Settings(_env_file=None, chunking_strategy=strat)
            assert s.chunking_strategy == strat


class TestSettingsValidation:

    def test_rag_needs_vector_db(self):
        """RAG enabled with vector_db_type=none should fail validation."""
        with pytest.raises(Exception):
            Settings(
                _env_file=None,
                rag_enabled=True,
                vector_db_type="none",
            )
