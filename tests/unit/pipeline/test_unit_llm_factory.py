# tests/unit/pipeline/test_llm_factory.py — v1
"""Tests for pipeline/llm_factory.py — LLM client creation and caching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ayextractor.config.settings import Settings
from ayextractor.llm.config import LLMAssignment
from ayextractor.pipeline.llm_factory import LLMFactory, _create_client


class TestLLMFactory:
    def test_callable_interface(self):
        """Factory should be callable (for runner.llm_factory)."""
        settings = Settings(anthropic_api_key="test-key")
        factory = LLMFactory(settings)
        with patch(
            "ayextractor.pipeline.llm_factory.resolve_llm",
            return_value=LLMAssignment(provider="anthropic", model="claude-sonnet-4-20250514", source="default"),
        ), patch(
            "ayextractor.pipeline.llm_factory._create_client",
            return_value=MagicMock(),
        ) as mock_create:
            client = factory("summarizer")
            assert client is not None
            mock_create.assert_called_once()

    def test_caching_same_assignment(self):
        """Same provider:model should return cached client."""
        settings = Settings(anthropic_api_key="test-key")
        factory = LLMFactory(settings)
        assignment = LLMAssignment(provider="anthropic", model="claude-sonnet-4-20250514", source="default")

        with patch(
            "ayextractor.pipeline.llm_factory.resolve_llm",
            return_value=assignment,
        ), patch(
            "ayextractor.pipeline.llm_factory._create_client",
            return_value=MagicMock(),
        ) as mock_create:
            c1 = factory.get_client("summarizer")
            c2 = factory.get_client("densifier")  # same assignment
            assert c1 is c2
            mock_create.assert_called_once()  # only created once

    def test_different_assignments_separate_clients(self):
        """Different provider:model should create separate clients."""
        settings = Settings(anthropic_api_key="k1", openai_api_key="k2")
        factory = LLMFactory(settings)

        call_count = 0
        def mock_resolve(component, _settings):
            nonlocal call_count
            call_count += 1
            if component == "summarizer":
                return LLMAssignment(provider="anthropic", model="claude-sonnet-4-20250514", source="default")
            return LLMAssignment(provider="openai", model="gpt-4o", source="component")

        with patch(
            "ayextractor.pipeline.llm_factory.resolve_llm",
            side_effect=mock_resolve,
        ), patch(
            "ayextractor.pipeline.llm_factory._create_client",
            side_effect=lambda a, s: MagicMock(name=a.key),
        ):
            c1 = factory.get_client("summarizer")
            c2 = factory.get_client("concept_extractor")
            assert c1 is not c2


class TestCreateClient:
    def test_unknown_provider_raises(self):
        assignment = LLMAssignment(provider="unknown", model="m", source="test")
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _create_client(assignment, settings)

    def test_anthropic_provider(self):
        assignment = LLMAssignment(provider="anthropic", model="claude-sonnet-4-20250514", source="test")
        settings = Settings(anthropic_api_key="test-key")
        with patch(
            "ayextractor.llm.adapters.anthropic_adapter.AnthropicAdapter",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            client = _create_client(assignment, settings)
            mock_cls.assert_called_once_with(
                api_key="test-key",
                model="claude-sonnet-4-20250514",
                max_tokens_default=settings.llm_max_tokens_per_agent,
            )

    def test_openai_provider(self):
        assignment = LLMAssignment(provider="openai", model="gpt-4o", source="test")
        settings = Settings(openai_api_key="test-key")
        with patch(
            "ayextractor.llm.adapters.openai_adapter.OpenAIAdapter",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            _create_client(assignment, settings)
            mock_cls.assert_called_once_with(
                api_key="test-key",
                model="gpt-4o",
            )

    def test_google_provider(self):
        assignment = LLMAssignment(provider="google", model="gemini-pro", source="test")
        settings = Settings(google_api_key="test-key")
        with patch(
            "ayextractor.llm.adapters.google_adapter.GoogleAdapter",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            _create_client(assignment, settings)
            mock_cls.assert_called_once_with(
                api_key="test-key",
                model="gemini-pro",
            )

    def test_ollama_provider(self):
        assignment = LLMAssignment(provider="ollama", model="llama3", source="test")
        settings = Settings()
        with patch(
            "ayextractor.llm.adapters.ollama_adapter.OllamaAdapter",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            _create_client(assignment, settings)
            mock_cls.assert_called_once_with(
                host=settings.ollama_base_url,
                model="llama3",
            )
