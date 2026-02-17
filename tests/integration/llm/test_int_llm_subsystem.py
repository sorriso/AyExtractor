# tests/integration/llm/test_int_llm_subsystem.py — v3
"""Integration tests for the LLM subsystem.

Covers: llm/config.py, llm/client_factory.py, llm/retry.py,
        llm/token_budget.py, llm/base_client.py, llm/models.py
        llm/adapters (mock-based for non-Ollama)

No Docker required — uses mock LLM client for adapter tests.

Changelog:
    v3: Fix test_phase_override — use Settings.model_construct() to bypass
        pydantic-settings env var / .env loading that overrides init kwargs.
        resolve_llm cascade logic is the test target, not Settings loading.
    v2: Initial rewrite with correct source API.
"""

from __future__ import annotations

import asyncio

import pytest

from ayextractor.config.settings import Settings
from ayextractor.llm.models import LLMResponse, Message


# =====================================================================
#  LLM CONFIG — Cascade resolution
# =====================================================================

class TestLLMConfig:
    """Test 3-level cascade resolution for LLM assignment.

    Uses Settings.model_construct() for tests requiring specific field
    values to avoid interference from .env file or env vars (Settings
    extends pydantic-settings BaseSettings which reads from environment).
    """

    def test_fallback_default(self):
        """Default Settings provides anthropic → 'default' source."""
        from ayextractor.llm.config import resolve_llm
        settings = Settings()
        result = resolve_llm("summarizer", settings)
        assert result.source == "default"
        assert result.provider == "anthropic"

    def test_default_provider_override(self):
        """LLM_DEFAULT_PROVIDER + MODEL → 'default' source."""
        from ayextractor.llm.config import resolve_llm
        # model_construct bypasses env loading → pure cascade test
        settings = Settings.model_construct(
            llm_default_provider="openai",
            llm_default_model="gpt-4o",
            llm_summarizer="",
        )
        result = resolve_llm("summarizer", settings)
        assert result.source == "default"
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    def test_phase_override(self):
        """Per-phase env var → 'phase' source beats 'default'.
        Note: 'summarizer' belongs to the 'chunking' phase (per PHASE_COMPONENT_MAP).

        Uses model_construct to guarantee llm_default_provider="" is not
        overridden by a .env file or env var in the devcontainer.
        """
        from ayextractor.llm.config import resolve_llm
        settings = Settings.model_construct(
            llm_default_provider="",
            llm_default_model="",
            llm_phase_chunking="google:gemini-pro",
            llm_summarizer="",
        )
        result = resolve_llm("summarizer", settings)
        assert result.source == "phase"
        assert result.provider == "google"

    def test_component_override_highest_priority(self):
        """Per-component override beats phase and default."""
        from ayextractor.llm.config import resolve_llm
        settings = Settings.model_construct(
            llm_default_provider="openai",
            llm_default_model="gpt-4o",
            llm_summarizer="ollama:llama3",
            llm_phase_chunking="google:gemini-pro",
        )
        result = resolve_llm("summarizer", settings)
        assert result.source == "component"
        assert result.provider == "ollama"
        assert result.model == "llama3"

    def test_resolve_all_returns_dict(self):
        """resolve_all returns assignment for every known component."""
        from ayextractor.llm.config import resolve_all
        settings = Settings()
        mapping = resolve_all(settings)
        assert isinstance(mapping, dict)
        assert "summarizer" in mapping
        assert "concept_extractor" in mapping
        for name, assignment in mapping.items():
            assert assignment.provider
            assert assignment.model

    def test_key_property(self):
        """LLMAssignment.key returns 'provider:model'."""
        from ayextractor.llm.config import LLMAssignment
        a = LLMAssignment(provider="ollama", model="llama3", source="test")
        assert a.key == "ollama:llama3"

    def test_parse_empty_returns_fallback(self):
        """Empty component override → skip to next level."""
        from ayextractor.llm.config import resolve_llm
        settings = Settings.model_construct(
            llm_summarizer="",
            llm_default_provider="anthropic",
            llm_default_model="claude-sonnet-4-20250514",
        )
        result = resolve_llm("summarizer", settings)
        assert result.source != "component"

    def test_unknown_component_fallback(self):
        """Unknown component (not in any phase) → default or fallback."""
        from ayextractor.llm.config import resolve_llm
        settings = Settings()
        result = resolve_llm("nonexistent_agent", settings)
        assert result.source in ("default", "fallback")

    def test_cascade_falls_through_to_fallback(self):
        """All levels empty → hardcoded fallback (anthropic:claude-sonnet)."""
        from ayextractor.llm.config import resolve_llm
        settings = Settings.model_construct(
            llm_default_provider="",
            llm_default_model="",
            llm_summarizer="",
            llm_phase_chunking="",
        )
        result = resolve_llm("summarizer", settings)
        assert result.source == "fallback"
        assert result.provider == "anthropic"


# =====================================================================
#  CLIENT FACTORY
# =====================================================================

class TestClientFactory:
    """Test LLM client factory creation."""

    def test_create_ollama_adapter(self):
        from ayextractor.llm.client_factory import create_llm_client
        client = create_llm_client("ollama", "llama3")
        assert client.provider_name == "ollama"

    def test_unsupported_provider_raises(self):
        from ayextractor.llm.client_factory import create_llm_client, UnsupportedProviderError
        with pytest.raises(UnsupportedProviderError):
            create_llm_client("nonexistent_provider", "model")

    def test_create_returns_base_client(self):
        from ayextractor.llm.base_client import BaseLLMClient
        from ayextractor.llm.client_factory import create_llm_client
        client = create_llm_client("ollama", "llama3")
        assert isinstance(client, BaseLLMClient)

    def test_register_custom_provider(self):
        from ayextractor.llm.client_factory import (
            _PROVIDER_REGISTRY,
            register_provider,
        )
        register_provider("custom", "ayextractor.llm.adapters.ollama_adapter.OllamaAdapter")
        assert "custom" in _PROVIDER_REGISTRY
        # Cleanup
        del _PROVIDER_REGISTRY["custom"]

    def test_create_with_settings(self):
        from ayextractor.llm.client_factory import create_llm_client
        settings = Settings(ollama_base_url="http://test:11434")
        client = create_llm_client("ollama", "llama3", settings=settings)
        assert client.provider_name == "ollama"
        assert client._model == "llama3"


# =====================================================================
#  RETRY LOGIC
# =====================================================================

class TestRetryLogic:
    """Test retry mechanism with mock async functions."""

    def test_classify_rate_limit(self):
        from ayextractor.llm.retry import classify_error
        assert classify_error(Exception("429 Too Many Requests")) == "rate_limit"
        assert classify_error(Exception("rate limit exceeded")) == "rate_limit"

    def test_classify_timeout(self):
        from ayextractor.llm.retry import classify_error
        assert classify_error(TimeoutError("request timed out")) == "timeout"

    def test_classify_server_error(self):
        from ayextractor.llm.retry import classify_error
        assert classify_error(Exception("500 Internal Server Error")) == "server_error"
        assert classify_error(Exception("502 bad gateway")) == "server_error"

    def test_classify_parse_error(self):
        from ayextractor.llm.retry import classify_error
        assert classify_error(Exception("JSON decode error")) == "parse_error"

    def test_classify_token_limit(self):
        from ayextractor.llm.retry import classify_error
        assert classify_error(Exception("token limit exceeded")) == "token_limit"

    def test_classify_unknown(self):
        from ayextractor.llm.retry import classify_error
        assert classify_error(Exception("something weird")) == "unknown"

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        from ayextractor.llm.retry import with_retry

        async def ok_fn():
            return "result"

        result = await with_retry(ok_fn, agent="test")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failure(self):
        from ayextractor.llm.retry import RetryConfig, with_retry

        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("500 server error")
            return "recovered"

        configs = {
            "server_error": RetryConfig(max_retries=5, base_delay_s=0.0),
        }
        result = await with_retry(flaky_fn, agent="test", retry_configs=configs)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        from ayextractor.llm.retry import LLMRetryExhausted, RetryConfig, with_retry

        async def always_fail():
            raise Exception("429 rate limit")

        configs = {
            "rate_limit": RetryConfig(max_retries=2, base_delay_s=0.0),
        }
        with pytest.raises(LLMRetryExhausted) as exc_info:
            await with_retry(always_fail, agent="my_agent", retry_configs=configs)
        assert exc_info.value.agent == "my_agent"
        assert exc_info.value.error_type == "rate_limit"

    @pytest.mark.asyncio
    async def test_retry_unknown_error_not_retried(self):
        from ayextractor.llm.retry import LLMRetryExhausted, with_retry

        async def unknown_fail():
            raise ValueError("unexpected")

        with pytest.raises(LLMRetryExhausted):
            await with_retry(unknown_fail, agent="test")

    def test_compute_delay_basic(self):
        from ayextractor.llm.retry import RetryConfig, _compute_delay
        config = RetryConfig(max_retries=3, base_delay_s=1.0, backoff_factor=2.0, jitter=False)
        assert _compute_delay(config, 0) == 1.0
        assert _compute_delay(config, 1) == 2.0
        assert _compute_delay(config, 2) == 4.0

    def test_compute_delay_with_jitter(self):
        from ayextractor.llm.retry import RetryConfig, _compute_delay
        config = RetryConfig(max_retries=3, base_delay_s=1.0, jitter=True)
        delays = [_compute_delay(config, 0) for _ in range(10)]
        assert all(0.5 <= d <= 1.5 for d in delays)


# =====================================================================
#  TOKEN BUDGET
# =====================================================================

class TestTokenBudget:
    """Test budget estimation and tracking."""

    def test_estimate_budget_basic(self):
        from ayextractor.llm.token_budget import estimate_budget
        budget = estimate_budget(text_tokens=5000, n_chunks=10, n_images=2)
        assert budget.total_estimated > 0
        assert "summarizer" in budget.per_agent
        assert "image_analyzer" in budget.per_agent
        assert budget.per_agent["image_analyzer"] == 2 * 1000

    def test_estimate_budget_no_images(self):
        from ayextractor.llm.token_budget import estimate_budget
        budget = estimate_budget(text_tokens=3000, n_chunks=5)
        assert budget.per_agent["image_analyzer"] == 0

    def test_critic_disabled_by_default(self):
        from ayextractor.llm.token_budget import estimate_budget
        budget = estimate_budget(text_tokens=3000, n_chunks=5)
        assert budget.per_agent["critic"] == 0

    def test_critic_enabled(self):
        from ayextractor.llm.token_budget import estimate_budget
        budget = estimate_budget(text_tokens=3000, n_chunks=5, critic_enabled=True)
        assert budget.per_agent["critic"] > 0

    def test_check_budget_within(self):
        from ayextractor.llm.token_budget import check_budget, estimate_budget
        budget = estimate_budget(text_tokens=3000, n_chunks=5)
        assert check_budget(budget, "summarizer") is True

    def test_check_budget_over(self):
        from ayextractor.llm.token_budget import check_budget, estimate_budget, record_usage
        budget = estimate_budget(text_tokens=3000, n_chunks=5)
        # Force over budget
        record_usage(budget, "summarizer", budget.per_agent["summarizer"] + 1)
        assert check_budget(budget, "summarizer") is False

    def test_record_usage_accumulates(self):
        from ayextractor.llm.token_budget import estimate_budget, record_usage
        budget = estimate_budget(text_tokens=3000, n_chunks=5)
        record_usage(budget, "summarizer", 100)
        record_usage(budget, "summarizer", 200)
        assert budget.consumed["summarizer"] == 300

    def test_estimate_with_settings(self):
        from ayextractor.llm.token_budget import estimate_budget
        settings = Settings(density_iterations=3, llm_max_tokens_per_agent=2048)
        budget = estimate_budget(text_tokens=3000, n_chunks=5, settings=settings)
        assert budget.per_agent["densifier"] == 3 * 2048


# =====================================================================
#  LLM MODELS
# =====================================================================

class TestLLMModels:
    """Test LLM model data classes."""

    def test_message_creation(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_llm_response_fields(self):
        resp = LLMResponse(
            content="answer",
            input_tokens=10,
            output_tokens=5,
            model="test",
            provider="test",
            latency_ms=100,
            raw_response={},
        )
        assert resp.content == "answer"
        assert resp.input_tokens == 10

    def test_mock_llm_interface(self, mock_llm):
        """Verify mock LLM conforms to BaseLLMClient interface."""
        from ayextractor.llm.base_client import BaseLLMClient
        assert isinstance(mock_llm, BaseLLMClient)
        assert mock_llm.provider_name == "mock"
        assert mock_llm.supports_vision is True

    @pytest.mark.asyncio
    async def test_mock_llm_complete(self, mock_llm):
        mock_llm.set_default("test response")
        result = await mock_llm.complete([Message(role="user", content="hi")])
        assert result.content == "test response"
        assert result.provider == "mock"
        assert len(mock_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_mock_llm_queued_responses(self, mock_llm):
        mock_llm.set_responses("first", "second", "third")
        r1 = await mock_llm.complete([Message(role="user", content="1")])
        r2 = await mock_llm.complete([Message(role="user", content="2")])
        r3 = await mock_llm.complete([Message(role="user", content="3")])
        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == "third"