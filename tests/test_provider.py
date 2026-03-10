"""Tests for the LLM provider abstraction."""

import json
import pytest

from src.llm.provider import MockProvider, get_provider
from src.llm.cache import PromptCache


class TestMockProvider:
    def test_default_response(self):
        provider = MockProvider(rate_limit_delay=0)
        response = provider.query("What should I do?")
        data = json.loads(response)
        assert data["action"] == "COOPERATE"
        assert "confidence" in data

    def test_custom_response(self):
        provider = MockProvider(rate_limit_delay=0)
        custom = json.dumps({"action": "DEFECT", "confidence": 0.9,
                              "observed_state_summary": "test",
                              "beliefs": {}, "short_rationale": "test"})
        provider.set_response("defect_prompt", custom)

        response = provider.query("This is a defect_prompt test")
        data = json.loads(response)
        assert data["action"] == "DEFECT"

    def test_usage_stats(self):
        provider = MockProvider(rate_limit_delay=0)
        provider.query("test")
        provider.query("test2")
        stats = provider.usage_stats
        assert stats["call_count"] == 2

    def test_caching_integration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PromptCache(cache_dir=tmpdir, enabled=True)
            provider = MockProvider(cache=cache, rate_limit_delay=0)

            r1 = provider.query("cached prompt")
            r2 = provider.query("cached prompt")

            assert r1 == r2
            assert cache.stats["hits"] == 1
            assert cache.stats["misses"] == 1
            # Only one API call should have been made
            assert provider.usage_stats["call_count"] == 1


class TestGetProvider:
    def test_mock_provider(self):
        provider = get_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")
