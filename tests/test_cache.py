"""Tests for the prompt caching system."""

import tempfile
import os
import pytest

from src.llm.cache import PromptCache


class TestPromptCache:
    def test_cache_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PromptCache(cache_dir=tmpdir, enabled=True)

            result = cache.get("test prompt", "model-1", 0.0)
            assert result is None
            assert cache.stats["misses"] == 1

            cache.put("test prompt", "model-1", 0.0, "response text")

            result = cache.get("test prompt", "model-1", 0.0)
            assert result == "response text"
            assert cache.stats["hits"] == 1

    def test_different_params_different_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PromptCache(cache_dir=tmpdir, enabled=True)

            cache.put("prompt", "model-1", 0.0, "response A")
            cache.put("prompt", "model-1", 0.5, "response B")

            assert cache.get("prompt", "model-1", 0.0) == "response A"
            assert cache.get("prompt", "model-1", 0.5) == "response B"

    def test_disabled_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PromptCache(cache_dir=tmpdir, enabled=False)

            cache.put("prompt", "model", 0.0, "response")
            assert cache.get("prompt", "model", 0.0) is None
