"""Prompt caching for LLM responses.

Caches LLM responses keyed by (prompt, model, temperature) to reduce
API costs and ensure reproducibility across runs with identical inputs.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class PromptCache:
    """File-based cache for LLM prompt-response pairs."""

    def __init__(self, cache_dir: str = ".llm_cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create a deterministic cache key from prompt parameters."""
        content = json.dumps({
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Retrieve a cached response, or None if not cached."""
        if not self.enabled:
            return None
        key = self._make_key(prompt, model, temperature)
        path = self._cache_path(key)
        if path.exists():
          try:  
              with open(path, "r") as f:
                  data = json.load(f)
              self._hits += 1
              return data["response"]   
          except (json.JSONDecodeError, KeyError, OSError) as e:    
              print(f"[WARNING] Failed to read cache file {path}: {e}")
              self._misses += 1
              return None
        self._misses += 1
        return None  

    def put(self, prompt: str, model: str, temperature: float, response: str) -> None:
        """Store a response in the cache."""
        if not self.enabled:
            return

        key = self._make_key(prompt, model, temperature)
        path = self._cache_path(key)

        data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "response": response,
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[WARNING] Failed to write cache file {path}: {e}")
            return        










    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}
