"""LLM provider abstraction layer.

Supports OpenAI, Anthropic, and a mock backend for testing.
All providers expose the same interface: send a prompt, get a string response.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from src.llm.cache import PromptCache


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model: str, temperature: float = 0.0,
                 max_tokens: int = 256, cache: Optional[PromptCache] = None,
                 rate_limit_delay: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.rate_limit_delay = rate_limit_delay
        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def query(self, prompt: str) -> str:
        """Send a prompt and return the response string, using cache if available."""
        if self.cache:
            cached = self.cache.get(prompt, self.model, self.temperature)
            if cached is not None:
                return cached

        # Rate limiting
        if self._call_count > 0 and self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

        response = self._call_api(prompt)
        self._call_count += 1

        if self.cache:
            self.cache.put(prompt, self.model, self.temperature, response)

        return response

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Provider-specific API call. Subclasses must implement."""
        pass

    @property
    def usage_stats(self) -> dict:
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "temperature": self.temperature,
            "call_count": self._call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "gpt-5.4-mini")
        super().__init__(model=model, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = response.usage
        if usage:
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "claude-sonnet-4-6")
        super().__init__(model=model, **kwargs)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _call_api(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        self._total_prompt_tokens += response.usage.input_tokens
        self._total_completion_tokens += response.usage.output_tokens
        return response.content[0].text


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "gemini-3.1-flash-lite")
        super().__init__(model=model, **kwargs)
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model)

    def _call_api(self, prompt: str) -> str:
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self._total_prompt_tokens += getattr(usage, "prompt_token_count", 0)
            self._total_completion_tokens += getattr(usage, "candidates_token_count", 0)
        return response.text


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider (uses OpenAI-compatible endpoint)."""

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "deepseek-v4-flash")
        super().__init__(model=model, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = response.usage
        if usage:
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
        return response.choices[0].message.content


class DoubaoSeedLiteProvider(LLMProvider):
    """Doubao Seed Lite API provider (Volcengine, OpenAI-compatible endpoint)."""

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "Doubao-Seed-2.0-lite")
        super().__init__(model=model, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("DOUBAO_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = response.usage
        if usage:
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
        return response.choices[0].message.content


class MockProvider(LLMProvider):
    """Mock LLM provider for testing. Returns deterministic responses."""

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "mock-v1")
        super().__init__(model=model, **kwargs)
        self._mock_responses = {}

    def set_response(self, prompt_substring: str, response: str):
        """Register a mock response for prompts containing the given substring."""
        self._mock_responses[prompt_substring] = response

    def _call_api(self, prompt: str) -> str:
        # Check for registered mock responses
        for substring, response in self._mock_responses.items():
            if substring in prompt:
                return response

        # Default: return a valid cooperate decision for PD
        return json.dumps({
            "observed_state_summary": "Mock observation of local neighborhood.",
            "beliefs": {
                "expected_neighbor_behavior": "uncertain",
                "risk_assessment": "low",
            },
            "action": "COOPERATE",
            "confidence": 0.5,
            "short_rationale": "Mock agent default response.",
        })


def get_provider(provider_name: str, cache: Optional[PromptCache] = None,
                 **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider by name."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
        "mock": MockProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. "
                         f"Available: {list(providers.keys())}")
    return providers[provider_name](cache=cache, **kwargs)
