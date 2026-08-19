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
    """OpenAI API provider.

    Supports both legacy models (gpt-4o, gpt-4o-mini, etc.) and newer
    model families (gpt-5.x, o1, o3, o4) that require different API params:
      - Use ``max_completion_tokens`` instead of ``max_tokens``
      - Do not pass ``temperature`` (only default value 1 is accepted)
    """

    # Model name prefixes that require the new API parameter set.
    _NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def __init__(self, **kwargs):
        model = kwargs.pop("model", "gpt-5.4-mini")
        super().__init__(model=model, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _is_new_api_model(self) -> bool:
        return self.model.startswith(self._NEW_API_PREFIXES)

    def _call_api(self, prompt: str) -> str:
        if self._is_new_api_model():
            # gpt-5.x / o-series: use max_completion_tokens, no temperature
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.max_tokens,
            )
        else:
            # Legacy models (gpt-4o, gpt-4o-mini, etc.)
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
    """Mock LLM provider for testing.

    Returns canned JSON responses appropriate for the (game, mode) combination
    so tests run without real API calls. For IC, auto-detects SIGNAL vs DECIDE
    phase from prompt content.

    Calculator-mode responses omit the ``beliefs`` field (per the design rule
    in prompts_revised.md — Calculator JSON is bare).
    """

    _VALID_GAMES = ("pd", "cv", "ic")
    _VALID_MODES = ("calculator", "reasoner", "roleplayer")
    _VALID_SIGNAL_FORMS = ("categorical", "free_form_text")

    def __init__(self, *, game: str, mode: str,
                 signal_form: str = "categorical", **kwargs):
        if game not in self._VALID_GAMES:
            raise ValueError(
                f"unknown game: {game!r}. Expected one of {self._VALID_GAMES}"
            )
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"unknown mode: {mode!r}. Expected one of {self._VALID_MODES}"
            )
        if signal_form not in self._VALID_SIGNAL_FORMS:
            raise ValueError(
                f"unknown signal_form: {signal_form!r}. "
                f"Expected one of {self._VALID_SIGNAL_FORMS}"
            )
        # Mock has no real API — always force rate_limit_delay=0. The config
        # default (0.1 sec for real providers) would otherwise add ~5 hours
        # to a 1000-agent × 50-step run for no reason.
        kwargs["rate_limit_delay"] = 0
        if signal_form == "free_form_text" and (game != "ic" or mode != "roleplayer"):
            raise ValueError(
                "signal_form='free_form_text' is only valid for game='ic', mode='roleplayer'"
            )

        model = kwargs.pop("model", "mock-v1")
        super().__init__(model=model, **kwargs)
        self.game = game
        self.mode = mode
        self.signal_form = signal_form
        self._mock_responses = {}

    def set_response(self, prompt_substring: str, response: str):
        """Register a mock response override for prompts containing the substring."""
        self._mock_responses[prompt_substring] = response

    def _call_api(self, prompt: str) -> str:
        # Explicit overrides win
        for substring, response in self._mock_responses.items():
            if substring in prompt:
                return response

        if self.game == "pd":
            return self._pd_response()
        if self.game == "cv":
            return self._cv_response()
        if self.game == "ic":
            phase = self._detect_ic_phase(prompt)
            return self._ic_response(phase)
        # Defensive — should be unreachable due to __init__ validation
        raise ValueError(f"internal error: unknown game {self.game!r}")

    @staticmethod
    def _detect_ic_phase(prompt: str) -> str:
        """Detect SIGNAL vs DECIDE from IC prompt content."""
        signal_markers = (
            "PHASE 1: SIGNAL",
            "What signal do you broadcast",
            "what statement do you issue",
        )
        if any(marker in prompt for marker in signal_markers):
            return "signal"
        return "decide"

    def _beliefs_block(self) -> dict:
        # Calculator mode JSON omits beliefs (per prompts_revised.md)
        if self.mode == "calculator":
            return {}
        return {
            "beliefs": {
                "expected_neighbor_behavior": "uncertain",
                "risk_assessment": "low",
            }
        }

    def _pd_response(self) -> str:
        action_by_mode = {
            "calculator": "A",
            "reasoner": "COOPERATE",
            "roleplayer": "OPEN",
        }
        return json.dumps({
            "observed_state_summary": "Mock PD observation.",
            **self._beliefs_block(),
            "action": action_by_mode[self.mode],
            "confidence": 0.5,
            "short_rationale": "Mock PD response.",
        })

    def _cv_response(self) -> str:
        action_by_mode = {
            "calculator": "A",
            "reasoner": "QUIET",
            "roleplayer": "STAY_HOME",
        }
        return json.dumps({
            "observed_state_summary": "Mock CV observation.",
            **self._beliefs_block(),
            "action": action_by_mode[self.mode],
            "confidence": 0.5,
            "short_rationale": "Mock CV response.",
        })

    def _ic_response(self, phase: str) -> str:
        if phase == "signal":
            if self.signal_form == "free_form_text":
                return json.dumps({
                    "observed_state_summary": "Mock IC observation.",
                    **self._beliefs_block(),
                    "signal_text": "We maintain a balanced military posture.",
                    "confidence": 0.5,
                    "short_rationale": "Mock IC free-form signal.",
                })
            signal_by_mode = {
                "calculator": "S1",
                "reasoner": "STRONG",
                "roleplayer": "STRONG",
            }
            return json.dumps({
                "observed_state_summary": "Mock IC observation.",
                **self._beliefs_block(),
                "signal": signal_by_mode[self.mode],
                "confidence": 0.5,
                "short_rationale": "Mock IC signal.",
            })

        # DECIDE phase — generate 8 mock per-neighbor decisions
        decisions = []
        for i in range(8):
            if self.mode == "calculator":
                role = "role_X" if i % 2 == 0 else "role_Y"
                decisions.append({
                    "id": i,
                    "role": role,
                    "value": 0.5,
                    "rationale": "mock",
                })
            else:
                # Reasoner / Role-player share the same JSON shape
                if i % 2 == 0:
                    decisions.append({
                        "neighbor_id": i,
                        "role": "proposer",
                        "demand": 0.5,
                        "rationale": "mock",
                    })
                else:
                    decisions.append({
                        "neighbor_id": i,
                        "role": "responder",
                        "threshold": 0.5,
                        "rationale": "mock",
                    })
        return json.dumps({
            "observed_state_summary": "Mock IC observation.",
            **self._beliefs_block(),
            "decisions": decisions,
            "confidence": 0.5,
            "short_rationale": "Mock IC decide.",
        })


def get_provider(provider_name: str, cache: Optional[PromptCache] = None,
                 **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider by name."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
        "doubao": DoubaoSeedLiteProvider,
        "mock": MockProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. "
                         f"Available: {list(providers.keys())}")
    return providers[provider_name](cache=cache, **kwargs)
