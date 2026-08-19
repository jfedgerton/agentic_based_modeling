"""Tests for the LLM provider abstraction."""

import json
import pytest

from src.llm.provider import MockProvider, get_provider
from src.llm.cache import PromptCache


class TestMockProvider:
    def test_default_pd_reasoner_response(self):
        provider = MockProvider(game="pd", mode="reasoner", rate_limit_delay=0)
        response = provider.query("What should I do?")
        data = json.loads(response)
        assert data["action"] == "COOPERATE"
        assert "confidence" in data
        assert "beliefs" in data  # reasoner mode includes beliefs

    def test_calculator_omits_beliefs(self):
        provider = MockProvider(game="pd", mode="calculator", rate_limit_delay=0)
        data = json.loads(provider.query("anything"))
        assert data["action"] == "A"
        assert "beliefs" not in data  # calculator JSON is bare per design rule

    def test_pd_roleplayer_returns_open(self):
        provider = MockProvider(game="pd", mode="roleplayer", rate_limit_delay=0)
        data = json.loads(provider.query("anything"))
        assert data["action"] == "OPEN"

    def test_cv_modes(self):
        for mode, expected in [("calculator", "A"), ("reasoner", "QUIET"), ("roleplayer", "STAY_HOME")]:
            provider = MockProvider(game="cv", mode=mode, rate_limit_delay=0)
            data = json.loads(provider.query("anything"))
            assert data["action"] == expected, f"mode={mode}"

    def test_ic_signal_phase_detected(self):
        provider = MockProvider(game="ic", mode="reasoner", rate_limit_delay=0)
        data = json.loads(provider.query("You are now in PHASE 1: SIGNAL.\nWhat signal do you broadcast?"))
        assert "signal" in data
        assert data["signal"] == "STRONG"

    def test_ic_decide_phase_default(self):
        provider = MockProvider(game="ic", mode="reasoner", rate_limit_delay=0)
        # Prompt without SIGNAL keywords → decide phase
        data = json.loads(provider.query("Now you must negotiate with each of your 8 neighbors."))
        assert "decisions" in data
        assert len(data["decisions"]) == 8

    def test_ic_calculator_decide_uses_id_and_value(self):
        provider = MockProvider(game="ic", mode="calculator", rate_limit_delay=0)
        data = json.loads(provider.query("Entities: ..."))  # decide phase
        d0 = data["decisions"][0]
        assert "id" in d0 and "value" in d0
        assert d0["role"] in ("role_X", "role_Y")

    def test_ic_freeform_signal(self):
        provider = MockProvider(game="ic", mode="roleplayer",
                                 signal_form="free_form_text", rate_limit_delay=0)
        data = json.loads(provider.query("PHASE 1: SIGNAL"))
        assert "signal_text" in data

    def test_custom_response_overrides_default(self):
        provider = MockProvider(game="pd", mode="reasoner", rate_limit_delay=0)
        custom = json.dumps({"action": "DEFECT", "confidence": 0.9,
                              "observed_state_summary": "test",
                              "beliefs": {}, "short_rationale": "test"})
        provider.set_response("defect_prompt", custom)
        response = provider.query("This is a defect_prompt test")
        data = json.loads(response)
        assert data["action"] == "DEFECT"

    def test_usage_stats(self):
        provider = MockProvider(game="pd", mode="reasoner", rate_limit_delay=0)
        provider.query("test")
        provider.query("test2")
        stats = provider.usage_stats
        assert stats["call_count"] == 2

    def test_caching_integration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PromptCache(cache_dir=tmpdir, enabled=True)
            provider = MockProvider(game="pd", mode="reasoner",
                                     cache=cache, rate_limit_delay=0)

            r1 = provider.query("cached prompt")
            r2 = provider.query("cached prompt")

            assert r1 == r2
            assert cache.stats["hits"] == 1
            assert cache.stats["misses"] == 1
            assert provider.usage_stats["call_count"] == 1

    def test_invalid_game_raises(self):
        with pytest.raises(ValueError, match="unknown game"):
            MockProvider(game="xx", mode="reasoner")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            MockProvider(game="pd", mode="xx")

    def test_freeform_only_for_ic_roleplayer(self):
        with pytest.raises(ValueError, match="free_form_text"):
            MockProvider(game="pd", mode="reasoner", signal_form="free_form_text")


class TestGetProvider:
    def test_mock_provider(self):
        provider = get_provider("mock", game="pd", mode="reasoner")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")
