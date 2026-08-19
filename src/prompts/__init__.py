"""Prompt template registry for the three-game benchmark.

Single entry point :func:`get_prompt` returns the right template string for a
given ``(game, mode, ...)`` combination. Template strings still contain
``{name}`` placeholders that callers fill in with ``str.format(**vars)``.

For IC, the ``{BLUFFING_HINT}`` placeholder is filled at runtime with either
:data:`BLUFFING_HINT` (when ``bluffing_enabled=True``) or an empty string.
Calculator mode templates do not contain this placeholder.
"""

from typing import Literal, Optional

from src.prompts.pd_prompts import (
    PD_CALCULATOR_PROMPT,
    PD_REASONER_PROMPT,
    PD_ROLEPLAYER_PROMPT,
)
from src.prompts.cv_prompts import (
    CV_CALCULATOR_EN_PROMPT, CV_REASONER_EN_PROMPT, CV_ROLEPLAYER_EN_PROMPT,
    CV_CALCULATOR_ZH_PROMPT, CV_REASONER_ZH_PROMPT, CV_ROLEPLAYER_ZH_PROMPT,
)
from src.prompts.ic_prompts import (
    BLUFFING_HINT,
    IC_CALCULATOR_SIGNAL_PROMPT, IC_CALCULATOR_DECIDE_PROMPT,
    IC_REASONER_SIGNAL_PROMPT, IC_REASONER_DECIDE_PROMPT,
    IC_ROLEPLAYER_CATEGORICAL_SIGNAL_PROMPT, IC_ROLEPLAYER_CATEGORICAL_DECIDE_PROMPT,
    IC_ROLEPLAYER_FREEFORM_SIGNAL_PROMPT, IC_ROLEPLAYER_FREEFORM_DECIDE_PROMPT,
)


Game = Literal["pd", "cv", "ic"]
Mode = Literal["calculator", "reasoner", "roleplayer"]
Phase = Literal["signal", "decide"]
Language = Literal["en", "zh"]
SignalForm = Literal["categorical", "free_form_text"]


_PD = {
    "calculator": PD_CALCULATOR_PROMPT,
    "reasoner": PD_REASONER_PROMPT,
    "roleplayer": PD_ROLEPLAYER_PROMPT,
}

_CV = {
    ("calculator", "en"): CV_CALCULATOR_EN_PROMPT,
    ("calculator", "zh"): CV_CALCULATOR_ZH_PROMPT,
    ("reasoner", "en"): CV_REASONER_EN_PROMPT,
    ("reasoner", "zh"): CV_REASONER_ZH_PROMPT,
    ("roleplayer", "en"): CV_ROLEPLAYER_EN_PROMPT,
    ("roleplayer", "zh"): CV_ROLEPLAYER_ZH_PROMPT,
}

_IC = {
    ("calculator", "signal", "categorical"): IC_CALCULATOR_SIGNAL_PROMPT,
    ("calculator", "decide", "categorical"): IC_CALCULATOR_DECIDE_PROMPT,
    ("reasoner", "signal", "categorical"): IC_REASONER_SIGNAL_PROMPT,
    ("reasoner", "decide", "categorical"): IC_REASONER_DECIDE_PROMPT,
    ("roleplayer", "signal", "categorical"): IC_ROLEPLAYER_CATEGORICAL_SIGNAL_PROMPT,
    ("roleplayer", "decide", "categorical"): IC_ROLEPLAYER_CATEGORICAL_DECIDE_PROMPT,
    ("roleplayer", "signal", "free_form_text"): IC_ROLEPLAYER_FREEFORM_SIGNAL_PROMPT,
    ("roleplayer", "decide", "free_form_text"): IC_ROLEPLAYER_FREEFORM_DECIDE_PROMPT,
}


def get_prompt(
    game: Game,
    mode: Mode,
    phase: Optional[Phase] = None,
    language: Language = "en",
    signal_form: SignalForm = "categorical",
) -> str:
    """Return the prompt template for the given combination.

    Args:
        game: ``"pd"``, ``"cv"``, or ``"ic"``.
        mode: ``"calculator"``, ``"reasoner"``, or ``"roleplayer"``.
        phase: ``"signal"`` or ``"decide"`` — required for IC, must be None for
            PD/CV.
        language: ``"en"`` or ``"zh"`` — only used for CV; must be ``"en"`` for
            PD/IC.
        signal_form: ``"categorical"`` or ``"free_form_text"`` — only used for
            IC roleplayer; must be ``"categorical"`` otherwise.

    Raises:
        ValueError: if the parameter combination is invalid.
    """
    if game == "pd":
        if phase is not None:
            raise ValueError(f"phase is only used for IC; got phase={phase!r} for PD")
        if language != "en":
            raise ValueError(f"language is only used for CV; got language={language!r} for PD")
        if signal_form != "categorical":
            raise ValueError(
                f"signal_form is only used for IC roleplayer; got signal_form={signal_form!r} for PD"
            )
        if mode not in _PD:
            raise ValueError(f"unknown mode for PD: {mode!r}")
        return _PD[mode]

    if game == "cv":
        if phase is not None:
            raise ValueError(f"phase is only used for IC; got phase={phase!r} for CV")
        if signal_form != "categorical":
            raise ValueError(
                f"signal_form is only used for IC roleplayer; got signal_form={signal_form!r} for CV"
            )
        key = (mode, language)
        if key not in _CV:
            raise ValueError(f"unknown (mode, language) for CV: {key!r}")
        return _CV[key]

    if game == "ic":
        if phase not in ("signal", "decide"):
            raise ValueError(f"phase must be 'signal' or 'decide' for IC; got {phase!r}")
        if language != "en":
            raise ValueError(f"language is only used for CV; got language={language!r} for IC")
        if signal_form == "free_form_text" and mode != "roleplayer":
            raise ValueError(
                f"signal_form='free_form_text' is only valid for IC roleplayer; got mode={mode!r}"
            )
        key = (mode, phase, signal_form)
        if key not in _IC:
            raise ValueError(f"unknown (mode, phase, signal_form) for IC: {key!r}")
        return _IC[key]

    raise ValueError(f"unknown game: {game!r}")


__all__ = ["get_prompt", "BLUFFING_HINT"]
