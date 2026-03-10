"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "config/default.yaml",
                overrides: dict | None = None) -> dict:
    """Load a YAML config file and apply optional overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base config."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
