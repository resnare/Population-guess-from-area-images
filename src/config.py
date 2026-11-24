"""Configuration loading with simple dotted-key command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _parse_value(text: str):
    """Turn a command-line string into a YAML scalar (int, float, bool, str)."""
    return yaml.safe_load(text)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply overrides such as ``train.epochs=60``."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return cfg
