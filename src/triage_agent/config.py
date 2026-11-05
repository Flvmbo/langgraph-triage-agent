"""Paths, the model registry and runtime settings."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_FILE = ROOT / "models.toml"

load_dotenv(ROOT / ".env")


def load_model_registry(path: Path = MODELS_FILE) -> dict[str, str]:
    """Return alias -> "provider:model-id", applying TRIAGE_MODEL_<ALIAS> env overrides."""
    with path.open("rb") as f:
        registry: dict[str, str] = dict(tomllib.load(f).get("models", {}))
    for alias in list(registry):
        override = os.getenv(f"TRIAGE_MODEL_{alias.upper().replace('-', '_')}")
        if override:
            registry[alias] = override
    return registry


def review_threshold() -> float:
    """Decisions with confidence below this pause for human review."""
    return float(os.getenv("TRIAGE_REVIEW_THRESHOLD", "0.7"))
