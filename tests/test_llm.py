"""The model-agnostic layer: alias resolution and env overrides."""

from __future__ import annotations

import pytest

from triage_agent.config import load_model_registry
from triage_agent.llm import resolve_model_id


def test_alias_and_raw_ids_resolve():
    registry = {"claude": "anthropic:claude-sonnet-5-5"}
    assert resolve_model_id("claude", registry) == "anthropic:claude-sonnet-5-5"
    assert resolve_model_id("openai:some-model", registry) == "openai:some-model"


def test_unknown_alias_lists_known_ones():
    with pytest.raises(ValueError, match="claude"):
        resolve_model_id("nope", {"claude": "anthropic:x"})


def test_registry_has_every_provider_and_env_override(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_CLAUDE_HAIKU", "anthropic:override")
    registry = load_model_registry()
    assert {"claude", "openai", "gemini", "bedrock", "llama"} <= set(registry)
    assert registry["claude-haiku"] == "anthropic:override"
    assert all(":" in model_id for model_id in registry.values())
