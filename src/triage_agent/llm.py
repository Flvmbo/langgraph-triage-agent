"""The model-agnostic LLM layer.

Every part of the agent talks to a LangChain ``BaseChatModel``; which provider sits behind
it is decided here, from a single alias or "provider:model-id" string. Adding a provider is
a models.toml entry plus its langchain-<provider> package - no graph code changes.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from .config import load_model_registry


def resolve_model_id(name: str, registry: dict[str, str] | None = None) -> str:
    """Map an alias (``claude``) or a raw ``provider:model-id`` string to a model ID."""
    registry = load_model_registry() if registry is None else registry
    if name in registry:
        return registry[name]
    if ":" in name:
        return name
    known = ", ".join(sorted(registry)) or "(none)"
    raise ValueError(f"Unknown model {name!r}. Use an alias ({known}) or 'provider:model-id'.")


def get_chat_model(
    name: str,
    *,
    temperature: float | None = None,
    registry: dict[str, str] | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Build a chat model for any supported provider.

    ``temperature`` is only forwarded when set: some reasoning models reject it.
    """
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(resolve_model_id(name, registry), **kwargs)


# Providers with native JSON-schema structured output. Newer Claude models do not support
# structured output via forced tool calling, so "json_schema" is required there; for the
# others it is the most reliable mode. Providers not listed use LangChain's default.
_JSON_SCHEMA_PROVIDERS = {"ChatAnthropic", "ChatOpenAI", "ChatGoogleGenerativeAI", "ChatOllama"}


def with_structured_output(model: BaseChatModel, schema: type) -> Runnable:
    """``model.with_structured_output(schema)`` using the best method for its provider.

    Returns the include_raw shape ({"raw", "parsed", "parsing_error"}) so callers can read
    token usage from the raw message.
    """
    if type(model).__name__ in _JSON_SCHEMA_PROVIDERS:
        return model.with_structured_output(schema, method="json_schema", include_raw=True)
    return model.with_structured_output(schema, include_raw=True)
