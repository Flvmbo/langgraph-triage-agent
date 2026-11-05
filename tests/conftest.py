"""Offline test doubles: a scripted chat model that supports tool binding."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from triage_agent.schemas import TriageDecision


class ScriptedToolModel(GenericFakeChatModel):
    """Replays scripted AIMessages (including tool calls) and accepts bind_tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolModel":
        return self


def scripted_model(*messages: AIMessage) -> ScriptedToolModel:
    return ScriptedToolModel(messages=iter(messages))


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def stub_decider(decision: TriageDecision, seen: list | None = None) -> RunnableLambda:
    """Stands in for model.with_structured_output(TriageDecision); records the prompt it got."""

    def _decide(prompt):
        if seen is not None:
            seen.append(prompt)
        return decision

    return RunnableLambda(_decide)


@pytest.fixture
def decision_factory():
    def make(**overrides) -> TriageDecision:
        base = dict(category="account_access", priority="medium", action="auto_resolve", team=None,
                    confidence=0.92, rationale="Standard lockout per account_access policy.",
                    policies_applied=["account_access"], customer_reply="Here is your reset link.")
        return TriageDecision(**{**base, **overrides})

    return make
