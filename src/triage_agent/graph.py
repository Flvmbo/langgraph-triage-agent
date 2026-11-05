"""The triage graph.

    START -> gather <-> tools          (tool-calling loop over the MCP tools)
               |
               v
             decide                    (structured output: TriageDecision)
               |
     confidence < threshold?
        yes -> human_review -> END     (interrupt: a person approves or overrides)
        no  -> END
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .config import review_threshold
from .llm import with_structured_output
from .prompts import DECIDE_SYSTEM, GATHER_SYSTEM, format_ticket
from .schemas import Ticket, TriageDecision


class TriageState(MessagesState):
    ticket: Ticket
    decision: TriageDecision | None
    reviewed_by: str | None
    decide_usage: dict[str, Any] | None


def initial_state(ticket: Ticket) -> dict[str, Any]:
    return {
        "ticket": ticket,
        "messages": [HumanMessage(format_ticket(ticket))],
        "decision": None,
        "reviewed_by": None,
        "decide_usage": None,
    }


def _as_text(content: Any) -> str:
    """Normalise message content (str or provider-specific content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def evidence_transcript(messages: list[BaseMessage]) -> str:
    """Flatten the gather loop into provider-neutral text for the decision step."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(_as_text(m.content))
        elif isinstance(m, ToolMessage):
            lines.append(f"[tool result: {m.name}] {_as_text(m.content)}")
        elif isinstance(m, AIMessage) and not m.tool_calls:
            text = _as_text(m.content).strip()
            if text:
                lines.append(f"[evidence summary] {text}")
    return "\n\n".join(lines)


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    decide_model: Runnable | None = None,
    threshold: float | None = None,
    checkpointer: Any | None = None,
    max_tool_rounds: int = 6,
):
    """Compile the triage graph for any chat model and tool set.

    ``decide_model`` defaults to ``model.with_structured_output(TriageDecision)``; tests
    inject a stub. ``threshold`` defaults to TRIAGE_REVIEW_THRESHOLD (pass 0 to never
    interrupt, e.g. in evals).
    """
    threshold = review_threshold() if threshold is None else threshold
    gather_model = model.bind_tools(tools)
    decider = decide_model if decide_model is not None else with_structured_output(model, TriageDecision)

    async def gather(state: TriageState) -> dict[str, Any]:
        response = await gather_model.ainvoke([SystemMessage(GATHER_SYSTEM), *state["messages"]])
        return {"messages": [response]}

    def route_after_gather(state: TriageState) -> Literal["tools", "decide"]:
        last = state["messages"][-1]
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if isinstance(last, AIMessage) and last.tool_calls and rounds <= max_tool_rounds:
            return "tools"
        return "decide"

    async def decide(state: TriageState) -> dict[str, Any]:
        prompt = [
            SystemMessage(DECIDE_SYSTEM),
            HumanMessage(
                "Evidence gathered for this ticket:\n\n"
                + evidence_transcript(state["messages"])
                + "\n\nReturn the triage decision."
            ),
        ]
        out = await decider.ainvoke(prompt)
        usage = None
        if isinstance(out, dict) and "parsed" in out:  # include_raw=True shape
            if out.get("parsed") is None:
                raise ValueError(f"Model returned an invalid decision: {out.get('parsing_error')}")
            usage = getattr(out.get("raw"), "usage_metadata", None)
            out = out["parsed"]
        decision = out if isinstance(out, TriageDecision) else TriageDecision.model_validate(out)
        return {"decision": decision, "decide_usage": dict(usage) if usage else None}

    def route_after_decide(state: TriageState) -> Literal["human_review", "__end__"]:
        return "human_review" if state["decision"].confidence < threshold else END

    def human_review(state: TriageState) -> dict[str, Any]:
        decision = state["decision"]
        answer = interrupt(
            {
                "ticket_id": state["ticket"].id,
                "reason": f"confidence {decision.confidence:.2f} is below the review threshold {threshold}",
                "decision": decision.model_dump(),
            }
        )
        if not isinstance(answer, dict):
            answer = {"approve": bool(answer)}
        if answer.get("override"):
            decision = TriageDecision.model_validate({**decision.model_dump(), **answer["override"]})
        return {"decision": decision, "reviewed_by": answer.get("reviewer", "human")}

    builder = StateGraph(TriageState)
    builder.add_node("gather", gather)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("decide", decide)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "gather")
    builder.add_conditional_edges("gather", route_after_gather, ["tools", "decide"])
    builder.add_edge("tools", "gather")
    builder.add_conditional_edges("decide", route_after_decide, ["human_review", END])
    builder.add_edge("human_review", END)

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def total_tokens(state: dict[str, Any]) -> int:
    """Tokens used across the gather loop and the decision call (0 when providers omit usage)."""
    tokens = sum(
        (m.usage_metadata or {}).get("total_tokens", 0)
        for m in state.get("messages", [])
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None)
    )
    return tokens + (state.get("decide_usage") or {}).get("total_tokens", 0)
