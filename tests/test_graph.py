"""Graph behaviour with a scripted model: tool loop, decision, human-in-the-loop, limits."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from triage_agent.graph import build_graph, evidence_transcript, initial_state
from triage_agent.schemas import Ticket

from .conftest import scripted_model, stub_decider, tool_call

TICKET = Ticket(id="T05", customer_id="C-1001", subject="Forgot my password", body="I'm locked out.")
CONFIG = {"configurable": {"thread_id": "test"}}


@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer."""
    return {"customer_id": customer_id, "plan": "starter", "flags": []}


@tool
def get_policy(topic: str) -> dict:
    """Return a policy."""
    return {"topic": topic, "policy": "Standard lockouts get a reset link (auto_resolve)."}


TOOLS = [lookup_customer, get_policy]


async def test_tool_loop_then_decision(decision_factory):
    model = scripted_model(
        tool_call("lookup_customer", {"customer_id": "C-1001"}, "c1"),
        tool_call("get_policy", {"topic": "account_access"}, "c2"),
        AIMessage(content="Starter customer, no flags; standard lockout policy applies."),
    )
    seen: list = []
    graph = build_graph(model, TOOLS, decide_model=stub_decider(decision_factory(), seen), threshold=0.7)

    state = await graph.ainvoke(initial_state(TICKET), CONFIG)

    assert state["decision"].action == "auto_resolve"
    assert state["reviewed_by"] is None
    tool_names = [m.name for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_names == ["lookup_customer", "get_policy"]
    # the decision step sees the tool evidence, not raw provider messages
    decide_prompt = seen[0][-1].content
    assert "[tool result: lookup_customer]" in decide_prompt
    assert "[evidence summary] Starter customer" in decide_prompt


async def test_low_confidence_interrupts_and_override_applies(decision_factory):
    model = scripted_model(AIMessage(content="Not enough evidence."))
    low = decision_factory(confidence=0.4)
    graph = build_graph(model, TOOLS, decide_model=stub_decider(low), threshold=0.7)

    state = await graph.ainvoke(initial_state(TICKET), CONFIG)
    assert "__interrupt__" in state
    payload = state["__interrupt__"][0].value
    assert payload["ticket_id"] == "T05" and payload["decision"]["confidence"] == 0.4

    state = await graph.ainvoke(
        Command(resume={"override": {"action": "escalate_to_human", "priority": "urgent"}, "reviewer": "lead"}),
        CONFIG,
    )
    assert state["decision"].action == "escalate_to_human"
    assert state["decision"].priority == "urgent"
    assert state["reviewed_by"] == "lead"


async def test_approval_keeps_decision(decision_factory):
    model = scripted_model(AIMessage(content="done"))
    graph = build_graph(model, TOOLS, decide_model=stub_decider(decision_factory(confidence=0.5)), threshold=0.7)
    await graph.ainvoke(initial_state(TICKET), CONFIG)
    state = await graph.ainvoke(Command(resume={"approve": True}), CONFIG)
    assert state["decision"].action == "auto_resolve"
    assert state["reviewed_by"] == "human"


async def test_tool_round_limit_stops_runaway_loops(decision_factory):
    calls = [tool_call("get_policy", {"topic": "refunds"}, f"c{i}") for i in range(10)]
    graph = build_graph(scripted_model(*calls), TOOLS, decide_model=stub_decider(decision_factory()),
                        threshold=0.0, max_tool_rounds=3)
    state = await graph.ainvoke(initial_state(TICKET), CONFIG)
    assert sum(isinstance(m, ToolMessage) for m in state["messages"]) == 3
    assert state["decision"] is not None


def test_evidence_transcript_handles_content_blocks():
    msgs = [
        ToolMessage(content=[{"type": "text", "text": '{"plan": "pro"}'}], name="lookup_customer", tool_call_id="x"),
        AIMessage(content=[{"type": "text", "text": "Pro customer."}]),
    ]
    text = evidence_transcript(msgs)
    assert '[tool result: lookup_customer] {"plan": "pro"}' in text
    assert "[evidence summary] Pro customer." in text
