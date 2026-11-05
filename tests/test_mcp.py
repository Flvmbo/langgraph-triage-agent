"""The MCP server's tools, called directly and over a real stdio MCP session."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from triage_agent import mcp_server
from triage_agent.graph import build_graph, initial_state
from triage_agent.schemas import Ticket
from triage_agent.tools import load_tools

from .conftest import scripted_model, stub_decider, tool_call


def test_lookup_customer_known_and_unknown():
    assert mcp_server.lookup_customer("C-1003")["flags"] == ["suspicious_login"]
    assert "error" in mcp_server.lookup_customer("C-9999")


def test_get_policy_normalises_topic_and_lists_topics_on_miss():
    assert mcp_server.get_policy("Flagged Accounts")["topic"] == "flagged_accounts"
    miss = mcp_server.get_policy("teleportation")
    assert "priorities" in miss["available_topics"]


def test_service_status():
    assert mcp_server.check_service_status("exports")["incident"]["id"] == "INC-2042"
    assert set(mcp_server.check_service_status()) >= {"api", "dashboard", "exports"}


def test_every_eval_label_uses_a_valid_value():
    from typing import get_args

    from triage_agent.config import DATA_DIR
    from triage_agent.schemas import Action, Category, Priority

    for line in (DATA_DIR / "eval_tickets.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["expected"]["category"] in get_args(Category), row["id"]
        assert row["expected"]["priority"] in get_args(Priority), row["id"]
        assert row["expected"]["action"] in get_args(Action), row["id"]
        assert mcp_server.lookup_customer(row["customer_id"]).get("error") is None, row["id"]


async def test_tools_load_over_stdio_mcp():
    tools = await load_tools()
    assert {t.name for t in tools} == {"lookup_customer", "get_policy", "check_service_status"}


async def test_graph_runs_against_real_mcp_tools(decision_factory):
    tools = await load_tools()
    model = scripted_model(
        tool_call("lookup_customer", {"customer_id": "C-1003"}, "c1"),
        AIMessage(content="Customer is flagged."),
    )
    seen: list = []
    graph = build_graph(model, tools, decide_model=stub_decider(decision_factory(), seen), threshold=0.0)
    ticket = Ticket(id="T06", customer_id="C-1003", subject="Password not working", body="Send a reset link.")

    state = await graph.ainvoke(initial_state(ticket), {"configurable": {"thread_id": "mcp"}})

    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "suspicious_login" in str(result.content)
    assert "suspicious_login" in seen[0][-1].content
