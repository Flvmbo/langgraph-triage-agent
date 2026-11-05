"""Triage one ticket from the command line, with human review when confidence is low.

    triage --model claude --eval-id T06
    triage --model openai --customer C-1001 --subject "Forgot password" --body "I'm locked out"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from langgraph.types import Command

from .config import DATA_DIR
from .graph import build_graph, initial_state, total_tokens
from .llm import get_chat_model
from .schemas import Ticket
from .tools import load_tools


def _ticket_from_args(args: argparse.Namespace) -> Ticket:
    if args.eval_id:
        for line in (DATA_DIR / "eval_tickets.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] == args.eval_id:
                return Ticket(id=row["id"], subject=row["subject"], body=row["body"], customer_id=row["customer_id"])
        raise SystemExit(f"No eval ticket with id {args.eval_id}")
    if not (args.subject and args.body):
        raise SystemExit("Pass --eval-id, or both --subject and --body")
    return Ticket(id=args.id, subject=args.subject, body=args.body, customer_id=args.customer)


def _ask_review(payload: dict) -> dict:
    print(f"\nHuman review needed: {payload['reason']}")
    print(json.dumps(payload["decision"], indent=2))
    choice = input("\n[a]pprove or [o]verride? ").strip().lower()
    if choice.startswith("o"):
        override = {}
        for field in ("action", "priority", "team"):
            value = input(f"  new {field} (blank = keep): ").strip()
            if value:
                override[field] = value
        return {"override": override, "reviewer": "cli-user"}
    return {"approve": True, "reviewer": "cli-user"}


async def _run(args: argparse.Namespace) -> None:
    ticket = _ticket_from_args(args)
    tools = await load_tools()
    graph = build_graph(get_chat_model(args.model), tools)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = await graph.ainvoke(initial_state(ticket), config)
    while state.get("__interrupt__"):
        resume = _ask_review(state["__interrupt__"][0].value)
        state = await graph.ainvoke(Command(resume=resume), config)

    if args.trace:
        for m in state["messages"]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"tool call: {call['name']}({json.dumps(call['args'])})")
    print(json.dumps(state["decision"].model_dump(), indent=2, ensure_ascii=False))
    print(f"\nreviewed_by: {state.get('reviewed_by') or 'auto'} | tokens: {total_tokens(state)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage a support ticket with any LLM.")
    parser.add_argument("--model", default="claude", help="alias from models.toml or provider:model-id")
    parser.add_argument("--eval-id", help="use a ticket from data/eval_tickets.jsonl, e.g. T06")
    parser.add_argument("--id", default="CLI-1")
    parser.add_argument("--customer", help="customer ID, e.g. C-1001")
    parser.add_argument("--subject")
    parser.add_argument("--body")
    parser.add_argument("--trace", action="store_true", help="print the tool calls the agent made")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
