"""Compare models on the labelled ticket set.

    python evals/run_evals.py --models claude openai gemini
    python evals/run_evals.py --models claude-haiku --limit 5

Each model runs the same graph and the same MCP tools on every ticket in
data/eval_tickets.jsonl. Scores: exact-match accuracy on category, priority and action,
plus "all three correct", error count, mean latency and total tokens. Results go to
evals/results.md (summary) and evals/results/<model>.jsonl (per ticket).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from triage_agent.config import DATA_DIR
from triage_agent.graph import build_graph, initial_state, total_tokens
from triage_agent.llm import get_chat_model
from triage_agent.schemas import Ticket
from triage_agent.tools import load_tools

EVALS_DIR = Path(__file__).resolve().parent
FIELDS = ("category", "priority", "action")


def load_dataset(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (DATA_DIR / "eval_tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def score(decision: dict | None, expected: dict) -> dict[str, bool]:
    hits = {f: bool(decision) and decision.get(f) == expected[f] for f in FIELDS}
    hits["all"] = all(hits[f] for f in FIELDS)
    return hits


async def evaluate_model(alias: str, rows: list[dict], tools, concurrency: int) -> dict:
    graph = build_graph(get_chat_model(alias), tools, threshold=0.0)  # evals never pause for review
    sem = asyncio.Semaphore(concurrency)

    async def run_one(row: dict) -> dict:
        ticket = Ticket(id=row["id"], subject=row["subject"], body=row["body"], customer_id=row["customer_id"])
        async with sem:
            start = time.perf_counter()
            try:
                state = await graph.ainvoke(initial_state(ticket), {"configurable": {"thread_id": f"{alias}-{row['id']}"}})
                decision, error, tokens = state["decision"].model_dump(), None, total_tokens(state)
            except Exception as exc:  # one bad ticket must not sink the whole run
                decision, error, tokens = None, f"{type(exc).__name__}: {exc}", 0
            latency = time.perf_counter() - start
        return {"id": row["id"], "expected": row["expected"], "decision": decision, "error": error,
                "latency_s": round(latency, 2), "tokens": tokens, "hits": score(decision, row["expected"])}

    results = await asyncio.gather(*(run_one(r) for r in rows))
    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    safe = alias.replace(":", "_").replace("/", "_")
    (out_dir / f"{safe}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")

    n = len(results)
    acc = {f: sum(r["hits"][f] for r in results) / n for f in (*FIELDS, "all")}
    return {
        "model": alias,
        **acc,
        "errors": sum(1 for r in results if r["error"]),
        "mean_latency_s": statistics.mean(r["latency_s"] for r in results),
        "tokens": sum(r["tokens"] for r in results),
        "n": n,
    }


def to_markdown(summaries: list[dict]) -> str:
    header = "| Model | Category | Priority | Action | All three | Errors | Mean latency | Tokens |\n|---|---|---|---|---|---|---|---|"
    rows = [
        f"| {s['model']} | {s['category']:.0%} | {s['priority']:.0%} | {s['action']:.0%} | {s['all']:.0%} "
        f"| {s['errors']}/{s['n']} | {s['mean_latency_s']:.1f}s | {s['tokens']:,} |"
        for s in sorted(summaries, key=lambda s: s["all"], reverse=True)
    ]
    return "\n".join([header, *rows]) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    rows = load_dataset(args.limit)
    tools = await load_tools()
    summaries = []
    for alias in args.models:
        print(f"evaluating {alias} on {len(rows)} tickets...")
        summaries.append(await evaluate_model(alias, rows, tools, args.concurrency))
    table = to_markdown(summaries)
    (EVALS_DIR / "results.md").write_text(f"# Eval results\n\n{len(rows)} tickets per model.\n\n{table}", encoding="utf-8")
    print("\n" + table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="aliases from models.toml or provider:model-id")
    parser.add_argument("--limit", type=int, help="only the first N tickets (quick smoke run)")
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
