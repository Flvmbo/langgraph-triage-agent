"""Compare models on the labelled ticket set.

    python evals/run_evals.py --models claude openai gemini
    python evals/run_evals.py --models claude-haiku --limit 5
    python evals/run_evals.py --models claude-opus claude claude-haiku --dataset hard

Each model runs the same graph and the same MCP tools on every ticket in
data/eval_tickets.jsonl. Scores: exact-match accuracy on category, priority and action,
plus "all three correct", error count, mean latency and total tokens. The hard set also
reports accuracy per trap type. Results go to evals/results.md (v1) or
evals/results_<dataset>.md, and per ticket to evals/results/<dataset>/<model>.jsonl.
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
DATASETS = {"v1": "eval_tickets.jsonl", "hard": "eval_tickets_hard.jsonl"}


def load_dataset(name: str = "v1", limit: int | None = None) -> list[dict]:
    path = DATA_DIR / DATASETS[name]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def score(decision: dict | None, expected: dict) -> dict[str, bool]:
    hits = {f: bool(decision) and decision.get(f) == expected[f] for f in FIELDS}
    hits["all"] = all(hits[f] for f in FIELDS)
    return hits


async def evaluate_model(alias: str, rows: list[dict], tools, concurrency: int, dataset: str = "v1") -> dict:
    graph = build_graph(get_chat_model(alias), tools, threshold=0.0)  # evals never pause for review
    sem = asyncio.Semaphore(concurrency)

    async def run_one(row: dict) -> dict:
        ticket = Ticket(id=row["id"], subject=row["subject"], body=row["body"], customer_id=row["customer_id"])
        async with sem:
            start = time.perf_counter()
            try:
                state = await graph.ainvoke(initial_state(ticket), {"configurable": {"thread_id": f"{alias}-{row['id']}"}})
                decision, error, tokens = state["decision"].model_dump(), None, total_tokens(state)
                tool_calls = [
                    {"name": c["name"], "args": c["args"]}
                    for m in state["messages"]
                    for c in (getattr(m, "tool_calls", None) or [])
                ]
            except Exception as exc:  # one bad ticket must not sink the whole run
                decision, error, tokens, tool_calls = None, f"{type(exc).__name__}: {exc}", 0, []
            latency = time.perf_counter() - start
        return {"id": row["id"], "trap": row.get("trap"), "expected": row["expected"], "decision": decision, "error": error,
                "latency_s": round(latency, 2), "tokens": tokens, "tool_calls": tool_calls,
                "hits": score(decision, row["expected"])}

    results = await asyncio.gather(*(run_one(r) for r in rows))
    out_dir = EVALS_DIR / "results" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = alias.replace(":", "_").replace("/", "_")
    (out_dir / f"{safe}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")

    n = len(results)
    acc = {f: sum(r["hits"][f] for r in results) / n for f in (*FIELDS, "all")}
    traps: dict[str, list[bool]] = {}
    for r in results:
        if r["trap"]:
            traps.setdefault(r["trap"], []).append(r["hits"]["all"])
    return {
        "by_trap": {t: (sum(h), len(h)) for t, h in traps.items()},
        "failures": [r["id"] for r in results if not r["hits"]["all"]],
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
    table = "\n".join([header, *rows]) + "\n"
    traps = sorted({t for s in summaries for t in s["by_trap"]})
    if traps:
        table += "\n**All three correct, by trap type**\n\n| Model | " + " | ".join(traps) + " | Failed tickets |\n"
        table += "|---" * (len(traps) + 2) + "|\n"
        for s in sorted(summaries, key=lambda s: s["all"], reverse=True):
            cells = [f"{s['by_trap'].get(t, (0, 0))[0]}/{s['by_trap'].get(t, (0, 0))[1]}" for t in traps]
            table += f"| {s['model']} | " + " | ".join(cells) + f" | {', '.join(s['failures']) or 'none'} |\n"
    return table


async def main_async(args: argparse.Namespace) -> None:
    rows = load_dataset(args.dataset, args.limit)
    tools = await load_tools()
    summaries = []
    for alias in args.models:
        print(f"evaluating {alias} on {len(rows)} tickets...")
        summaries.append(await evaluate_model(alias, rows, tools, args.concurrency, args.dataset))
    table = to_markdown(summaries)
    out = EVALS_DIR / ("results.md" if args.dataset == "v1" else f"results_{args.dataset}.md")
    out.write_text(f"# Eval results ({args.dataset} set)\n\n{len(rows)} tickets per model.\n\n{table}", encoding="utf-8")
    print("\n" + table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="aliases from models.toml or provider:model-id")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="v1", help="v1 (30 tickets) or hard (16 trap tickets)")
    parser.add_argument("--limit", type=int, help="only the first N tickets (quick smoke run)")
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
