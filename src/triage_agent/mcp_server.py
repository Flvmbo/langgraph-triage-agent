"""An MCP server exposing the support tools the agent can call.

Run standalone with ``triage-mcp`` (stdio transport). The agent starts it as a subprocess
through langchain-mcp-adapters, so any MCP-capable client (Claude Desktop, Claude Code,
another agent) can reuse exactly the same tools.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import DATA_DIR

mcp = FastMCP("triage-tools")


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@mcp.tool()
def lookup_customer(customer_id: str) -> dict[str, Any]:
    """Look up a customer by ID (e.g. "C-1001"): plan, account status, flags and renewal date."""
    customer = _load("customers.json").get(customer_id.strip())
    if customer is None:
        return {"error": f"No customer with id {customer_id!r}"}
    return {"customer_id": customer_id, **customer}


@mcp.tool()
def get_policy(topic: str) -> dict[str, Any]:
    """Return the support policy for a topic.

    Topics: priorities, refunds, double_charge, payment_methods, past_due, account_access,
    flagged_accounts, security_incidents, phishing, known_incidents, outages, bugs,
    feature_requests, cancellations, trials, privacy_requests, discounts.
    """
    policies = _load("policies.json")
    key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in policies:
        return {"error": f"Unknown topic {topic!r}", "available_topics": sorted(policies)}
    return {"topic": key, "policy": policies[key]}


@mcp.tool()
def check_service_status(component: str = "all") -> dict[str, Any]:
    """Return live service status and any active incident.

    Components: api, dashboard, exports, mobile_app, sso, or "all".
    """
    status = _load("service_status.json")
    key = component.strip().lower().replace(" ", "_").replace("-", "_")
    if key == "all":
        return status
    if key not in status:
        return {"error": f"Unknown component {component!r}", "available_components": sorted(status)}
    return {"component": key, **status[key]}


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
