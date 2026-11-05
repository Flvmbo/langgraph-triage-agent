"""Load the MCP server's tools as LangChain tools."""

from __future__ import annotations

import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def mcp_client() -> MultiServerMCPClient:
    """A client that launches our MCP server as a stdio subprocess.

    More servers (a CRM, a ticketing system) are one more entry in this dict.
    """
    return MultiServerMCPClient(
        {
            "triage": {
                "command": sys.executable,
                "args": ["-m", "triage_agent.mcp_server"],
                "transport": "stdio",
            }
        }
    )


async def load_tools() -> list[BaseTool]:
    return await mcp_client().get_tools()
