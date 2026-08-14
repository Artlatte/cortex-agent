"""Bridge MCP server tools into native cortex ``Tool`` objects."""

from __future__ import annotations

from typing import Any

from cortex.agent.tools import Tool, ToolRegistry
from cortex.errors import MCPError
from cortex.mcp.client import MCPClient


async def mcp_to_agent_tools(
    client: MCPClient, registry: ToolRegistry | None = None
) -> list[Tool]:
    """Convert the client's MCP tools into cortex Tools, optionally registering them.

    Each tool's async func forwards the argument dict to ``client.call_tool`` and
    returns the normalised content. A failed MCP call raises :class:`MCPError` so
    ``Tool.run`` records a failed :class:`ToolResult`. Name collisions with tools
    already in ``registry`` (or with another MCP tool) are resolved by prefixing
    the server ``name``.
    """
    existing = {t.name for t in registry.list()} if registry is not None else set()
    converted: list[Tool] = []
    used = set(existing)
    for mtool in await client.list_tools():
        name = mtool.name if mtool.name not in used else f"{client.name}_{mtool.name}"
        used.add(name)

        async def invoke(_mcp_name: str = mtool.name, **kwargs: Any) -> str:
            result = await client.call_tool(_mcp_name, kwargs)
            if result.is_error:
                raise MCPError(f"MCP tool '{_mcp_name}' failed: {result.content}")
            return result.content

        tool = Tool(
            name=name,
            description=mtool.description,
            parameters=mtool.input_schema,
            func=invoke,
            is_async=True,
        )
        converted.append(tool)
        if registry is not None:
            registry.register(tool)
    return converted
