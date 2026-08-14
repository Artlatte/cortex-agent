"""Tests for the MCP stdio client and the tool bridge."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from cortex.errors import MCPError
from cortex.mcp import MCPClient, mcp_to_agent_tools

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_mcp_server.py"
SERVER_COMMAND = [sys.executable, str(FIXTURE_PATH)]


@pytest.fixture
async def client() -> AsyncIterator[MCPClient]:
    c = MCPClient(SERVER_COMMAND)
    await c.connect()
    try:
        yield c
    finally:
        await c.close()


async def test_connect_and_list_tools(client: MCPClient) -> None:
    assert client.is_connected is True
    tools = await client.list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"echo", "add"}
    assert by_name["echo"].description == "Echo the given text"
    assert by_name["echo"].input_schema["properties"]["text"] == {"type": "string"}
    assert by_name["echo"].input_schema["required"] == ["text"]
    assert by_name["add"].input_schema["properties"]["a"] == {"type": "integer"}


async def test_call_tool_round_trip(client: MCPClient) -> None:
    echo = await client.call_tool("echo", {"text": "hello"})
    assert echo.content == "hello"
    assert echo.is_error is False
    add = await client.call_tool("add", {"a": 2, "b": 3})
    assert add.content == "5"
    assert add.is_error is False


async def test_call_missing_tool_raises(client: MCPClient) -> None:
    with pytest.raises(MCPError):
        await client.call_tool("missing", {})


async def test_context_manager() -> None:
    c = MCPClient(SERVER_COMMAND)
    async with c as connected:
        assert connected.is_connected is True
        assert len(await connected.list_tools()) == 2
    assert c.is_connected is False


async def test_bridge(client: MCPClient) -> None:
    tools = await mcp_to_agent_tools(client)
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"echo", "add"}
    result = await by_name["echo"].run({"text": "hi"})
    assert result.output == "hi"
    assert result.error is None


async def test_timeout_raises_mcp_error() -> None:
    c = MCPClient(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        request_timeout=0.5,
    )
    try:
        with pytest.raises(MCPError):
            await asyncio.wait_for(c.list_tools(), timeout=5)
    finally:
        await c.close()
