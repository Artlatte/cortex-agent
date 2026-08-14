"""MCP stdio client: JSON-RPC 2.0 over newline-delimited stdio.

``MCPClient`` spawns an MCP server as a subprocess, performs the MCP handshake
(``initialize`` + ``notifications/initialized``), and routes request/response
pairs over stdout using monotonically increasing request ids. The child's
stderr is drained on a background task so it never blocks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from dataclasses import dataclass
from typing import Any

from cortex.errors import MCPError
from cortex.logging import log
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.mcp.client")


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPToolResult:
    """Result of calling an MCP tool."""

    content: str
    is_error: bool = False


class MCPClient:
    """Async MCP stdio client speaking newline-delimited JSON-RPC 2.0."""

    def __init__(
        self,
        command: list[str] | str,
        env: dict[str, str] | None = None,
        request_timeout: float = 30.0,
        name: str = "mcp",
    ) -> None:
        self._argv = shlex.split(command) if isinstance(command, str) else list(command)
        self._env = env  # None → inherit the parent environment
        self._request_timeout = request_timeout
        self.name = name  # used as the namespace prefix by the bridge

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}

    @property
    def is_connected(self) -> bool:
        """Whether the server process is spawned and still running."""
        return self._proc is not None and self._proc.returncode is None

    async def connect(self) -> None:
        """Spawn the server subprocess and complete the MCP handshake."""
        if self.is_connected:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._drain_task = asyncio.create_task(self._drain_stderr())
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cortex-agent", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> list[MCPTool]:
        """Return the tools advertised by the server (``tools/list``)."""
        await self._ensure_connected()
        result = await self._request("tools/list", {})
        tools = (result or {}).get("tools", [])
        return [
            MCPTool(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("inputSchema", {"type": "object"}),
            )
            for item in tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Invoke a server tool (``tools/call``) and normalise its content."""
        await self._ensure_connected()
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        parts: list[str] = []
        for item in (result or {}).get("content", []):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        is_error = bool((result or {}).get("isError", False))
        return MCPToolResult(content="".join(parts), is_error=is_error)

    async def close(self) -> None:
        """Cancel background tasks, terminate the child, and clear pending requests."""
        proc = self._proc
        self._proc = None

        tasks = [t for t in (self._reader_task, self._drain_task) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()

        self._fail_pending("MCP client closed")
        self._reader_task = None
        self._drain_task = None

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _ensure_connected(self) -> None:
        if not self.is_connected:
            await self.connect()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise MCPError("MCP client is not connected")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        METRICS.inc("mcp_requests_total", method=method)
        try:
            proc.stdin.write((json.dumps(payload) + "\n").encode())
            await proc.stdin.drain()
            return await asyncio.wait_for(future, timeout=self._request_timeout)
        except TimeoutError as exc:
            raise MCPError(
                f"MCP request '{method}' timed out after {self._request_timeout}s"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError("MCP client is not connected")
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        proc.stdin.write((json.dumps(payload) + "\n").encode())
        await proc.stdin.drain()

    async def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self._handle_line(line)
        except Exception as exc:  # noqa: BLE001 - surface reader failures as debug logs
            log(logger, logging.DEBUG, "mcp reader loop failed", error=str(exc))
        finally:
            self._fail_pending("MCP server closed the connection")

    def _handle_line(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            log(logger, logging.DEBUG, "mcp ignored invalid line", error=str(exc))
            return
        request_id = message.get("id")
        if request_id is None:
            log(logger, logging.DEBUG, "mcp notification", method=message.get("method"))
            return
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        if "error" in message:
            error = message["error"]
            future.set_exception(
                MCPError(f"MCP error {error.get('code')}: {error.get('message')}")
            )
        else:
            future.set_result(message.get("result"))

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            log(logger, logging.DEBUG, "mcp stderr", line=line.decode(errors="replace").rstrip())

    def _fail_pending(self, message: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(MCPError(message))
        self._pending.clear()
