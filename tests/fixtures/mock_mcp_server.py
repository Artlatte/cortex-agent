"""Minimal newline-delimited JSON-RPC 2.0 MCP server for tests (stdlib only).

Run directly: ``python mock_mcp_server.py``.
"""

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two integers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
]


def send(request_id, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def call_tool(name, arguments):
    if name == "echo":
        return {"content": [{"type": "text", "text": arguments.get("text", "")}]}
    if name == "add":
        total = arguments.get("a", 0) + arguments.get("b", 0)
        return {"content": [{"type": "text", "text": str(total)}]}
    return None


def main():
    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        if method == "initialize":
            send(
                request_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-server", "version": "0.1.0"},
                },
            )
        elif method == "tools/list":
            send(request_id, result={"tools": TOOLS})
        elif method == "tools/call":
            result = call_tool(params.get("name"), params.get("arguments", {}))
            if result is None:
                send(
                    request_id,
                    error={"code": -32602, "message": f"Unknown tool: {params.get('name')}"},
                )
            else:
                send(request_id, result=result)
        # notifications and unknown methods are ignored


if __name__ == "__main__":
    main()
