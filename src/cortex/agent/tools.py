"""Tool abstractions shared by the agent loop and the MCP bridge.

A ``Tool`` carries a JSON-Schema parameter spec plus a callable, and knows how
to render itself for OpenAI / Anthropic / Gemini function-calling formats.
The ``@tool`` decorator derives the schema from type hints, including full
Pydantic models.
"""

from __future__ import annotations

import inspect
import json
import types
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from cortex.errors import ToolNotFoundError
from cortex.metrics import METRICS

NoneType = type(None)

_PY_TO_JSON: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
    list: {"type": "array", "items": {}},
}


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is None:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not NoneType]
        if len(inner) == 1:
            return _type_to_schema(inner[0])
        return {"anyOf": [_type_to_schema(a) for a in inner]}
    if origin is list:
        item_ann = get_args(annotation)
        return {"type": "array", "items": _type_to_schema(item_ann[0]) if item_ann else {}}
    if origin is dict:
        return {"type": "object"}
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    return dict(_PY_TO_JSON.get(annotation, {}))


def _schema_from_func(func: Callable) -> dict[str, Any]:
    """Build a JSON Schema for the callable's arguments from its signature."""
    hints: dict[str, Any] = {}
    with suppress(Exception):
        hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str
        schema = _type_to_schema(annotation)
        if not schema:
            schema = {"type": "string"}
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _stringify(output: Any) -> str:
    if output is None:
        return "done"
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except TypeError:
        return str(output)


@dataclass
class ToolResult:
    """Result of executing a tool; ``error`` is set instead of ``output`` on failure."""

    tool_name: str
    output: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_message(self) -> str:
        return self.error if self.error else self.output


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    is_async: bool = False

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool, converting any exception into a failed ToolResult."""
        try:
            if self.is_async:
                output = await self.func(**arguments)
            else:
                output = self.func(**arguments)
            METRICS.inc("tool_calls_total", tool=self.name, error="false")
            return ToolResult(tool_name=self.name, output=_stringify(output))
        except Exception as exc:  # noqa: BLE001
            METRICS.inc("tool_calls_total", tool=self.name, error="true")
            return ToolResult(
                tool_name=self.name, error=f"{type(exc).__name__}: {exc}"
            )


class ToolRegistry:
    """Ordered collection of tools with multi-provider schema rendering."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool_obj: Tool | Callable) -> Tool:
        if isinstance(tool_obj, Tool):
            self._tools[tool_obj.name] = tool_obj
            return tool_obj
        meta = getattr(tool_obj, "_cortex_tool", None)
        if meta is None:
            raise ToolNotFoundError(
                f"{tool_obj!r} is not a Tool; decorate it with @tool first"
            )
        self._tools[meta.name] = meta
        return meta

    def get(self, name: str) -> Tool:
        tool_obj = self._tools.get(name)
        if tool_obj is None:
            raise ToolNotFoundError(f"tool '{name}' is not registered")
        return tool_obj

    def has(self, name: str) -> bool:
        return name in self._tools

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # -- provider schemas ---------------------------------------------------
    def to_openai(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def to_anthropic(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._tools.values()
        ]

    def to_gemini(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": _to_gemini_schema(t.parameters),
            }
            for t in self._tools.values()
        ]


_GEMINI_TYPE_MAP = {
    "string": "STRING",
    "integer": "NUMBER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema to Gemini's uppercase function declaration schema."""

    def convert(node: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        json_type = node.get("type", "object")
        out["type"] = _GEMINI_TYPE_MAP.get(json_type, "STRING")
        if "enum" in node:
            out["type"] = "STRING"
            out["enum"] = node["enum"]
        if node.get("properties"):
            out["properties"] = {k: convert(v) for k, v in node["properties"].items()}
        if node.get("items"):
            out["items"] = convert(node["items"])
        if node.get("required"):
            out["required"] = node["required"]
        if node.get("description"):
            out["description"] = node["description"]
        return out

    return convert(schema)


def tool(name: str | None = None, description: str | None = None) -> Callable:
    """Decorator turning a typed function into a :class:`Tool`.

    Usage::

        @tool(description="Search the web")
        def web_search(query: str, max_results: int = 5) -> list[dict]:
            ...
    """

    def decorator(func: Callable) -> Callable:
        meta = Tool(
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or "",
            parameters=_schema_from_func(func),
            func=func,
            is_async=inspect.iscoroutinefunction(func),
        )
        func._cortex_tool = meta
        return func

    return decorator
