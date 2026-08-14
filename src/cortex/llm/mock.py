"""Mock providers: deterministic LLM stand-ins for tests and offline demos.

``MockProvider`` is scripted (you enqueue exact responses). ``DemoProvider``
emulates a tool-using agent loop end-to-end without any network or API key, so
the examples in ``examples/`` run out of the box.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable
from typing import Any

from cortex.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, Usage


class MockProvider(LLMProvider):
    """Deterministic scripted provider. Call :meth:`enqueue` or pass a
    ``handler`` that receives ``(messages, tools, response_format)``."""

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-model",
        responses: list[LLMResponse] | None = None,
        handler: Callable[..., LLMResponse] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._queue: deque[LLMResponse] = deque(responses or [])
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, *responses: LLMResponse) -> None:
        self._queue.extend(responses)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "tools": tools, "response_format": response_format}
        )
        if self._handler is not None:
            return self._handler(messages, tools=tools, response_format=response_format)
        if not self._queue:
            raise RuntimeError("mock provider exhausted: no responses enqueued")
        return self._queue.popleft()


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_EXPR_RE = re.compile(
    r"(?:\([^()]*\)|\d+(?:\.\d+)?)(?:\s*[-+*/%^]\s*(?:\([^()]*\)|\d+(?:\.\d+)?))*"
)


def _guess_arguments(tool: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Heuristic argument filler for the offline demo."""
    schema = tool.get("parameters", {}) or {}
    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or list(properties)
    arguments: dict[str, Any] = {}
    for name in required:
        prop = properties.get(name, {})
        prop_type = prop.get("type", "string")
        if prop_type == "integer":
            numbers = _NUMBER_RE.findall(user_text)
            arguments[name] = int(float(numbers[0])) if numbers else 0
        elif prop_type == "number":
            numbers = _NUMBER_RE.findall(user_text)
            arguments[name] = float(numbers[0]) if numbers else 0.0
        elif name == "expression":
            match = _EXPR_RE.search(user_text)
            arguments[name] = match.group(0) if match else "1+1"
        else:
            arguments[name] = user_text.strip()
    return arguments


class DemoProvider(LLMProvider):
    """Offline provider that emulates a ReAct tool-using assistant.

    - If tools exist and the last message is a user question, it issues a tool
      call for the first registered tool with heuristically filled arguments.
    - Once tool results are present it synthesizes a final answer that quotes
      them, then never calls tools again (a trivial "memory" of progress).
    """

    def __init__(self, name: str = "demo", model: str = "demo-model") -> None:
        self.name = name
        self.model = model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        tool_results = [m for m in messages if m.role == "tool"]
        if response_format is not None:
            # Structured-output path: return a schema-shaped JSON object built
            # from the request, good enough for the offline planner demo.
            schema = response_format.get("json_schema", {})
            payload = _schema_example(schema)
            if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
                base = payload["steps"][0] if payload["steps"] else {}
                titles = ["调研背景与现状", "分析关键机制与实现", "整理结论与建议"]
                payload["steps"] = [
                    {
                        **base,
                        "step_id": i,
                        "title": title,
                        "description": f"完成「{title}」并给出证据",
                        "depends_on": [i - 1] if i > 1 else [],
                    }
                    for i, title in enumerate(titles, start=1)
                ]
                payload["reasoning"] = "离线演示模式：自动生成 3 步调研计划"
                user_text = next(
                    (m.content for m in reversed(messages) if m.role == "user"), ""
                )
                payload["goal"] = user_text[:100] or "演示任务"
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                provider=self.name,
                model=self.model,
            )
        if tools and not tool_results:
            tool = tools[0]
            user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="demo-call-1",
                        name=tool["name"],
                        arguments=_guess_arguments(tool, user_text),
                    )
                ],
                finish_reason="tool_calls",
                provider=self.name,
                model=self.model,
            )
        evidence = "\n".join(
            f"- [{m.name}] {m.content[:300]}" for m in tool_results
        )
        answer = (
            "我已经通过工具收集到了相关信息，下面是整理后的答案：\n\n"
            f"{evidence}\n\n"
            "（本回答由离线 DemoProvider 生成，用于演示 Agent 的 ReAct 工具调用流程。）"
        )
        return LLMResponse(
            content=answer,
            finish_reason="stop",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            provider=self.name,
            model=self.model,
        )


def _schema_example(schema: dict[str, Any]) -> Any:
    """Build a minimal valid example for a JSON schema (objects/arrays/primitives/$refs)."""

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if ref and ref.startswith("#/$defs/"):
            return schema.get("$defs", {}).get(ref.split("/")[-1], {})
        return node

    def example(node: dict[str, Any]) -> Any:
        node = resolve(node)
        if "anyOf" in node and node["anyOf"]:
            node = resolve(node["anyOf"][0])
        if "enum" in node and node["enum"]:
            return node["enum"][0]
        node_type = node.get("type", "object")
        if node_type == "object":
            out: dict[str, Any] = {}
            for key, prop in (node.get("properties") or {}).items():
                out[key] = example(prop)
            return out
        if node_type == "array":
            items = node.get("items") or {"type": "string"}
            return [example(items)]
        if node_type == "integer":
            return 1
        if node_type == "number":
            return 1.0
        if node_type == "boolean":
            return True
        return "示例值"

    return example(schema)
