"""Provider adapters: OpenAI-compatible, Anthropic and Gemini.

Each adapter translates the neutral format from :mod:`cortex.llm.base` to the
provider's wire format and normalizes the response back. A ``transport``
parameter lets tests inject :class:`httpx.MockTransport`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from cortex.config import ProviderConfig
from cortex.errors import ProviderError
from cortex.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, Usage

logger = logging.getLogger("cortex.llm")

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _parse_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _raise_http_error(provider: str, response: httpx.Response) -> None:
    raise ProviderError(
        provider,
        f"HTTP {response.status_code}: {response.text[:500]}",
        status_code=response.status_code,
        retryable=response.status_code in _RETRYABLE_STATUS,
    )


class OpenAICompatibleProvider(LLMProvider):
    """Covers OpenAI, DeepSeek, GLM, Moonshot, OpenRouter and any other
    OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = config.name
        self.model = config.model
        self._base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self._key = config.resolve_api_key()
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=config.timeout_seconds, transport=transport
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": [self._message(m) for m in messages]}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            response = await self._client.post("/chat/completions", json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._client.timeout}s", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"transport error: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            _raise_http_error(self.name, response)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call-{len(tool_calls)}",
                    name=fn.get("name") or "",
                    arguments=_parse_json(fn.get("arguments"), {}),
                )
            )
        usage_data = data.get("usage") or {}
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
            provider=self.name,
            model=self.model,
            raw=data,
        )

    @staticmethod
    def _message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in message.tool_calls
            ]
        return payload

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API adapter (Claude models)."""

    def __init__(
        self,
        config: ProviderConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = config.name
        self.model = config.model
        self._base_url = (config.base_url or "https://api.anthropic.com").rstrip("/")
        self._key = config.resolve_api_key()
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=config.timeout_seconds, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": "2023-06-01"}
        if self._key:
            headers["x-api-key"] = self._key
        return headers

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        if response_format and response_format.get("type") == "json_schema":
            system_parts.append(
                "You must respond with a single JSON object conforming to this JSON Schema:\n"
                + json.dumps(response_format["json_schema"], ensure_ascii=False)
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 1024,
            "messages": self._map_messages([m for m in messages if m.role != "system"]),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            response = await self._client.post("/v1/messages", json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"transport error: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            _raise_http_error(self.name, response)
        data = response.json()
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.get("id") or f"toolu-{len(tool_calls)}", name=block.get("name") or "", arguments=block.get("input") or {})
                )
        usage_data = data.get("usage") or {}
        return LLMResponse(
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            ),
            finish_reason=data.get("stop_reason"),
            provider=self.name,
            model=self.model,
            raw=data,
        )

    @staticmethod
    def _map_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                mapped.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or message.name or "unknown",
                                "content": message.content,
                            }
                        ],
                    }
                )
            elif message.role == "assistant":
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for tc in message.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                mapped.append({"role": "assistant", "content": content})
            else:
                mapped.append({"role": message.role, "content": message.content})
        return mapped

    async def aclose(self) -> None:
        await self._client.aclose()


_GEMINI_TYPE_MAP = {
    "string": "STRING",
    "integer": "NUMBER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _to_gemini_schema(node: dict[str, Any]) -> dict[str, Any]:
    """JSON Schema -> Gemini uppercase schema."""
    out: dict[str, Any] = {"type": _GEMINI_TYPE_MAP.get(node.get("type", "object"), "STRING")}
    if "enum" in node:
        out["type"] = "STRING"
        out["enum"] = node["enum"]
    if node.get("properties"):
        out["properties"] = {k: _to_gemini_schema(v) for k, v in node["properties"].items()}
    if node.get("items"):
        out["items"] = _to_gemini_schema(node["items"])
    if node.get("required"):
        out["required"] = node["required"]
    if node.get("description"):
        out["description"] = node["description"]
    return out


class GeminiProvider(LLMProvider):
    """Google Gemini generateContent adapter (supports function calling)."""

    def __init__(
        self,
        config: ProviderConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = config.name
        self.model = config.model
        self._base_url = (config.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        self._key = config.resolve_api_key()
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=config.timeout_seconds, transport=transport
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if response_format and response_format.get("type") == "json_schema":
            generation_config["response_mime_type"] = "application/json"
            generation_config["responseSchema"] = _to_gemini_schema(response_format["json_schema"])
        payload: dict[str, Any] = {
            "contents": self._map_messages([m for m in messages if m.role != "system"]),
            "generationConfig": generation_config,
        }
        system_parts = [m.content for m in messages if m.role == "system"]
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": _to_gemini_schema(t.get("parameters", {"type": "object"})),
                        }
                        for t in tools
                    ]
                }
            ]
        headers = {"x-goog-api-key": self._key} if self._key else {}
        url = f"/v1beta/models/{self.model}:generateContent"
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"transport error: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            _raise_http_error(self.name, response)
        data = response.json()
        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(id=f"{fc.get('name', 'fn')}-{len(tool_calls)}", name=fc.get("name") or "", arguments=fc.get("args") or {})
                )
        usage_data = data.get("usageMetadata") or {}
        return LLMResponse(
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            ),
            finish_reason=candidate.get("finishReason"),
            provider=self.name,
            model=self.model,
            raw=data,
        )

    @staticmethod
    def _map_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                mapped.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.name or "tool",
                                    "response": {"name": message.name or "tool", "content": message.content},
                                }
                            }
                        ],
                    }
                )
            elif message.role == "assistant":
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append({"text": message.content})
                for tc in message.tool_calls:
                    parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
                mapped.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:
                mapped.append({"role": message.role, "parts": [{"text": message.content}]})
        return mapped

    async def aclose(self) -> None:
        await self._client.aclose()


def build_provider(config: ProviderConfig, transport: httpx.AsyncBaseTransport | None = None) -> LLMProvider:
    """Factory: build the right adapter for a ProviderConfig."""
    if config.kind == "anthropic":
        return AnthropicProvider(config, transport=transport)
    if config.kind == "gemini":
        return GeminiProvider(config, transport=transport)
    if config.kind == "openai":
        return OpenAICompatibleProvider(config, transport=transport)
    if config.kind == "mock":
        from cortex.llm.mock import MockProvider

        return MockProvider(name=config.name, model=config.model)
    if config.kind == "demo":
        from cortex.llm.mock import DemoProvider

        return DemoProvider(name=config.name, model=config.model)
    raise ValueError(f"unknown provider kind: {config.kind}")
