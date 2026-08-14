"""Shared LLM types: messages, tool calls, usage and the provider protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ChatMessage:
    """A neutral chat message. Tool results carry ``name`` (tool name) and
    ``tool_call_id``; assistant messages that issued tool calls carry them in
    ``tool_calls`` so history can be replayed to the provider."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    provider: str | None = None
    model: str | None = None
    raw: Any = None


class LLMProvider(ABC):
    """Adapter contract. ``tools`` is a list of neutral tool descriptors
    ``{"name", "description", "parameters"}`` and ``response_format`` is either
    ``{"type": "json_schema", "json_schema": {...}}``, ``{"type": "json_object"}``
    or ``None`` — each adapter maps them to its wire format."""

    name: str
    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One non-streaming completion. Timeouts are configured per provider
        via ``ProviderConfig.timeout_seconds`` (httpx client level)."""

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Default streaming implementation falls back to a single chunk."""
        result = await self.chat(
            messages,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if result.content:
            yield result.content

    async def aclose(self) -> None:  # noqa: B027 - optional cleanup hook
        """Release resources (HTTP clients). Optional for adapters."""
