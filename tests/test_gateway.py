"""Gateway resilience tests: retry, circuit breaker, fallback chain."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cortex.config import CircuitBreakerConfig, ProviderConfig
from cortex.errors import AllProvidersFailedError, CircuitOpenError, ProviderError
from cortex.llm.base import ChatMessage
from cortex.llm.gateway import LLMGateway, RetryPolicy
from cortex.llm.providers import OpenAICompatibleProvider


def make_provider(name: str, handler) -> OpenAICompatibleProvider:
    cfg = ProviderConfig(name=name, kind="openai", model="m1", base_url="https://test/v1")
    return OpenAICompatibleProvider(cfg, transport=httpx.MockTransport(handler))


def ok_payload(content: str = "hi", tool_calls: list | None = None) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def error_payload(status: int, message: str = "err") -> httpx.Response:
    return httpx.Response(status, json={"error": message}, request=httpx.Request("POST", "https://test/v1/chat/completions"))


MESSAGES = [ChatMessage(role="user", content="hello")]


async def test_chat_success_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "m1"
        return httpx.Response(
            200,
            json=ok_payload(
                content="calling",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"expr": "1+1"}'},
                    }
                ],
            ),
            request=request,
        )

    gateway = LLMGateway([make_provider("p1", handler)])
    result = await gateway.chat(MESSAGES, tools=[{"name": "calc", "description": "", "parameters": {}}])
    assert result.provider == "p1"
    assert result.content == "calling"
    assert result.tool_calls[0].name == "calc"
    assert result.tool_calls[0].arguments == {"expr": "1+1"}
    assert result.usage.total_tokens == 15


async def test_retry_then_success():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            return error_payload(429)
        return httpx.Response(200, json=ok_payload("finally"), request=request)

    gateway = LLMGateway(
        [make_provider("p1", handler)],
        retry=RetryPolicy(max_retries=2, base_delay=0.001, max_delay=0.01),
    )
    result = await gateway.chat(MESSAGES)
    assert result.content == "finally"
    assert attempts["count"] == 3


async def test_non_retryable_error_no_retry():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return error_payload(401)

    gateway = LLMGateway(
        [make_provider("p1", handler)],
        retry=RetryPolicy(max_retries=3, base_delay=0.001),
    )
    with pytest.raises(AllProvidersFailedError):
        await gateway.chat(MESSAGES)
    assert attempts["count"] == 1  # 401 must not be retried


async def test_fallback_to_next_provider():
    def handler_p1(request: httpx.Request) -> httpx.Response:
        return error_payload(500)

    def handler_p2(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_payload("from p2"), request=request)

    gateway = LLMGateway(
        [make_provider("p1", handler_p1), make_provider("p2", handler_p2)],
        retry=RetryPolicy(max_retries=0),
    )
    result = await gateway.chat(MESSAGES)
    assert result.provider == "p2"
    assert result.content == "from p2"


async def test_circuit_breaker_opens_and_half_open_recovers():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] >= 3:  # the rejected call never reaches the network
            return httpx.Response(200, json=ok_payload("recovered"), request=request)
        return error_payload(500)

    gateway = LLMGateway(
        [make_provider("p1", handler)],
        circuit_breaker=CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=0.05),
        retry=RetryPolicy(max_retries=0),
    )
    provider = gateway.get_provider("p1")
    # Two failures trip the breaker.
    with pytest.raises(ProviderError):
        await gateway._call_with_resilience(provider, MESSAGES, tools=None, response_format=None, temperature=None, max_tokens=None)
    with pytest.raises(ProviderError):
        await gateway._call_with_resilience(provider, MESSAGES, tools=None, response_format=None, temperature=None, max_tokens=None)
    assert gateway.breaker_state("p1") == "open"
    # While open, requests are rejected without hitting the network.
    with pytest.raises(CircuitOpenError):
        await gateway._call_with_resilience(provider, MESSAGES, tools=None, response_format=None, temperature=None, max_tokens=None)
    assert attempts["count"] == 2
    # After the cooldown, a half-open probe is allowed and succeeds.
    await asyncio.sleep(0.06)
    assert gateway.breaker_state("p1") == "half_open"
    result = await gateway._call_with_resilience(provider, MESSAGES, tools=None, response_format=None, temperature=None, max_tokens=None)
    assert result.content == "recovered"
    assert gateway.breaker_state("p1") == "closed"


async def test_timeout_is_retryable():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=ok_payload("after timeout"), request=request)

    gateway = LLMGateway(
        [make_provider("p1", handler)],
        retry=RetryPolicy(max_retries=1, base_delay=0.001),
    )
    result = await gateway.chat(MESSAGES)
    assert result.content == "after timeout"
    assert attempts["count"] == 2


async def test_all_providers_failed_aggregates_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return error_payload(500, "boom")

    gateway = LLMGateway(
        [make_provider("p1", handler), make_provider("p2", handler)],
        retry=RetryPolicy(max_retries=0),
    )
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await gateway.chat(MESSAGES)
    assert len(exc_info.value.errors) == 2
    assert all(isinstance(e, ProviderError) for e in exc_info.value.errors)
