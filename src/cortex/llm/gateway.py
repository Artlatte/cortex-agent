"""Multi-provider gateway with per-provider circuit breaker, retry with
exponential backoff + jitter, latency metrics and a fallback chain.

Resilience model
----------------
1. A request targets a primary provider (explicit ``provider=`` or the default).
2. That provider gets up to ``max_retries`` retries on retryable errors
   (429/5xx/timeouts) with exponential backoff and full jitter.
3. Consecutive failures trip a per-provider circuit breaker; while open the
   provider is skipped without burning latency. After a cooldown it allows a
   single half-open probe.
4. If a provider finally fails, the gateway falls back to the next provider
   (ordered by weight); only when all fail is ``AllProvidersFailedError`` raised.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Literal

from cortex.config import CircuitBreakerConfig
from cortex.errors import AllProvidersFailedError, CircuitOpenError, ProviderError
from cortex.llm.base import ChatMessage, LLMProvider, LLMResponse
from cortex.logging import log, trace_span
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.llm.gateway")


class CircuitBreaker:
    """Failure-count circuit breaker with cooldown and half-open probing."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._threshold = config.failure_threshold
        self._cooldown = config.cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at < self._cooldown:
            return "open"
        return "half_open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            self._failures = 0


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0


class LLMGateway:
    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        default_provider: str | None = None,
        circuit_breaker: CircuitBreakerConfig | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one LLM provider is required")
        self._providers = {p.name: p for p in providers}
        self.default_provider = default_provider or providers[0].name
        if self.default_provider not in self._providers:
            raise ValueError(f"default provider '{self.default_provider}' is not configured")
        breaker_cfg = circuit_breaker or CircuitBreakerConfig()
        self._breakers = {name: CircuitBreaker(breaker_cfg) for name in self._providers}
        self._retry = retry or RetryPolicy()

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers.values())

    def get_provider(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ValueError(f"unknown provider '{name}'") from exc

    def breaker_state(self, name: str) -> str:
        return self._breakers[name].state

    def _candidates(self, provider: str | None) -> list[str]:
        names = list(self._providers)
        if provider is not None:
            return [provider] + [n for n in names if n != provider]
        primary = self.default_provider
        rest = [n for n in names if n != primary]
        # stable fallback order: weight desc, then config order
        rest.sort(key=lambda n: (-getattr(self._providers[n], "weight", 1), names.index(n)))
        return [primary] + rest

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        provider: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        errors: list[Exception] = []
        for name in self._candidates(provider):
            provider_obj = self._providers[name]
            try:
                return await self._call_with_resilience(
                    provider_obj,
                    messages,
                    tools=tools,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - record and try next provider
                errors.append(exc)
                log(
                    logger,
                    logging.WARNING,
                    "provider failed, trying next in fallback chain",
                    provider=name,
                    error=str(exc),
                )
        raise AllProvidersFailedError(errors)

    async def _call_with_resilience(
        self,
        provider: LLMProvider,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        breaker = self._breakers[provider.name]
        if not breaker.allow():
            METRICS.inc("llm_requests_total", provider=provider.name, status="circuit_open")
            raise CircuitOpenError(provider.name, "circuit breaker is open")
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                with trace_span("llm.chat", provider=provider.name, model=provider.model, attempt=attempt):
                    result = await provider.chat(
                        messages,
                        tools=tools,
                        response_format=response_format,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                breaker.record_success()
                METRICS.set("llm_circuit_state", 0, provider=provider.name)
                METRICS.inc("llm_requests_total", provider=provider.name, status="ok")
                METRICS.observe("llm_latency_ms", (time.perf_counter() - started) * 1000, provider=provider.name)
                return result
            except Exception as exc:  # noqa: BLE001
                breaker.record_failure()
                METRICS.set("llm_circuit_state", 1 if breaker.state == "open" else 0, provider=provider.name)
                retryable = isinstance(exc, ProviderError) and exc.retryable
                status = "error" if retryable else "fatal"
                if isinstance(exc, ProviderError) and exc.status_code:
                    status = f"http_{exc.status_code}"
                METRICS.inc("llm_requests_total", provider=provider.name, status=status)
                if not retryable or attempt > self._retry.max_retries:
                    raise
                delay = min(self._retry.base_delay * (2 ** (attempt - 1)), self._retry.max_delay)
                delay *= random.uniform(0.5, 1.0)
                METRICS.inc("llm_retries_total", provider=provider.name)
                log(
                    logger,
                    logging.WARNING,
                    "retrying after provider error",
                    provider=provider.name,
                    attempt=attempt,
                    delay=round(delay, 3),
                    error=str(exc),
                )
                await asyncio.sleep(delay)

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        provider: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ):
        """Non-native streaming: one chunk per completion."""
        result = await self.chat(
            messages, provider=provider, tools=tools, temperature=temperature
        )
        if result.content:
            yield result.content

    async def aclose(self) -> None:
        await asyncio.gather(
            *[p.aclose() for p in self._providers.values()], return_exceptions=True
        )
