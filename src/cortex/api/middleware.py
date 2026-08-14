"""HTTP middleware: request IDs + structured access logs, per-IP rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from cortex.logging import get_trace_id, log, new_trace_id, reset_trace_id, set_trace_id
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.api.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a trace id per request, logs a structured access line and tags
    the response with ``X-Request-ID`` so logs can be correlated with clients."""

    async def dispatch(self, request: Request, call_next):
        token = set_trace_id(new_trace_id())
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - started) * 1000)
            log(
                logger,
                logging.ERROR,
                "request failed",
                method=request.method,
                path=request.url.path,
                status=500,
                duration_ms=duration_ms,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = get_trace_id() or ""
        METRICS.inc(
            "http_requests_total",
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        )
        log(
            logger,
            logging.INFO,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        reset_trace_id(token)
        return response


class TokenBucket:
    """Token bucket rate limiter keyed by arbitrary string (per client IP)."""

    def __init__(self, rate_per_minute: int, capacity: int | None = None) -> None:
        self.rate = rate_per_minute / 60.0
        self.capacity = float(capacity or rate_per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str) -> bool:
        async with self._lock:
            now = time.monotonic()
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_per_minute: int = 120) -> None:
        super().__init__(app)
        self._bucket = TokenBucket(rate_per_minute)

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        if not await self._bucket.acquire(client):
            METRICS.inc("http_ratelimited_total", path=request.url.path)
            return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})
        return await call_next(request)
