"""Structured JSON logging with trace / span context propagation.

Every request that enters the API gets a ``trace_id``; each logical unit of
work runs inside a ``span`` so logs from LLM calls, tools and retrievers can
be correlated end-to-end.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("span_id", default=None)


def get_trace_id() -> str | None:
    return _trace_id.get()


def get_span_id() -> str | None:
    return _span_id.get()


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def set_trace_id(trace_id: str) -> contextvars.Token:
    """Set the current trace id (used by API middleware); returns a reset token."""
    return _trace_id.set(trace_id)


def reset_trace_id(token: contextvars.Token) -> None:
    """Restore the previous trace id from a token returned by :func:`set_trace_id`."""
    _trace_id.reset(token)


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a structured log line with extra JSON fields."""
    logger.log(level, message, extra={"extra_fields": fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
            "span_id": get_span_id(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # httpx/httpcore are extremely noisy at DEBUG; keep them quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@contextmanager
def trace_span(name: str, **attrs: Any) -> Iterator[None]:
    """Context manager that logs span start/end and propagates trace context."""
    logger = logging.getLogger("cortex.trace")
    trace_token = _trace_id.set(_trace_id.get() or new_trace_id())
    span_token = _span_id.set(uuid.uuid4().hex[:8])
    start = time.perf_counter()
    log(logger, logging.INFO, f"span.start {name}", **attrs)
    try:
        yield
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        log(logger, logging.INFO, f"span.end {name}", duration_ms=duration_ms, **attrs)
        _span_id.reset(span_token)
        _trace_id.reset(trace_token)
