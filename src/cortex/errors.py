"""Unified exception hierarchy for the Cortex platform."""

from __future__ import annotations


class CortexError(Exception):
    """Base class for every Cortex-specific error."""


class ConfigurationError(CortexError):
    """Invalid or missing configuration."""


class ProviderError(CortexError):
    """An upstream LLM provider rejected or failed a request."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class ProviderTimeoutError(ProviderError):
    """The upstream LLM provider timed out."""


class CircuitOpenError(ProviderError):
    """The circuit breaker for a provider is open and rejected the request."""


class AllProvidersFailedError(CortexError):
    """Every configured provider failed (including fallbacks)."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        detail = "; ".join(str(e) for e in errors)
        super().__init__(f"all providers failed: {detail}")


class ToolNotFoundError(CortexError):
    """The agent requested a tool that is not registered."""


class ToolExecutionError(CortexError):
    """A tool raised while executing."""


class StructuredOutputError(CortexError):
    """The model output could not be parsed/validated against the schema."""


class DocumentLoadError(CortexError):
    """A document could not be parsed."""


class MCPError(CortexError):
    """MCP transport or protocol failure."""
