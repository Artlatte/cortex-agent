"""Shared fixtures for the Cortex test suite."""

from __future__ import annotations

import asyncio

import pytest

from cortex.agent.react import AgentResult
from cortex.llm.base import Usage
from cortex.metrics import METRICS


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Isolate metrics between tests (they are process-global)."""
    METRICS.reset()
    yield
    METRICS.reset()


class FakeAgent:
    """Deterministic stand-in for ReActAgent in runtime/orchestrator tests."""

    def __init__(self, delay: float = 0.0, prefix: str = "answer") -> None:
        self.delay = delay
        self.prefix = prefix
        self.calls: list[str] = []

    async def run(self, task: str, session_id: str | None = None) -> AgentResult:
        self.calls.append(task)
        if self.delay:
            await asyncio.sleep(self.delay)
        return AgentResult(
            answer=f"{self.prefix}:{task}",
            total_usage=Usage(total_tokens=7),
            iterations=1,
            session_id=session_id,
        )


class FakeAgentWithFailure:
    """Fails on the first N calls, then behaves like FakeAgent."""

    def __init__(self, failures: int = 1, prefix: str = "answer") -> None:
        self.failures = failures
        self.prefix = prefix
        self.calls: list[str] = []

    async def run(self, task: str, session_id: str | None = None) -> AgentResult:
        self.calls.append(task)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("simulated agent failure")
        return AgentResult(
            answer=f"{self.prefix}:{task}",
            total_usage=Usage(total_tokens=7),
            iterations=1,
            session_id=session_id,
        )


@pytest.fixture
def fake_agent() -> FakeAgent:
    return FakeAgent()
