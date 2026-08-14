"""Agent runtime tests: lifecycle, idempotency, concurrency, persistence."""

from __future__ import annotations

import asyncio

from conftest import FakeAgent

from cortex.agent.runtime import AgentRuntime, Session


async def _wait_for_status(runtime: AgentRuntime, session_id: str, wanted: str, tries: int = 200) -> dict:
    status = {}
    for _ in range(tries):
        task = runtime._tasks.get(session_id)
        task_done = task is None or task.done()
        status = await runtime.get_status(session_id)
        # Wait for BOTH the in-memory status AND the task's final persist,
        # otherwise restore() may read a stale (pre-persist) DB row.
        if status["status"] == wanted and task_done:
            break
        await asyncio.sleep(0.01)
    return status


async def test_submit_runs_to_completion():
    runtime = AgentRuntime(FakeAgent())
    session = await runtime.submit("任务A")
    assert session.status == "queued"
    status = await _wait_for_status(runtime, session.session_id, "done")
    assert status["status"] == "done"
    assert status["answer"] == "answer:任务A"


async def test_idempotency_key_dedupes_execution():
    agent = FakeAgent()
    runtime = AgentRuntime(agent)
    first = await runtime.submit("任务A", idempotency_key="k1")
    second = await runtime.submit("任务A", idempotency_key="k1")
    assert first.session_id == second.session_id
    await _wait_for_status(runtime, first.session_id, "done")
    assert len(agent.calls) == 1


async def test_concurrency_limit_queues_overflow():
    agent = FakeAgent(delay=0.05)
    runtime = AgentRuntime(agent, max_concurrency=2)
    sessions = [await runtime.submit(f"任务{i}") for i in range(4)]
    statuses = [await runtime.get_status(s.session_id) for s in sessions]
    assert statuses.count("running") <= 2
    for session in sessions:
        status = await _wait_for_status(runtime, session.session_id, "done", tries=400)
        assert status["status"] == "done"


async def test_cancel_queued_session():
    agent = FakeAgent(delay=0.2)
    runtime = AgentRuntime(agent, max_concurrency=1)
    first = await runtime.submit("任务1")
    second = await runtime.submit("任务2")
    await asyncio.sleep(0.02)  # let the first acquire the semaphore
    assert await runtime.cancel(second.session_id)
    status = await _wait_for_status(runtime, second.session_id, "cancelled")
    assert status["status"] == "cancelled"
    await _wait_for_status(runtime, first.session_id, "done", tries=400)


async def test_persistence_and_restore(tmp_path):
    db_path = str(tmp_path / "sessions.db")
    agent = FakeAgent()
    runtime = AgentRuntime(agent, db_path=db_path)
    session = await runtime.submit("任务X")
    await _wait_for_status(runtime, session.session_id, "done")
    # Simulate a session that was mid-flight when the process died.
    stale = Session(session_id="stale1", task="旧任务", status="running", created_at=1.0)
    await runtime._persist(stale)

    restored_runtime = AgentRuntime(agent, db_path=db_path)
    count = await restored_runtime.restore()
    assert count == 2
    done = await restored_runtime.get_status(session.session_id)
    assert done["status"] == "done"
    stale_status = await restored_runtime.get_status("stale1")
    assert stale_status["status"] == "failed"
    assert "restart" in stale_status["error"]
