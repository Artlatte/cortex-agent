"""Agent runtime: session lifecycle, concurrency control, idempotency and
SQLite persistence.

The runtime is the production surface between the API and the agent:

- **Concurrency**: a semaphore caps the number of simultaneously executing
  agents; overflow work is queued.
- **Idempotency**: ``idempotency_key`` maps to a session; re-submitting the
  same key returns the existing session instead of double-executing.
- **Persistence**: every status transition is written to SQLite so sessions
  survive restarts.
- **Cancellation**: queued/running sessions can be cancelled safely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortex.agent.react import AgentResult, ReActAgent
from cortex.logging import log, trace_span
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.agent.runtime")

VALID_STATUSES = ("queued", "running", "done", "failed", "cancelled")


@dataclass
class Session:
    session_id: str
    task: str
    status: str = "queued"
    answer: str | None = None
    error: str | None = None
    iterations: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    idempotency_key: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRuntime:
    def __init__(
        self,
        agent: ReActAgent,
        max_concurrency: int = 4,
        db_path: str | None = None,
    ) -> None:
        self.agent = agent
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sessions: dict[str, Session] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._idempotency: dict[str, str] = {}
        self._db_path = db_path
        self._queue_size = 0
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    # -- persistence --------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        assert self._db_path is not None
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    error TEXT,
                    iterations INTEGER NOT NULL DEFAULT 0,
                    steps TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    finished_at REAL
                )
                """
            )

    async def _persist(self, session: Session) -> None:
        if self._db_path is None:
            return
        await asyncio.to_thread(self._blocking_persist, session)

    def _blocking_persist(self, session: Session) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, idempotency_key, task, status, answer,
                    error, iterations, steps, created_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    idempotency_key = excluded.idempotency_key,
                    status = excluded.status,
                    answer = excluded.answer,
                    error = excluded.error,
                    iterations = excluded.iterations,
                    steps = excluded.steps,
                    finished_at = excluded.finished_at
                """,
                (
                    session.session_id,
                    session.idempotency_key,
                    session.task,
                    session.status,
                    session.answer,
                    session.error,
                    session.iterations,
                    json.dumps(session.steps, ensure_ascii=False),
                    session.created_at,
                    session.finished_at,
                ),
            )

    async def _find_by_idempotency_key(self, key: str) -> Session | None:
        if self._db_path is None:
            return None
        row = await asyncio.to_thread(self._blocking_find_by_key, key)
        if row is None:
            return None
        return self._row_to_session(row)

    def _blocking_find_by_key(self, key: str):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE idempotency_key = ?", (key,)
            ).fetchone()

    @staticmethod
    def _row_to_session(row) -> Session:
        return Session(
            session_id=row["session_id"],
            idempotency_key=row["idempotency_key"],
            task=row["task"],
            status=row["status"],
            answer=row["answer"],
            error=row["error"],
            iterations=row["iterations"],
            steps=json.loads(row["steps"]),
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )

    async def restore(self) -> int:
        """Reload persisted sessions at startup; unfinished ones are marked failed."""
        if self._db_path is None:
            return 0
        rows = await asyncio.to_thread(self._blocking_all_rows)
        restored = 0
        for row in rows:
            session = self._row_to_session(row)
            if session.status in ("queued", "running"):
                session.status = "failed"
                session.error = "interrupted by restart"
                session.finished_at = time.time()
                await self._persist(session)
            self._sessions[session.session_id] = session
            if session.idempotency_key:
                self._idempotency[session.idempotency_key] = session.session_id
            restored += 1
        return restored

    def _blocking_all_rows(self):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()

    # -- lifecycle -----------------------------------------------------------
    async def submit(
        self,
        task: str,
        session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Session:
        """Queue a task. Same idempotency_key → same session, executed once."""
        if idempotency_key:
            existing = self._idempotency.get(idempotency_key)
            if existing is not None and existing in self._sessions:
                METRICS.inc("runtime_submits_total", result="idempotent_hit")
                return self._sessions[existing]
            persisted = await self._find_by_idempotency_key(idempotency_key)
            if persisted is not None:
                self._idempotency[idempotency_key] = persisted.session_id
                self._sessions[persisted.session_id] = persisted
                METRICS.inc("runtime_submits_total", result="idempotent_hit")
                return persisted
        session = Session(
            session_id=session_id or uuid.uuid4().hex,
            task=task,
            idempotency_key=idempotency_key,
            created_at=time.time(),
        )
        self._sessions[session.session_id] = session
        if idempotency_key:
            self._idempotency[idempotency_key] = session.session_id
        await self._persist(session)
        task_obj = asyncio.create_task(self._run(session))
        self._tasks[session.session_id] = task_obj
        task_obj.add_done_callback(lambda _t, sid=session.session_id: self._tasks.pop(sid, None))
        METRICS.inc("runtime_submits_total", result="queued")
        METRICS.set("runtime_queue_size", self._queue_size)
        log(logger, logging.INFO, "session submitted", session_id=session.session_id, idempotency_key=idempotency_key)
        return session

    async def _run(self, session: Session) -> None:
        self._queue_size += 1
        METRICS.set("runtime_queue_size", self._queue_size)
        acquired = False
        started: float | None = None
        try:
            async with self._semaphore:
                acquired = True
                self._queue_size -= 1
                METRICS.set("runtime_queue_size", self._queue_size)
                if session.status == "cancelled":
                    return
                session.status = "running"
                await self._persist(session)
                started = time.time()
                with trace_span("runtime.session", session_id=session.session_id):
                    try:
                        result: AgentResult = await self.agent.run(
                            session.task, session_id=session.session_id
                        )
                    except asyncio.CancelledError:
                        session.status = "cancelled"
                        session.error = "cancelled by user"
                        raise
                    except Exception as exc:  # noqa: BLE001
                        session.status = "failed"
                        session.error = f"{type(exc).__name__}: {exc}"
                        log(
                            logger,
                            logging.ERROR,
                            "session failed",
                            session_id=session.session_id,
                            error=str(exc),
                        )
                    else:
                        session.status = "done"
                        session.answer = result.answer
                        session.iterations = result.iterations
                        session.steps = [asdict(step) for step in result.steps]
        except asyncio.CancelledError:
            if not acquired:  # cancelled while queued on the semaphore
                self._queue_size -= 1
                METRICS.set("runtime_queue_size", max(0, self._queue_size))
                session.status = "cancelled"
                session.error = "cancelled by user"
            raise
        finally:
            session.finished_at = time.time()
            await self._persist(session)
            METRICS.inc("runtime_sessions_total", status=session.status)
            if started is not None:
                METRICS.observe(
                    "runtime_session_duration_ms",
                    (session.finished_at - started) * 1000,
                    status=session.status,
                )

    async def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def get_status(self, session_id: str) -> dict[str, Any] | None:
        session = await self.get(session_id)
        if session is None:
            return None
        data = session.to_dict()
        data["running"] = session.status == "running"
        return data

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        sessions = sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)
        return [s.to_dict() for s in sessions[:limit]]

    async def cancel(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or session.status not in ("queued", "running"):
            return False
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
        else:
            session.status = "cancelled"
            session.finished_at = time.time()
            await self._persist(session)
        return True

    async def close(self) -> None:
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
