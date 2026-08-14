"""Agent memory: short-term rolling window with summarization, and a
SQLite-backed long-term episodic memory with embedding retrieval."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cortex.llm.base import ChatMessage
from cortex.logging import log

logger = logging.getLogger("cortex.agent.memory")


class ShortTermMemory:
    """Rolling conversation window. When the window overflows, the oldest half
    is compressed into a summary (requires an LLM callable) so the agent keeps
    conversational continuity without unbounded context growth."""

    def __init__(
        self,
        max_turns: int = 12,
        summarize_enabled: bool = True,
        summarize_llm: Callable[[list[ChatMessage]], Awaitable[str]] | None = None,
    ) -> None:
        self.max_turns = max_turns
        self.summarize_enabled = summarize_enabled
        self._summarize_llm = summarize_llm
        self._messages: list[ChatMessage] = []
        self._summary: str | None = None

    @property
    def summary(self) -> str | None:
        return self._summary

    def add(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def add_many(self, messages: list[ChatMessage]) -> None:
        self._messages.extend(messages)

    def get_messages(self) -> list[ChatMessage]:
        return list(self._messages[-self.max_turns * 2 :])

    def is_full(self) -> bool:
        return len(self._messages) > self.max_turns * 2

    async def maybe_summarize(self) -> bool:
        """Compress the oldest half when the window overflows. Returns True if
        a summarization happened."""
        if not self.is_full() or not self.summarize_enabled:
            return False
        if self._summarize_llm is None:
            # Without an LLM, drop the oldest half (hard truncation).
            self._messages = self._messages[self.max_turns :]
            return True
        split = len(self._messages) - self.max_turns * 2
        old, recent = self._messages[:split], self._messages[split:]
        prompt = [
            ChatMessage(
                role="system",
                content="你是对话摘要助手，用不超过100字的中文概括以下对话的关键信息（用户目标、已确认事实、待办事项）。",
            )
        ] + old
        try:
            summary = await self._summarize_llm(prompt)
        except Exception as exc:  # noqa: BLE001 - summarization is best effort
            log(logger, logging.WARNING, "memory summarization failed, truncating", error=str(exc))
            summary = None
        if summary:
            self._summary = summary
        self._messages = recent
        return True

    def clear(self) -> None:
        self._messages.clear()
        self._summary = None


@dataclass
class MemoryHit:
    memory_id: str
    key: str
    content: str
    metadata: dict[str, Any]
    score: float


class LongTermMemory:
    """SQLite episodic memory. Each entry is a keyed fact with metadata and an
    embedding; retrieval ranks stored memories by cosine similarity. For demo
    scale this scans the table — production swaps in a vector database."""

    def __init__(
        self,
        db_path: str,
        embedder: Any | None = None,
    ) -> None:
        self.db_path = str(db_path)
        if embedder is None:
            from cortex.rag.embeddings import HashEmbedder

            embedder = HashEmbedder()
        self.embedder = embedder
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    key TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _blocking_store(self, key: str, content: str, metadata: dict[str, Any], embedding: list[float]) -> str:
        now = time.time()
        memory_id = f"mem-{int(now * 1000)}-{abs(hash(key)) % 10000}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (memory_id, key, content, metadata, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    content = excluded.content,
                    metadata = excluded.metadata,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    key,
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(embedding),
                    now,
                    now,
                ),
            )
        return memory_id

    async def store(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> str:
        embedding = await self.embedder.embed_query(content)
        return await asyncio.to_thread(
            self._blocking_store, key, content, metadata or {}, embedding
        )

    def _blocking_all(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT memory_id, key, content, metadata, embedding FROM memories"
            ).fetchall()
        return [dict(row) for row in rows]

    async def search(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        query_vector = await self.embedder.embed_query(query)
        rows = await asyncio.to_thread(self._blocking_all)

        def score_row(row: dict[str, Any]) -> MemoryHit:
            stored = json.loads(row["embedding"])
            dot = sum(a * b for a, b in zip(stored, query_vector, strict=True))
            norm = (sum(a * a for a in stored) ** 0.5) * (
                sum(b * b for b in query_vector) ** 0.5
            )
            similarity = dot / norm if norm else 0.0
            return MemoryHit(
                memory_id=row["memory_id"],
                key=row["key"],
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                score=similarity,
            )

        hits = sorted((score_row(row) for row in rows), key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    async def get(self, key: str) -> MemoryHit | None:
        hits = await asyncio.to_thread(self._blocking_get, key)
        return hits

    def _blocking_get(self, key: str) -> MemoryHit | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT memory_id, key, content, metadata, embedding FROM memories WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return MemoryHit(
            memory_id=row["memory_id"],
            key=row["key"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            score=1.0,
        )

    async def delete(self, key: str) -> bool:
        def blocking() -> bool:
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
                return cursor.rowcount > 0

        return await asyncio.to_thread(blocking)

    async def count(self) -> int:
        def blocking() -> int:
            with self._connect() as conn:
                return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        return await asyncio.to_thread(blocking)
