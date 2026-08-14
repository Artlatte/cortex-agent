"""Memory tests: short-term window/summarization and long-term SQLite memory."""

from __future__ import annotations

from cortex.agent.memory import LongTermMemory, ShortTermMemory
from cortex.llm.base import ChatMessage


async def test_short_term_summarization_compresses_history():
    async def fake_llm(messages: list[ChatMessage]) -> str:
        return "用户之前询问了天气情况"

    memory = ShortTermMemory(max_turns=2, summarize_llm=fake_llm)
    for i in range(10):
        memory.add(ChatMessage(role="user", content=f"消息{i}"))
    assert memory.is_full()
    changed = await memory.maybe_summarize()
    assert changed
    assert memory.summary == "用户之前询问了天气情况"
    assert len(memory.get_messages()) <= 4


async def test_short_term_truncates_without_llm():
    memory = ShortTermMemory(max_turns=2, summarize_enabled=True, summarize_llm=None)
    for i in range(10):
        memory.add(ChatMessage(role="user", content=f"消息{i}"))
    await memory.maybe_summarize()
    assert memory.summary is None
    assert len(memory.get_messages()) <= 4


async def test_long_term_memory_store_search_delete(tmp_path):
    memory = LongTermMemory(str(tmp_path / "mem.db"))
    await memory.store("python", "Python 是一种解释型编程语言，语法简洁")
    await memory.store("java", "Java 是一种编译型语言，常用于企业级开发")
    await memory.store("weather", "今天天气晴朗，适合出门")

    hits = await memory.search("Python 编程", top_k=2)
    assert hits and hits[0].key == "python"

    got = await memory.get("java")
    assert got is not None and "Java" in got.content

    assert await memory.count() == 3
    assert await memory.delete("weather")
    assert await memory.count() == 2


async def test_long_term_memory_upsert_by_key(tmp_path):
    memory = LongTermMemory(str(tmp_path / "mem.db"))
    await memory.store("k", "v1")
    await memory.store("k", "v2")
    assert await memory.count() == 1
    assert (await memory.get("k")).content == "v2"
