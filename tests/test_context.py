"""Context builder tests: token budget trimming."""

from __future__ import annotations

from cortex.agent.context import ContextBuilder
from cortex.agent.memory import ShortTermMemory
from cortex.agent.tools import ToolRegistry
from cortex.llm.base import ChatMessage


def test_context_builder_trims_old_messages_to_budget():
    memory = ShortTermMemory(max_turns=100)
    for i in range(50):
        memory.add(ChatMessage(role="user", content=f"历史消息{i}"))
    builder = ContextBuilder("系统提示词", ToolRegistry(), memory, token_budget=30)
    messages = builder.build("当前问题")
    assert messages[0].role == "system"
    assert messages[-1].content == "当前问题"
    assert len(messages) < 52  # 1 system + 1 user + trimmed history
    # The newest history survives, the oldest is dropped.
    contents = [m.content for m in messages]
    assert "历史消息49" in contents
    assert "历史消息0" not in contents


def test_context_builder_keeps_everything_when_budget_is_large():
    memory = ShortTermMemory(max_turns=100)
    memory.add(ChatMessage(role="user", content="历史1"))
    builder = ContextBuilder("系统提示词", ToolRegistry(), memory, token_budget=10000)
    messages = builder.build("当前问题")
    assert [m.content for m in messages] == ["系统提示词", "历史1", "当前问题"]
