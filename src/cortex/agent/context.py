"""Agent context engineering: system prompt assembly and token-budget trimming."""

from __future__ import annotations

import logging

from cortex.agent.memory import ShortTermMemory
from cortex.agent.tools import ToolRegistry
from cortex.llm.base import ChatMessage
from cortex.logging import log

logger = logging.getLogger("cortex.agent.context")


class ContextBuilder:
    """Builds the message list for one agent turn.

    Composition: system prompt + tool manifest + memory summary + recent
    history + current user input, trimmed to a token budget (heuristic:
    ~3 chars per token, safe for CJK). The newest messages are always kept.
    """

    def __init__(
        self,
        system_prompt: str,
        registry: ToolRegistry,
        memory: ShortTermMemory,
        token_budget: int = 8000,
    ) -> None:
        self.system_prompt = system_prompt
        self.registry = registry
        self.memory = memory
        self.token_budget = token_budget

    def _system_message(self) -> ChatMessage:
        tools_manifest = "\n".join(
            f"- {t.name}: {t.description}" for t in self.registry.list()
        )
        content = self.system_prompt
        if tools_manifest:
            content += "\n\n可用工具:\n" + tools_manifest
        return ChatMessage(role="system", content=content)

    @staticmethod
    def _estimate_tokens(message: ChatMessage) -> int:
        return max(1, len(message.content) // 3)

    def build(self, user_input: str) -> list[ChatMessage]:
        messages: list[ChatMessage] = [self._system_message()]
        summary = self.memory.summary
        if summary:
            messages.append(ChatMessage(role="system", content=f"[对话历史摘要] {summary}"))
        messages.extend(self.memory.get_messages())
        messages.append(ChatMessage(role="user", content=user_input))

        budget = self.token_budget
        # Always keep the system messages and the newest user message.
        used = sum(self._estimate_tokens(m) for m in messages[:1]) + self._estimate_tokens(
            messages[-1]
        )
        if summary:
            used += self._estimate_tokens(messages[1])
        body = messages[1:-1] if not summary else messages[2:-1]
        kept: list[ChatMessage] = []
        for message in reversed(body):
            cost = self._estimate_tokens(message)
            if used + cost > budget:
                break
            kept.append(message)
            used += cost
        kept.reverse()
        truncated = len(body) - len(kept)
        if truncated > 0:
            log(logger, logging.INFO, "context trimmed to token budget", dropped=truncated)
            kept.insert(
                0,
                ChatMessage(
                    role="system",
                    content=f"(为控制上下文长度，已省略较早的 {truncated} 条消息)",
                ),
            )
        head = messages[:1]
        if summary:
            head.append(messages[1])
        return head + kept + [messages[-1]]
