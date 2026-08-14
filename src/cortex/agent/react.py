"""ReAct agent loop: reason → act → observe, with tool error recovery.

Each iteration asks the LLM for either tool calls or a final answer; tool
errors are fed back as observations so the model can self-correct, and the
loop is bounded by ``max_iterations`` to guarantee termination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from cortex.agent.context import ContextBuilder
from cortex.agent.memory import ShortTermMemory
from cortex.agent.tools import ToolRegistry
from cortex.config import AgentConfig
from cortex.errors import ToolNotFoundError
from cortex.llm.base import ChatMessage, LLMResponse, Usage
from cortex.llm.gateway import LLMGateway
from cortex.logging import log, trace_span
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.agent.react")

DEFAULT_SYSTEM_PROMPT = (
    "你是一个严谨的 AI 助手。你可以使用工具获取实时信息或执行操作。\n"
    "规则：\n"
    "1. 先分析问题，必要时调用工具，最后给出完整、有依据的答案。\n"
    "2. 一次可以并行调用多个工具；工具失败时根据错误信息调整策略。\n"
    "3. 引用工具返回的信息时，说明信息来源。\n"
    "4. 信息充分后立即给出最终答案，不要无意义地重复调用工具。"
)


@dataclass
class AgentStep:
    iteration: int
    kind: Literal["llm", "tool", "final"]
    detail: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    session_id: str | None = None


class ReActAgent:
    """A single tool-using agent bound to one gateway and one tool registry."""

    def __init__(
        self,
        gateway: LLMGateway,
        registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory: ShortTermMemory | None = None,
        config: AgentConfig | None = None,
        provider: str | None = None,
        name: str = "react",
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.memory = memory or ShortTermMemory()
        self.config = config or AgentConfig()
        self.provider = provider
        self.name = name
        self.context_builder = ContextBuilder(
            system_prompt, registry, self.memory, self.config.token_budget
        )

    def _neutral_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self.registry.list()
        ]

    async def run(self, user_input: str, session_id: str | None = None) -> AgentResult:
        messages = self.context_builder.build(user_input)
        turn_messages = [ChatMessage(role="user", content=user_input)]
        steps: list[AgentStep] = []
        total_usage = Usage()

        for iteration in range(1, self.config.max_iterations + 1):
            response = await self._call_llm(messages, iteration)
            total_usage = total_usage.merge(response.usage)
            assistant = ChatMessage(
                role="assistant", content=response.content or "", tool_calls=response.tool_calls
            )
            messages.append(assistant)
            turn_messages.append(assistant)

            if not response.tool_calls:
                answer = response.content or "（模型未返回内容）"
                steps.append(AgentStep(iteration=iteration, kind="final", detail=answer))
                self.memory.add_many(turn_messages)
                METRICS.inc("agent_runs_total", agent=self.name, status="ok")
                return AgentResult(
                    answer=answer,
                    steps=steps,
                    total_usage=total_usage,
                    iterations=iteration,
                    session_id=session_id,
                )

            for tool_call in response.tool_calls:
                observation = await self._run_tool(tool_call.name, tool_call.arguments)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=observation,
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                    )
                )
                turn_messages.append(messages[-1])
                steps.append(
                    AgentStep(
                        iteration=iteration,
                        kind="tool",
                        detail=observation[:300],
                        tool_name=tool_call.name,
                        tool_arguments=tool_call.arguments,
                    )
                )

        answer = (
            f"达到最大迭代次数（{self.config.max_iterations}），任务已停止。"
            "请缩小问题范围后重试。"
        )
        steps.append(AgentStep(iteration=self.config.max_iterations, kind="final", detail=answer))
        self.memory.add_many(turn_messages)
        METRICS.inc("agent_runs_total", agent=self.name, status="max_iterations")
        return AgentResult(
            answer=answer,
            steps=steps,
            total_usage=total_usage,
            iterations=self.config.max_iterations,
            session_id=session_id,
        )

    async def _call_llm(self, messages: list[ChatMessage], iteration: int) -> LLMResponse:
        with trace_span("agent.llm", agent=self.name, iteration=iteration):
            return await self.gateway.chat(
                messages,
                tools=self._neutral_tools(),
                provider=self.provider,
                temperature=self.config.temperature,
            )

    async def _run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        with trace_span("agent.tool", tool=name):
            try:
                tool = self.registry.get(name)
            except ToolNotFoundError:
                return (
                    f"错误：工具 '{name}' 不存在。"
                    f"可用工具：{', '.join(t.name for t in self.registry.list())}"
                )
            result = await tool.run(arguments)
            log(
                logger,
                logging.INFO if result.ok else logging.WARNING,
                "tool executed",
                tool=name,
                ok=result.ok,
            )
            return result.to_message()
