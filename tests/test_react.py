"""ReAct agent loop tests: tool calls, error recovery, iteration bound."""

from __future__ import annotations

from cortex.agent.builtin_tools import calculator, current_time
from cortex.agent.react import ReActAgent
from cortex.agent.tools import ToolRegistry
from cortex.config import AgentConfig
from cortex.llm.base import LLMResponse, ToolCall, Usage
from cortex.llm.gateway import LLMGateway
from cortex.llm.mock import MockProvider


def make_gateway(*responses: LLMResponse) -> LLMGateway:
    return LLMGateway([MockProvider(responses=list(responses))])


async def test_tool_call_then_final_answer():
    gateway = make_gateway(
        LLMResponse(
            tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "2+2"})]
        ),
        LLMResponse(content="2+2 的结果是 4", usage=Usage(total_tokens=10)),
    )
    registry = ToolRegistry()
    registry.register(calculator)
    agent = ReActAgent(gateway, registry, name="test-agent")
    result = await agent.run("请计算 2+2")

    assert result.answer == "2+2 的结果是 4"
    assert result.iterations == 2
    tool_steps = [s for s in result.steps if s.kind == "tool"]
    assert len(tool_steps) == 1
    assert tool_steps[0].tool_name == "calculator"
    assert "2+2 = 4" in tool_steps[0].detail
    # The exchange was persisted into short-term memory.
    assert len(agent.memory.get_messages()) == 4  # user, assistant, tool, assistant


async def test_tool_error_is_fed_back_and_recovered():
    gateway = make_gateway(
        LLMResponse(tool_calls=[ToolCall(id="1", name="no_such_tool", arguments={})]),
        LLMResponse(content="该工具不可用，我改用其他方式回答"),
    )
    agent = ReActAgent(gateway, ToolRegistry(), name="test-agent")
    result = await agent.run("做点事情")
    assert result.answer == "该工具不可用，我改用其他方式回答"
    tool_step = next(s for s in result.steps if s.kind == "tool")
    assert "不存在" in tool_step.detail


async def test_max_iterations_bounds_loop():
    response = LLMResponse(
        tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "1+1"})]
    )
    gateway = make_gateway(*([response] * 10))
    registry = ToolRegistry()
    registry.register(calculator)
    config = AgentConfig(max_iterations=3)
    agent = ReActAgent(gateway, registry, config=config, name="test-agent")
    result = await agent.run("无限调用工具")
    assert result.iterations == 3
    assert "达到最大迭代次数" in result.answer
    assert len(gateway.providers[0].calls) == 3  # type: ignore[union-attr]


async def test_agent_without_tools_answers_directly():
    gateway = make_gateway(LLMResponse(content="直接回答"))
    agent = ReActAgent(gateway, ToolRegistry(), name="test-agent")
    result = await agent.run("你好")
    assert result.answer == "直接回答"
    assert result.iterations == 1


async def test_clock_tool_returns_iso_time():
    registry = ToolRegistry()
    registry.register(current_time)
    tool = registry.get("current_time")
    output = (await tool.run({})).output
    assert "T" in output  # ISO 8601
