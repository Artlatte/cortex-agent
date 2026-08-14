"""Multi-agent orchestrator tests: plan → parallel research → critique → write."""

from __future__ import annotations

import json

from conftest import FakeAgent

from cortex.agent.orchestrator import MultiAgentOrchestrator
from cortex.agent.planner import Planner
from cortex.llm.base import LLMResponse
from cortex.llm.gateway import LLMGateway
from cortex.llm.mock import MockProvider

PLAN_JSON = {
    "goal": "调研混合检索",
    "reasoning": "并行查资料后汇总",
    "steps": [
        {"step_id": 1, "title": "检索背景", "description": "查背景资料", "depends_on": []},
        {"step_id": 2, "title": "检索实现", "description": "查实现方案", "depends_on": []},
    ],
}


def make_gateway() -> LLMGateway:
    def handler(messages, tools=None, response_format=None):
        if response_format is not None:
            if "核查" in str(messages[0].content)[:50] or "研究证据" in str(messages[0].content):
                return LLMResponse(
                    content=json.dumps(
                        {"verdict": "pass", "issues": [], "suggestions": []}, ensure_ascii=False
                    )
                )
            return LLMResponse(content=json.dumps(PLAN_JSON, ensure_ascii=False))
        return LLMResponse(content="最终答案：混合检索能兼顾关键词与语义。")

    return LLMGateway([MockProvider(handler=handler)])


async def test_orchestrator_full_pipeline():
    gateway = make_gateway()
    planner = Planner(gateway)
    researchers = [FakeAgent(prefix="research"), FakeAgent(prefix="research")]
    orchestrator = MultiAgentOrchestrator(planner, researchers, gateway, max_parallel=2)  # type: ignore[arg-type]
    result = await orchestrator.run("调研混合检索")

    assert len(result.plan.steps) == 2
    assert len(result.findings) == 2
    assert all(f.ok for f in result.findings)
    assert result.critique is not None and result.critique.verdict == "pass"
    assert "最终答案" in result.answer
    assert all(agent.calls for agent in researchers)  # both researchers worked
    assert "调研" in result.report()
