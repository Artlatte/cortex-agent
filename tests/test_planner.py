"""Planner and plan-executor tests."""

from __future__ import annotations

import json

from conftest import FakeAgent, FakeAgentWithFailure

from cortex.agent.planner import Plan, PlannedExecutor, Planner
from cortex.llm.base import LLMResponse
from cortex.llm.gateway import LLMGateway
from cortex.llm.mock import MockProvider

PLAN_JSON = {
    "goal": "写一份调研报告",
    "reasoning": "先查资料，再整理，最后成文",
    "steps": [
        {"step_id": 1, "title": "查资料", "description": "检索相关资料", "depends_on": []},
        {"step_id": 2, "title": "整理要点", "description": "提炼关键信息", "depends_on": [1]},
        {"step_id": 3, "title": "成文", "description": "撰写报告", "depends_on": [2]},
    ],
}


def make_planner() -> Planner:
    def handler(messages, tools=None, response_format=None):
        assert response_format is not None
        return LLMResponse(content=json.dumps(PLAN_JSON, ensure_ascii=False))

    return Planner(LLMGateway([MockProvider(handler=handler)]))


async def test_planner_produces_validated_plan():
    plan = await make_planner().plan("写一份调研报告")
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 3
    assert plan.steps[1].depends_on == [1]


async def test_planned_executor_runs_steps_in_order():
    planner = make_planner()
    executor = PlannedExecutor(planner, FakeAgent())  # type: ignore[arg-type]
    execution = await executor.execute("写一份调研报告")
    assert execution.succeeded
    assert [r.step.step_id for r in execution.results] == [1, 2, 3]


async def test_planned_executor_replans_on_failure():
    planner = make_planner()
    agent = FakeAgentWithFailure(failures=1)
    executor = PlannedExecutor(planner, agent, max_replans=1)  # type: ignore[arg-type]
    execution = await executor.execute("写一份调研报告")
    # First step failed, a replan happened, remaining steps were executed.
    assert not execution.results[0].ok
    assert len(execution.results) >= 3
    assert execution.succeeded or execution.results[-1].ok
