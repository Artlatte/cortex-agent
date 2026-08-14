"""Task planning: decompose a goal into an ordered, dependency-aware plan via
structured generation, then execute it step by step with replanning on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from cortex.agent.react import AgentResult, ReActAgent
from cortex.llm.base import ChatMessage
from cortex.llm.gateway import LLMGateway
from cortex.llm.structured import generate_structured
from cortex.logging import log
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.agent.planner")


class PlanStep(BaseModel):
    step_id: int = Field(description="步骤编号，从 1 开始")
    title: str = Field(description="步骤标题，简短")
    description: str = Field(description="该步骤要完成什么，包含完成标准")
    depends_on: list[int] = Field(default_factory=list, description="依赖的前置步骤编号")


class Plan(BaseModel):
    goal: str = Field(description="要达成的目标")
    reasoning: str = Field(description="任务分解思路（2-3 句）")
    steps: list[PlanStep] = Field(description="有序执行步骤，3-6 步")


PLANNER_PROMPT = """你是一位资深的 AI 任务规划专家。你的职责是把复杂目标拆解为可执行、可验证的有序步骤。

要求：
1. 步骤数量 3-6 步，按执行顺序排列；
2. 每一步都有明确的完成标准（描述中说明"完成标志"）；
3. 步骤之间如有依赖关系，用 depends_on 标注前置步骤编号；
4. 步骤描述要具体，能直接交给执行 Agent；
5. 只输出符合 JSON Schema 的内容。"""


class Planner:
    """Goal → Plan via schema-constrained generation."""

    def __init__(self, gateway: LLMGateway, provider: str | None = None) -> None:
        self.gateway = gateway
        self.provider = provider

    async def plan(self, goal: str, context: str = "") -> Plan:
        messages = [
            ChatMessage(
                role="user",
                content=f"目标：{goal}\n背景信息：{context or '（无）'}",
            )
        ]
        log(logger, logging.INFO, "planning task", goal=goal[:120])
        plan = await generate_structured(
            self.gateway,
            messages,
            Plan,
            provider=self.provider,
            system_prompt=PLANNER_PROMPT,
            temperature=0.0,
        )
        METRICS.inc("planner_runs_total")
        return plan


@dataclass
class PlanStepResult:
    step: PlanStep
    result: AgentResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None


@dataclass
class PlanExecution:
    plan: Plan
    results: list[PlanStepResult] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return all(r.ok for r in self.results)


class PlannedExecutor:
    """Executes a plan sequentially. A failed step triggers a replan of the
    remaining steps (bounded by ``max_replans``), which is how the system
    adapts when reality diverges from the initial plan."""

    def __init__(self, planner: Planner, agent: ReActAgent, max_replans: int = 2) -> None:
        self.planner = planner
        self.agent = agent
        self.max_replans = max_replans

    async def execute(self, goal: str, context: str = "") -> PlanExecution:
        plan = await self.planner.plan(goal, context)
        execution = PlanExecution(plan=plan)
        remaining: list[PlanStep] = list(plan.steps)
        replans = 0
        while remaining:
            step = remaining.pop(0)
            task_prompt = f"[步骤 {step.step_id}/{len(plan.steps)}] {step.title}\n要求：{step.description}"
            try:
                result = await self.agent.run(task_prompt)
                failed = "达到最大迭代次数" in result.answer
                execution.results.append(
                    PlanStepResult(step=step, result=result, error=result.answer if failed else None)
                )
            except Exception as exc:  # noqa: BLE001
                execution.results.append(PlanStepResult(step=step, error=str(exc)))
            if not execution.results[-1].ok and remaining:
                if replans >= self.max_replans:
                    log(logger, logging.ERROR, "replan limit reached, stopping", step=step.step_id)
                    break
                replans += 1
                done = "\n".join(
                    f"- 步骤{r.step.step_id} {r.step.title}: {'成功' if r.ok else '失败: ' + (r.error or '')}"
                    for r in execution.results
                )
                log(logger, logging.WARNING, "replanning remaining steps", replans=replans)
                new_plan = await self.planner.plan(
                    goal,
                    context=f"{context}\n已完成/失败情况：\n{done}",
                )
                remaining = list(new_plan.steps)
        METRICS.inc("plan_executions_total", status="ok" if execution.succeeded else "partial")
        return execution
