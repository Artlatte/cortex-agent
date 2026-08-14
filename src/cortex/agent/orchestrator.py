"""Multi-agent orchestration: Planner → parallel Researchers → Critic → Writer.

Four roles collaborate on one task:
- **Planner** decomposes the goal into an ordered plan (structured output).
- **Researchers** (a pool of ReAct agents with RAG tools) execute the plan
  steps in parallel, bounded by a semaphore.
- **Critic** verifies the findings against the evidence and either passes the
  answer or lists concrete issues (structured output).
- **Writer** synthesizes the final answer with source citations and must
  address the critic's feedback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from cortex.agent.planner import Plan, Planner
from cortex.agent.react import AgentResult, ReActAgent
from cortex.errors import StructuredOutputError
from cortex.llm.base import ChatMessage, Usage
from cortex.llm.gateway import LLMGateway
from cortex.llm.structured import generate_structured
from cortex.logging import log, trace_span
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.agent.orchestrator")


class Critique(BaseModel):
    verdict: Literal["pass", "revise"] = Field(description="pass=证据充分可直接成稿；revise=存在缺口")
    issues: list[str] = Field(default_factory=list, description="具体问题清单")
    suggestions: list[str] = Field(default_factory=list, description="改进建议")


CRITIC_PROMPT = """你是一位严格的事实核查员。请基于给定的研究证据，判断是否可以形成可靠答案。

核查要点：
1. 每个关键结论是否有证据支撑（标注证据来源）；
2. 证据之间是否存在矛盾；
3. 是否存在未覆盖的重要方面。
只有证据充分才给出 pass，否则给出 revise 并列出具体问题。只输出符合 JSON Schema 的内容。"""

WRITER_PROMPT = """你是一位资深技术分析师。请基于研究 Agent 收集的证据撰写最终答案。

要求：
1. 结构清晰：先给结论，再给论证，最后给建议；
2. 每条关键信息标注来源（步骤编号或文档片段）；
3. 若核查员提出 issues，必须逐条回应或补充；
4. 不要编造证据中没有的内容。"""


@dataclass
class ResearchFinding:
    step_title: str
    step_id: int
    result: AgentResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None

    def evidence(self) -> str:
        if not self.ok:
            return f"[步骤{self.step_id}] {self.step_title}: 失败（{self.error}）"
        return f"[步骤{self.step_id}] {self.step_title}:\n{self.result.answer}"


@dataclass
class OrchestrationResult:
    task: str
    plan: Plan
    findings: list[ResearchFinding] = field(default_factory=list)
    critique: Critique | None = None
    answer: str = ""
    total_usage: Usage = field(default_factory=Usage)

    def report(self) -> str:
        """Human-readable trace of the whole orchestration."""
        lines = [f"任务: {self.task}", "", f"计划: {self.plan.reasoning}"]
        for step in self.plan.steps:
            deps = f" (依赖 {step.depends_on})" if step.depends_on else ""
            lines.append(f"  {step.step_id}. {step.title}{deps}: {step.description}")
        lines.append("")
        for finding in self.findings:
            lines.append(f"◆ 研究结果 {finding.evidence()[:500]}")
        if self.critique:
            lines.append(
                f"\n◆ 核查: {self.critique.verdict}"
                + (f" — 问题: {'; '.join(self.critique.issues)}" if self.critique.issues else "")
            )
        lines.append(f"\n◆ 最终答案:\n{self.answer}")
        return "\n".join(lines)


class MultiAgentOrchestrator:
    def __init__(
        self,
        planner: Planner,
        research_agents: list[ReActAgent],
        gateway: LLMGateway,
        provider: str | None = None,
        max_parallel: int = 4,
    ) -> None:
        if not research_agents:
            raise ValueError("at least one research agent is required")
        self.planner = planner
        self.research_agents = research_agents
        self.gateway = gateway
        self.provider = provider
        self.max_parallel = max_parallel

    async def run(self, task: str, context: str = "") -> OrchestrationResult:
        with trace_span("orchestrator.run", task=task[:120]):
            plan = await self.planner.plan(task, context)
            findings = await self._research(plan)
            critique = await self._critique(task, findings)
            answer = await self._write(task, findings, critique)
            total = Usage()
            for finding in findings:
                if finding.result is not None:
                    total = total.merge(finding.result.total_usage)
            METRICS.inc("orchestrations_total", status="ok" if critique is None or critique.verdict == "pass" else "revised")
            return OrchestrationResult(
                task=task, plan=plan, findings=findings, critique=critique, answer=answer, total_usage=total
            )

    async def _research(self, plan: Plan) -> list[ResearchFinding]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def research_one(index: int, step) -> ResearchFinding:
            agent = self.research_agents[index % len(self.research_agents)]
            async with semaphore:
                try:
                    with trace_span("orchestrator.research", step=step.step_id, title=step.title):
                        result = await agent.run(
                            f"[研究步骤 {step.step_id}] {step.title}\n任务：{step.description}"
                        )
                    return ResearchFinding(step_title=step.title, step_id=step.step_id, result=result)
                except Exception as exc:  # noqa: BLE001
                    log(logger, logging.ERROR, "research step failed", step=step.step_id, error=str(exc))
                    return ResearchFinding(step_title=step.title, step_id=step.step_id, error=str(exc))

        with trace_span("orchestrator.research_parallel", steps=len(plan.steps)):
            findings = await asyncio.gather(
                *(research_one(i, step) for i, step in enumerate(plan.steps))
            )
        return list(findings)

    async def _critique(self, task: str, findings: list[ResearchFinding]) -> Critique | None:
        evidence = "\n\n".join(f.evidence() for f in findings)
        messages = [ChatMessage(role="user", content=f"任务：{task}\n\n研究证据：\n{evidence}")]
        try:
            return await generate_structured(
                self.gateway,
                messages,
                Critique,
                provider=self.provider,
                system_prompt=CRITIC_PROMPT,
                temperature=0.0,
            )
        except StructuredOutputError as exc:
            # Degrade gracefully: without a critic, hand off to the writer anyway.
            log(logger, logging.WARNING, "critic failed, skipping verification", error=str(exc))
            return None

    async def _write(
        self,
        task: str,
        findings: list[ResearchFinding],
        critique: Critique | None,
    ) -> str:
        evidence = "\n\n".join(f.evidence() for f in findings)
        critique_text = (
            f"\n\n核查意见（verdict={critique.verdict}）：\n"
            + "\n".join(f"- {i}" for i in (critique.issues + critique.suggestions))
            if critique
            else ""
        )
        messages = [
            ChatMessage(
                role="user",
                content=f"任务：{task}\n\n研究证据：\n{evidence}{critique_text}",
            )
        ]
        response = await self.gateway.chat(
            [ChatMessage(role="system", content=WRITER_PROMPT)] + messages,
            provider=self.provider,
            temperature=0.2,
        )
        return response.content or "（撰写 Agent 未返回内容）"
