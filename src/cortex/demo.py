"""Offline demos: run end-to-end without any API key (DemoProvider)."""

from __future__ import annotations

import logging
from pathlib import Path

from cortex.agent.builtin_tools import default_registry, make_rag_tools
from cortex.agent.memory import LongTermMemory, ShortTermMemory
from cortex.agent.orchestrator import MultiAgentOrchestrator
from cortex.agent.planner import Planner
from cortex.agent.react import ReActAgent
from cortex.config import CortexConfig
from cortex.llm.gateway import LLMGateway
from cortex.llm.mock import DemoProvider
from cortex.logging import setup_logging
from cortex.rag.pipeline import RAGPipeline

logger = logging.getLogger("cortex.demo")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = REPO_ROOT / "examples" / "data"


def _fix_console() -> None:
    """Keep demos crash-free on GBK consoles (replace unencodable chars)."""
    import sys
    from contextlib import suppress

    with suppress(AttributeError, ValueError):
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(errors="replace")


def _demo_gateway() -> LLMGateway:
    return LLMGateway([DemoProvider()])


def _print_banner(title: str) -> None:
    line = "=" * 66
    print(f"\n{line}\n  {title}\n{line}")


async def run_agent_demo() -> None:
    """ReAct loop with calculator + clock tools (offline)."""
    _fix_console()
    _print_banner("Demo 1 · ReAct Agent（工具调用循环，离线）")
    agent = ReActAgent(_demo_gateway(), default_registry(), name="demo-agent")
    task = "请帮我计算 (12 + 8) * 3 的结果，并告诉我今天的日期。"
    print(f"提问: {task}\n")
    result = await agent.run(task)
    for step in result.steps:
        if step.kind == "tool":
            print(f"  [工具] 第 {step.iteration} 轮 调用 {step.tool_name}({step.tool_arguments})")
            print(f"     ↳ 结果: {step.detail[:120]}")
        else:
            print(f"  [思考] 第 {step.iteration} 轮 {step.kind}: {step.detail[:120]}")
    print(f"\n最终答案:\n{result.answer}")


async def run_rag_demo() -> None:
    """Hybrid retrieval over examples/data + RAG-grounded QA (offline)."""
    _fix_console()
    _print_banner("Demo 2 · RAG 知识库（文档解析 → 切分 → 混合检索 → 重排序）")
    config = CortexConfig.default()
    pipeline = RAGPipeline(config)
    report = await pipeline.ingest([str(SAMPLE_DATA)])
    print(
        f"入库完成: {report.files} 个文件, {report.documents} 个文档, "
        f"{report.chunks} 个切片, 耗时 {report.elapsed_ms} ms, 错误 {len(report.errors)}"
    )
    query = "什么是混合检索（Hybrid Search）？它和纯向量检索相比有什么优势？"
    print(f"\n检索问题: {query}\n")
    result = await pipeline.search(query, top_k=3)
    for i, hit in enumerate(result.hits, 1):
        source = hit.doc.metadata.get("source", "?")
        print(f"  [{i}] 相关度 {hit.score:.3f} | 来源 {source}")
        print(f"      {hit.doc.page_content[:110]}...\n")

    print("用 Agent（RAG 工具 + ReAct）回答同一问题：\n")
    memory = LongTermMemory(str(REPO_ROOT / "data" / "demo_memory.db"))
    registry = make_rag_tools(pipeline, memory)
    agent = ReActAgent(_demo_gateway(), registry, name="rag-demo")
    answer = await agent.run(query)
    print(f"{answer.answer}")


async def run_multi_agent_demo() -> None:
    """Planner → parallel researchers → critic → writer (offline)."""
    _fix_console()
    _print_banner("Demo 3 · 多 Agent 编排（规划 → 并行研究 → 核查 → 撰写）")
    config = CortexConfig.default()
    pipeline = RAGPipeline(config)
    await pipeline.ingest([str(SAMPLE_DATA)])
    registry = make_rag_tools(pipeline)
    gateway = _demo_gateway()
    planner = Planner(gateway)
    researchers = [
        ReActAgent(gateway, registry, memory=ShortTermMemory(), name=f"researcher-{i}")
        for i in range(2)
    ]
    orchestrator = MultiAgentOrchestrator(planner, researchers, gateway, max_parallel=2)
    task = "调研：RAG 系统为什么要引入混合检索和重排序？请输出要点说明。"
    result = await orchestrator.run(task)
    print(result.report())


async def run_all_demos() -> None:
    setup_logging("WARNING")
    await run_agent_demo()
    await run_rag_demo()
    await run_multi_agent_demo()
