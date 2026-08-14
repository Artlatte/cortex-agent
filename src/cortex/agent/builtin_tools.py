"""Built-in tools: a sandboxed calculator, a clock, and RAG/memory bridges."""

from __future__ import annotations

import ast
import datetime

from cortex.agent.memory import LongTermMemory
from cortex.agent.tools import ToolRegistry, tool
from cortex.rag.pipeline import RAGPipeline

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _validate_expression(node: ast.AST) -> None:
    """Whitelist-only AST validation so calculator can never run arbitrary code."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        _validate_expression(node.left)
        _validate_expression(node.right)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        _validate_expression(node.operand)
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@tool(description="安全计算一个数学表达式，支持 + - * / // % ** 和括号，例如 '2*(3+4)'")
def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    _validate_expression(tree.body)
    result = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    return f"{expression} = {result}"


@tool(description="获取当前日期和时间（ISO 格式）")
def current_time() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def make_rag_tools(
    pipeline: RAGPipeline, memory: LongTermMemory | None = None
) -> ToolRegistry:
    """RAG search + long-term memory tools bound to a pipeline/memory instance."""

    @tool(description="在已入库的知识库中执行混合检索（向量+BM25+重排序），返回相关文档片段及来源")
    async def rag_search(query: str, top_k: int = 4) -> str:
        result = await pipeline.search(query, top_k=top_k)
        if not result.hits:
            return "知识库中没有找到相关内容。"
        lines = []
        for i, hit in enumerate(result.hits, 1):
            source = hit.doc.metadata.get("source", "unknown")
            lines.append(f"[{i}]（来源: {source}，相关度: {hit.score:.3f}）\n{hit.doc.page_content[:400]}")
        return "\n\n".join(lines)

    registry = ToolRegistry()
    registry.register(rag_search)

    if memory is not None:

        @tool(description="把一条重要信息写入长期记忆（key 唯一，重复写入会覆盖）")
        async def remember(key: str, content: str) -> str:
            await memory.store(key, content)
            return f"已记住（key={key}）"

        @tool(description="从长期记忆中检索与查询最相关的记忆条目")
        async def recall(query: str, top_k: int = 3) -> str:
            hits = await memory.search(query, top_k=top_k)
            if not hits:
                return "长期记忆中没有相关条目。"
            return "\n".join(
                f"[{i}]（key={hit.key}，相关度: {hit.score:.3f}）{hit.content}"
                for i, hit in enumerate(hits, 1)
            )

        registry.register(remember)
        registry.register(recall)

    return registry


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(current_time)
    return registry
