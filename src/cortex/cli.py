"""Command line interface.

Examples::

    cortex serve                          # start the HTTP service (offline demo provider)
    cortex demo agent                     # ReAct agent demo
    cortex demo rag                       # RAG ingest + hybrid search demo
    cortex demo multi                     # multi-agent orchestration demo
    cortex rag ingest examples/data --save data/index
    cortex rag search "什么是混合检索" --top-k 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import suppress
from dataclasses import asdict

from cortex import __version__
from cortex.config import CortexConfig
from cortex.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex", description="Cortex Agent — 企业级多智能体 + RAG 平台"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--config", default=None, help="配置文件路径（JSON）")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="开发模式热重载")

    demo = sub.add_parser("demo", help="运行离线演示")
    demo.add_argument("which", nargs="?", choices=["agent", "rag", "multi", "all"], default="all")

    rag = sub.add_parser("rag", help="RAG 知识库操作")
    rag_sub = rag.add_subparsers(dest="rag_command", required=True)
    ingest = rag_sub.add_parser("ingest", help="入库文件/目录")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--config", default=None)
    ingest.add_argument("--save", default=None, help="入库后把索引保存到目录")
    search = rag_sub.add_parser("search", help="混合检索")
    search.add_argument("query")
    search.add_argument("--config", default=None)
    search.add_argument("--top-k", type=int, default=None)
    search.add_argument("--save-dir", default=None, help="从已保存的索引目录加载")
    return parser


async def _rag_command(args: argparse.Namespace) -> None:
    config = CortexConfig.load(args.config)
    from cortex.rag.pipeline import RAGPipeline

    if args.save_dir:
        pipeline = await RAGPipeline.load(args.save_dir, config)
    else:
        pipeline = RAGPipeline(config)
    if args.rag_command == "ingest":
        report = await pipeline.ingest(args.paths)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        if args.save:
            pipeline.save(args.save)
            print(f"索引已保存到 {args.save}")
    else:
        result = await pipeline.search(args.query, top_k=args.top_k)
        print(f"query: {result.query}（{result.elapsed_ms} ms，{len(result.hits)} 条命中）\n")
        for i, hit in enumerate(result.hits, 1):
            source = hit.doc.metadata.get("source", "?")
            print(f"[{i}] score={hit.score:.3f} source={source}")
            print(f"    {hit.doc.page_content[:160]}\n")


def main(argv: list[str] | None = None) -> int:
    # Never crash on GBK consoles: replace unencodable characters instead.
    with suppress(AttributeError, ValueError):
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    if args.command == "serve":
        import uvicorn

        from cortex.api.app import create_app

        config = CortexConfig.load(args.config)
        host = args.host or config.api.host
        port = args.port or config.api.port
        uvicorn.run(create_app(config), host=host, port=port, reload=args.reload)
    elif args.command == "demo":
        from cortex.demo import run_agent_demo, run_multi_agent_demo, run_rag_demo

        setup_logging("WARNING")
        if args.which == "agent":
            asyncio.run(run_agent_demo())
        elif args.which == "rag":
            asyncio.run(run_rag_demo())
        elif args.which == "multi":
            asyncio.run(run_multi_agent_demo())
        else:
            from cortex.demo import run_all_demos

            asyncio.run(run_all_demos())
    elif args.command == "rag":
        asyncio.run(_rag_command(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
