"""FastAPI application factory: wires gateway + agent + RAG + runtime."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from cortex import __version__
from cortex.agent.builtin_tools import default_registry, make_rag_tools
from cortex.agent.memory import LongTermMemory, ShortTermMemory
from cortex.agent.react import ReActAgent
from cortex.agent.runtime import AgentRuntime
from cortex.api.middleware import RateLimitMiddleware, RequestContextMiddleware
from cortex.api.routes import router
from cortex.config import CortexConfig
from cortex.llm.gateway import LLMGateway
from cortex.llm.providers import build_provider
from cortex.logging import log, setup_logging
from cortex.rag.pipeline import RAGPipeline

logger = logging.getLogger("cortex.api.app")


def build_stack(config: CortexConfig):
    """Assemble the production stack from a config object."""
    gateway = LLMGateway(
        [build_provider(p) for p in config.providers],
        default_provider=config.default_provider,
        circuit_breaker=config.circuit_breaker,
    )
    pipeline = RAGPipeline(config)
    long_memory = LongTermMemory(config.memory.long_term_db_path)
    registry = default_registry()
    for tool_obj in make_rag_tools(pipeline, long_memory).list():
        registry.register(tool_obj)
    agent = ReActAgent(
        gateway,
        registry,
        memory=ShortTermMemory(
            max_turns=config.memory.short_term_max_turns,
            summarize_enabled=config.memory.summarize_enabled,
        ),
        config=config.agent,
    )
    sessions_db = str(Path(config.data_dir) / "sessions.db")
    runtime = AgentRuntime(agent, db_path=sessions_db)
    return gateway, pipeline, agent, runtime


def create_app(config: CortexConfig | None = None) -> FastAPI:
    config = config or CortexConfig.load()
    setup_logging(config.log_level)
    gateway, pipeline, agent, runtime = build_stack(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        restored = await runtime.restore()
        log(logger, logging.INFO, "cortex started", version=__version__, restored_sessions=restored)
        yield
        await runtime.close()
        await gateway.aclose()

    app = FastAPI(title="Cortex Agent", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.gateway = gateway
    app.state.pipeline = pipeline
    app.state.agent = agent
    app.state.runtime = runtime
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RateLimitMiddleware, rate_per_minute=config.api.rate_limit_per_minute)
    app.include_router(router)
    return app
