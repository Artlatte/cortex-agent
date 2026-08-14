"""HTTP routes: health, chat, RAG ingest/search, agent sessions, metrics."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from cortex import __version__
from cortex.metrics import METRICS

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    session_id: str | None = None
    stream: bool = False


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1, description="要入库的文件或目录路径")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RunAgentRequest(BaseModel):
    task: str = Field(min_length=1, description="交给 Agent 的任务")
    session_id: str | None = None
    idempotency_key: str | None = Field(default=None, description="幂等键，重复提交返回同一会话")


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    state = request.app.state
    return {
        "status": "ok",
        "version": __version__,
        "providers": [p.name for p in state.gateway.providers],
        "rag_documents": state.pipeline.stats() if state.pipeline else None,
        "sessions": len(state.runtime.list_sessions(limit=10000)),
    }


async def _sse_chunks(answer: str):
    """Emit an answer as SSE delta chunks (demonstration streaming)."""
    for i in range(0, len(answer), 12):
        yield f"data: {json.dumps({'delta': answer[i:i + 12]}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"


@router.post("/v1/chat", response_model=None)
async def chat(body: ChatRequest, request: Request) -> StreamingResponse | dict:
    agent = request.app.state.agent
    if body.stream:

        async def event_stream():
            result = await agent.run(body.question, session_id=body.session_id)
            async for chunk in _sse_chunks(result.answer):
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    result = await agent.run(body.question, session_id=body.session_id)
    return {
        "answer": result.answer,
        "iterations": result.iterations,
        "usage": result.total_usage.__dict__,
        "steps": [
            {"iteration": s.iteration, "kind": s.kind, "tool": s.tool_name} for s in result.steps
        ],
    }


@router.post("/v1/rag/ingest")
async def rag_ingest(body: IngestRequest, request: Request) -> dict:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not configured")
    report = await pipeline.ingest(body.paths)
    return {
        "files": report.files,
        "documents": report.documents,
        "chunks": report.chunks,
        "errors": report.errors,
        "elapsed_ms": report.elapsed_ms,
    }


@router.post("/v1/rag/search")
async def rag_search(body: SearchRequest, request: Request) -> dict:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not configured")
    result = await pipeline.search(body.query, top_k=body.top_k)
    return {
        "query": result.query,
        "elapsed_ms": result.elapsed_ms,
        "hits": [
            {
                "score": hit.score,
                "sources": hit.sources,
                "content": hit.doc.page_content,
                "metadata": hit.doc.metadata,
            }
            for hit in result.hits
        ],
    }


@router.post("/v1/agents/run", status_code=202)
async def run_agent(body: RunAgentRequest, request: Request) -> dict:
    runtime = request.app.state.runtime
    session = await runtime.submit(
        body.task,
        session_id=body.session_id,
        idempotency_key=body.idempotency_key,
    )
    return session.to_dict()


@router.get("/v1/agents")
async def list_agents(request: Request) -> dict:
    return {"sessions": request.app.state.runtime.list_sessions()}


@router.get("/v1/agents/{session_id}")
async def get_agent(session_id: str, request: Request) -> dict:
    session = await request.app.state.runtime.get_status(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    return session


@router.delete("/v1/agents/{session_id}")
async def cancel_agent(session_id: str, request: Request) -> dict:
    cancelled = await request.app.state.runtime.cancel(session_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail=f"session '{session_id}' cannot be cancelled")
    return {"session_id": session_id, "cancelled": True}


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(METRICS.render_prometheus(), media_type="text/plain; version=0.0.4")
