"""API integration tests (offline DemoProvider, no network)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from cortex.api.app import create_app
from cortex.config import CortexConfig


def make_client(tmp_path, rate_limit: int = 1000) -> TestClient:
    config = CortexConfig.default()
    config.data_dir = str(tmp_path / "data")
    config.memory.long_term_db_path = str(tmp_path / "mem.db")
    config.api.rate_limit_per_minute = rate_limit
    return TestClient(create_app(config))


def test_healthz(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "demo" in body["providers"]
    assert response.headers.get("X-Request-ID")


def test_agent_run_and_poll(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/agents/run", json={"task": "你好"})
        assert response.status_code == 202
        session_id = response.json()["session_id"]
        status = {}
        for _ in range(300):
            status = client.get(f"/v1/agents/{session_id}").json()
            if status["status"] == "done":
                break
            time.sleep(0.01)
        assert status["status"] == "done"
        assert status["answer"]


def test_agent_idempotency_key(tmp_path):
    with make_client(tmp_path) as client:
        first = client.post(
            "/v1/agents/run", json={"task": "你好", "idempotency_key": "k1"}
        ).json()
        second = client.post(
            "/v1/agents/run", json={"task": "你好", "idempotency_key": "k1"}
        ).json()
        assert first["session_id"] == second["session_id"]


def test_rag_ingest_and_search(tmp_path):
    doc = tmp_path / "kb.md"
    doc.write_text(
        "# 产品说明\n\nCortex Agent 支持混合检索与重排序，融合向量相似度与 BM25 关键词匹配。\n",
        encoding="utf-8",
    )
    with make_client(tmp_path) as client:
        ingest = client.post("/v1/rag/ingest", json={"paths": [str(doc)]})
        assert ingest.status_code == 200
        assert ingest.json()["chunks"] >= 1
        search = client.post("/v1/rag/search", json={"query": "混合检索", "top_k": 3})
        assert search.status_code == 200
        hits = search.json()["hits"]
        assert hits and "混合检索" in hits[0]["content"]


def test_chat_endpoint_offline(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/chat", json={"question": "请帮我计算 2+2"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["iterations"] >= 1


def test_metrics_endpoint(tmp_path):
    with make_client(tmp_path) as client:
        client.get("/healthz")
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_rate_limit_returns_429(tmp_path):
    with make_client(tmp_path, rate_limit=2) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").status_code == 429
