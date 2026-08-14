"""Central configuration.

Settings can be provided as a JSON file (path from the ``CORTEX_CONFIG`` env
var) or assembled in code. Each provider references its API key through an
environment variable name so secrets never live in the config file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """A single upstream LLM provider."""

    name: str
    kind: Literal["openai", "anthropic", "gemini", "mock", "demo"] = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    weight: int = 1

    def resolve_api_key(self) -> str | None:
        """Read the API key from the referenced env var, if any."""
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0


class EmbeddingConfig(BaseModel):
    """Embedding backend. ``hash`` is a deterministic offline fallback."""

    provider: Literal["hash", "openai"] = "hash"
    base_url: str | None = None
    api_key_env: str | None = None
    model: str = "text-embedding-3-small"
    dim: int = 256

    def resolve_api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    vector_weight: float = 0.5
    top_k: int = 6
    rrf_k: int = 60
    reranker: Literal["rule", "cross_encoder"] = "rule"


class MemoryConfig(BaseModel):
    short_term_max_turns: int = 12
    summarize_enabled: bool = True
    long_term_db_path: str = "data/long_term.db"


class AgentConfig(BaseModel):
    max_iterations: int = 8
    token_budget: int = 8000
    temperature: float = 0.2


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    rate_limit_per_minute: int = 120


class CortexConfig(BaseModel):
    providers: list[ProviderConfig]
    default_provider: str | None = None
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    log_level: str = "INFO"
    data_dir: str = "data"

    @classmethod
    def load(cls, path: str | Path | None = None) -> CortexConfig:
        """Load config from ``CORTEX_CONFIG`` (or an explicit path), else defaults."""
        if path is None:
            path = os.environ.get("CORTEX_CONFIG")
        if path:
            return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls.default()

    @classmethod
    def default(cls) -> CortexConfig:
        """Offline-friendly defaults: a single deterministic mock provider."""
        return cls(
            providers=[ProviderConfig(name="demo", kind="demo", model="demo-model")],
            default_provider="demo",
        )
