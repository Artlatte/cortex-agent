"""Embedding backends: a deterministic offline hash embedder and an HTTP one.

``HashEmbedder`` needs no dependencies and is useful for tests and small
offline corpora; ``OpenAICompatibleEmbedder`` talks to any OpenAI-compatible
``/embeddings`` endpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Protocol

import httpx

from cortex.config import EmbeddingConfig
from cortex.errors import ConfigurationError, ProviderError


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf"


class Embedder(Protocol):
    """Interface for text embedding providers."""

    dim: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic, dependency-free embedding via hashed character n-grams."""

    def __init__(self, dim: int = 256, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = []
        run: list[str] = []
        for char in lowered:
            if _is_cjk(char):
                if run:
                    self._flush_ascii_run(run, tokens)
                tokens.append(char)
            elif "a" <= char <= "z" or "0" <= char <= "9":
                run.append(char)
            elif run:
                self._flush_ascii_run(run, tokens)
        if run:
            self._flush_ascii_run(run, tokens)
        return tokens

    @staticmethod
    def _flush_ascii_run(run: list[str], tokens: list[str]) -> None:
        word = "".join(run)
        run.clear()
        if len(word) >= 3:
            tokens.extend(word[i : i + 3] for i in range(len(word) - 2))
        else:
            tokens.append(word)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in self._tokens(text):
            digest = hashlib.md5(f"{self.seed}:{token}".encode()).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            return [value / norm for value in vector]
        return vector

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAICompatibleEmbedder:
    """Embedding client for any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout_seconds = timeout_seconds

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                raise ProviderError(
                    "openai", f"embedding request failed: {exc}", retryable=True
                ) from exc
            if response.status_code == 200:
                items = response.json()["data"]
                items = sorted(items, key=lambda item: item.get("index", 0))
                return [list(item["embedding"]) for item in items]
            retryable = response.status_code == 429 or response.status_code >= 500
            error = ProviderError(
                "openai",
                f"embedding request failed with status {response.status_code}",
                status_code=response.status_code,
                retryable=retryable,
            )
            if attempt == 0 and retryable:
                await asyncio.sleep(0.5)
                continue
            raise error
        raise ProviderError("openai", "embedding request failed after retries")

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Construct an :class:`Embedder` from an :class:`EmbeddingConfig`."""
    if config.provider == "hash":
        return HashEmbedder(config.dim)
    if config.provider == "openai":
        if not config.base_url:
            raise ConfigurationError(
                "OpenAI embeddings require a base_url and an API key "
                "(OPENAI_API_KEY environment variable)"
            )
        api_key = config.resolve_api_key()
        if not api_key:
            raise ConfigurationError(
                "OpenAI embeddings require an API key; set the OPENAI_API_KEY "
                "environment variable or configure api_key_env"
            )
        return OpenAICompatibleEmbedder(config.base_url, api_key, config.model, config.dim)
    raise ConfigurationError(f"unknown embedding provider: {config.provider}")
