"""Hybrid retrieval: RRF fusion of dense vectors and BM25 with optional reranking."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from cortex.errors import ConfigurationError
from cortex.logging import log
from cortex.metrics import METRICS
from cortex.rag.bm25 import BM25Index, tokenize
from cortex.rag.document import Document
from cortex.rag.embeddings import Embedder
from cortex.rag.vector_store import VectorStore

logger = logging.getLogger("cortex.rag")


@dataclass
class ScoredDoc:
    """A retrieved document with its score and the per-source breakdown."""

    doc: Document
    score: float
    sources: dict[str, float]


@dataclass
class RetrievalResult:
    """The outcome of a hybrid search."""

    query: str
    hits: list[ScoredDoc]
    elapsed_ms: int


class Reranker(Protocol):
    """Synchronous reranker; invoked in a worker thread by the retriever."""

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]: ...


class RuleBasedReranker:
    """Cheap lexical reranker that boosts documents sharing query terms."""

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return docs
        for scored in docs:
            doc_tokens = set(tokenize(scored.doc.page_content))
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            scored.score += 0.1 * overlap
        return docs


class CrossEncoderReranker:
    """Sentence-transformers cross-encoder reranker with graceful degradation."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._unavailable = False

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        if not docs or self._unavailable:
            return docs
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ImportError:
                self._unavailable = True
                log(
                    logger,
                    logging.WARNING,
                    "sentence_transformers is not installed; cross-encoder reranking disabled",
                )
                return docs
        pairs = [(query, scored.doc.page_content) for scored in docs]
        scores = self._model.predict(pairs)
        for scored, score in zip(docs, scores, strict=True):
            scored.score = float(score)
        return docs


class HybridRetriever:
    """Combines dense vector search and BM25 via reciprocal rank fusion."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        vector_weight: float = 0.5,
        top_k: int = 8,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.vector_weight = vector_weight
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.reranker = reranker

    async def index(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents (and their embeddings) to both underlying indexes."""
        ids: list[str] = []
        for n, document in enumerate(documents):
            doc_id = document.metadata.get("doc_id") or f"doc-{n}"
            ids.append(str(doc_id))
        self.vector_store.add(embeddings, ids, documents)
        for doc_id, document in zip(ids, documents, strict=True):
            self.bm25.add(doc_id, document.page_content)

    async def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Run hybrid search and return the fused, reranked ranking."""
        start = time.perf_counter()
        final_k = top_k if top_k is not None else self.top_k
        candidate_k = max(final_k * 4, 20)

        query_vector = await self.embedder.embed_query(query)
        vector_hits = self.vector_store.search(query_vector, top_k=candidate_k)
        bm25_hits = self.bm25.search(query, top_k=candidate_k)

        vector_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(vector_hits, start=1)}
        bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_hits, start=1)}
        vector_score = dict(vector_hits)
        bm25_score = dict(bm25_hits)

        doc_ids: dict[str, None] = {}
        for doc_id in vector_rank:
            doc_ids[doc_id] = None
        for doc_id in bm25_rank:
            doc_ids[doc_id] = None

        scored: list[ScoredDoc] = []
        for doc_id in doc_ids:
            rank_v = vector_rank.get(doc_id, candidate_k + 1)
            rank_b = bm25_rank.get(doc_id, candidate_k + 1)
            fused = self.vector_weight / (self.rrf_k + rank_v) + (
                1.0 - self.vector_weight
            ) / (self.rrf_k + rank_b)
            scored.append(
                ScoredDoc(
                    doc=self.vector_store.get(doc_id),
                    score=fused,
                    sources={
                        "vector": vector_score.get(doc_id, 0.0),
                        "bm25": bm25_score.get(doc_id, 0.0),
                        "fused": fused,
                    },
                )
            )

        if self.reranker is not None and scored:
            scored = await asyncio.to_thread(self.reranker.rerank, query, scored)

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = scored[:final_k]
        for item in scored:
            item.sources["fused"] = item.score

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        METRICS.inc("rag_queries_total")
        return RetrievalResult(query=query, hits=scored, elapsed_ms=elapsed_ms)


def build_reranker(name: str) -> Reranker:
    """Construct a reranker by name (``rule`` or ``cross_encoder``)."""
    if name == "rule":
        return RuleBasedReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker()
    raise ConfigurationError(f"unknown reranker: {name}")
