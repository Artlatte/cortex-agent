"""End-to-end RAG pipeline: ingestion, hybrid search and persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from cortex.config import CortexConfig
from cortex.metrics import METRICS
from cortex.rag.bm25 import BM25Index
from cortex.rag.chunking import RecursiveCharacterTextSplitter
from cortex.rag.document import Document
from cortex.rag.embeddings import Embedder, build_embedder
from cortex.rag.loaders import SUPPORTED_SUFFIXES, get_loader
from cortex.rag.retriever import HybridRetriever, RetrievalResult, build_reranker
from cortex.rag.vector_store import VectorStore

_EMBED_BATCH_SIZE = 32


@dataclass
class IngestReport:
    """Summary of a single ingestion run."""

    files: int
    documents: int
    chunks: int
    errors: list[str]
    elapsed_ms: int


class RAGPipeline:
    """Owns the embedding model, splitter, indexes and hybrid retriever."""

    def __init__(self, config: CortexConfig, embedder: Embedder | None = None) -> None:
        self.config = config
        self.embedder = embedder if embedder is not None else build_embedder(config.embedding)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        self.vector_store = VectorStore(self.embedder.dim)
        self.bm25 = BM25Index()
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25=self.bm25,
            vector_weight=config.rag.vector_weight,
            top_k=config.rag.top_k,
            rrf_k=config.rag.rrf_k,
            reranker=build_reranker(config.rag.reranker),
        )
        self._doc_count = 0

    @staticmethod
    def _collect_files(paths: list[str | Path]) -> list[Path]:
        files: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                for candidate in sorted(path.rglob("*")):
                    if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                        files.append(candidate)
            else:
                files.append(path)
        return files

    @staticmethod
    def _make_doc_id(path: Path, chunk: Document) -> str:
        index = chunk.metadata.get("chunk_index", 0)
        return f"{path}:{index}"

    async def ingest(self, paths: list[str | Path]) -> IngestReport:
        """Load, chunk, embed and index the given files and/or directories."""
        start = time.perf_counter()
        files = self._collect_files(paths)
        errors: list[str] = []
        document_count = 0
        chunk_count = 0

        for path in files:
            try:
                loader = get_loader(path)
                documents = loader.load(path)
                chunks = self.splitter.split_documents(documents)
                for chunk in chunks:
                    chunk.metadata["doc_id"] = self._make_doc_id(path, chunk)
                embeddings: list[list[float]] = []
                for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
                    batch = chunks[i : i + _EMBED_BATCH_SIZE]
                    embeddings.extend(
                        await self.embedder.embed_texts([c.page_content for c in batch])
                    )
                await self.retriever.index(chunks, embeddings)
                document_count += len(documents)
                chunk_count += len(chunks)
                METRICS.inc("rag_ingest_files_total", error="false")
            except Exception as exc:  # noqa: BLE001 - isolate per-file failures
                errors.append(f"{path}: {exc!r}")
                METRICS.inc("rag_ingest_files_total", error="true")

        self._doc_count += document_count
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return IngestReport(
            files=len(files),
            documents=document_count,
            chunks=chunk_count,
            errors=errors,
            elapsed_ms=elapsed_ms,
        )

    async def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Delegate a hybrid search to the underlying retriever."""
        return await self.retriever.search(query, top_k=top_k)

    def save(self, directory: str | Path) -> None:
        """Persist the vector store and configuration to ``directory``."""
        directory = Path(directory)
        self.vector_store.save(directory)
        payload = {
            "embedding": self.config.embedding.model_dump(),
            "rag": self.config.rag.model_dump(),
            "documents": self._doc_count,
        }
        (directory / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    async def load(cls, directory: str | Path, config: CortexConfig) -> RAGPipeline:
        """Restore a saved pipeline, rebuilding the BM25 index from the store."""
        directory = Path(directory)
        pipeline = cls(config)
        pipeline.vector_store = VectorStore.load(directory)
        pipeline.bm25 = BM25Index()
        for doc_id, document in pipeline.vector_store.items():
            pipeline.bm25.add(doc_id, document.page_content)
        pipeline.retriever = HybridRetriever(
            embedder=pipeline.embedder,
            vector_store=pipeline.vector_store,
            bm25=pipeline.bm25,
            vector_weight=config.rag.vector_weight,
            top_k=config.rag.top_k,
            rrf_k=config.rag.rrf_k,
            reranker=build_reranker(config.rag.reranker),
        )
        try:
            meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            pipeline._doc_count = int(meta.get("documents", 0))
        except (OSError, ValueError, KeyError):
            pipeline._doc_count = len(pipeline.vector_store)
        return pipeline

    def stats(self) -> dict:
        """Return a summary of the current index state."""
        return {
            "documents": self._doc_count,
            "chunks": len(self.vector_store),
            "dim": self.embedder.dim,
        }
