"""RAG module: ingestion, chunking, embeddings, retrieval and pipelines."""

from cortex.rag.bm25 import BM25Index, tokenize
from cortex.rag.chunking import RecursiveCharacterTextSplitter
from cortex.rag.document import Document
from cortex.rag.embeddings import (
    Embedder,
    HashEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from cortex.rag.loaders import (
    DocumentLoader,
    DocxLoader,
    PdfLoader,
    TextLoader,
    get_loader,
)
from cortex.rag.pipeline import IngestReport, RAGPipeline
from cortex.rag.retriever import (
    CrossEncoderReranker,
    HybridRetriever,
    Reranker,
    RetrievalResult,
    RuleBasedReranker,
    ScoredDoc,
    build_reranker,
)
from cortex.rag.vector_store import VectorStore

__all__ = [
    "Document",
    "DocumentLoader",
    "TextLoader",
    "PdfLoader",
    "DocxLoader",
    "get_loader",
    "RecursiveCharacterTextSplitter",
    "Embedder",
    "HashEmbedder",
    "OpenAICompatibleEmbedder",
    "build_embedder",
    "BM25Index",
    "tokenize",
    "VectorStore",
    "ScoredDoc",
    "RetrievalResult",
    "HybridRetriever",
    "Reranker",
    "RuleBasedReranker",
    "CrossEncoderReranker",
    "build_reranker",
    "RAGPipeline",
    "IngestReport",
]
