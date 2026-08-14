"""Tests for hybrid (vector + BM25) retrieval with RRF fusion and reranking."""

import pytest

from cortex.rag import (
    BM25Index,
    Document,
    HashEmbedder,
    HybridRetriever,
    RuleBasedReranker,
    ScoredDoc,
    VectorStore,
)


def _make_docs() -> list[Document]:
    return [
        Document("机器学习是人工智能的一个分支", {"doc_id": "ml"}),
        Document("今天天气很好适合出门散步", {"doc_id": "weather"}),
        Document("深度学习使用神经网络进行训练", {"doc_id": "dl"}),
    ]


async def _build(weight: float, reranker=None) -> HybridRetriever:
    embedder = HashEmbedder(dim=64)
    store = VectorStore(dim=64)
    bm25 = BM25Index()
    retriever = HybridRetriever(
        embedder,
        store,
        bm25,
        vector_weight=weight,
        top_k=10,
        reranker=reranker,
    )
    docs = _make_docs()
    embeddings = await embedder.embed_texts([d.page_content for d in docs])
    await retriever.index(docs, embeddings)
    return retriever


async def test_vector_weight_zero_pure_bm25() -> None:
    retriever = await _build(0.0)
    result = await retriever.search("机器学习")
    assert result.hits[0].doc.metadata["doc_id"] == "ml"


async def test_vector_weight_one_pure_vector() -> None:
    retriever = await _build(1.0)
    result = await retriever.search("机器学习")
    assert result.hits[0].doc.metadata["doc_id"] == "ml"


async def test_default_fusion_and_sources_keys() -> None:
    retriever = await _build(0.5)
    result = await retriever.search("什么是机器学习")
    assert result.hits[0].doc.metadata["doc_id"] == "ml"
    sources = result.hits[0].sources
    assert {"vector", "bm25", "fused"} <= set(sources)


async def test_rule_reranker_boosts_term_sharing_doc() -> None:
    retriever = await _build(0.5, reranker=RuleBasedReranker())
    result = await retriever.search("机器学习")
    assert result.hits[0].doc.metadata["doc_id"] == "ml"


def test_rule_reranker_directly_boosts_score() -> None:
    reranker = RuleBasedReranker()
    docs = [
        ScoredDoc(doc=Document("机器学习", {"doc_id": "a"}), score=0.1, sources={}),
        ScoredDoc(doc=Document("今天天气", {"doc_id": "b"}), score=0.15, sources={}),
    ]
    reranked = reranker.rerank("机器学习", docs)
    assert reranked[0].score == pytest.approx(0.2)  # 0.1 + 0.1 * 1.0
    assert reranked[1].score == pytest.approx(0.15)  # unchanged (no shared terms)
