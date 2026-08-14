"""Tests for BM25 tokenization and indexing."""

from cortex.rag import BM25Index, tokenize


def test_tokenize_cjk_characters() -> None:
    assert tokenize("机器学习") == ["机", "器", "学", "习"]


def test_tokenize_ascii_runs() -> None:
    assert tokenize("Hello, World 42!") == ["hello", "world", "42"]


def test_doc_containing_query_term_outranks_other() -> None:
    index = BM25Index()
    index.add("a", "the quick brown fox jumps over the lazy dog")
    index.add("b", "a completely unrelated sentence about nothing")
    results = index.search("quick fox", top_k=10)
    assert results[0][0] == "a"
    assert results[0][1] > results[1][1]


def test_add_remove_and_replace_semantics() -> None:
    index = BM25Index()
    index.add("a", "machine learning")
    index.add("b", "deep learning")
    assert len(index) == 2

    index.remove("a")
    assert len(index) == 1
    assert all(doc_id != "a" for doc_id, _ in index.search("learning"))

    # Re-adding an existing id replaces its content in place.
    index.add("b", "reinforcement learning")
    results = index.search("reinforcement")
    assert results[0][0] == "b"
