"""Tests for the cosine-similarity vector store and its persistence."""

import pytest

from cortex.rag import Document, VectorStore


def test_cosine_ordering() -> None:
    store = VectorStore(dim=3)
    store.add(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["a", "b"],
        [Document("x", {"doc_id": "a"}), Document("y", {"doc_id": "b"})],
    )
    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] > hits[1][1]


def test_get_missing_doc_raises_keyerror() -> None:
    store = VectorStore(dim=2)
    with pytest.raises(KeyError):
        store.get("missing")


def test_save_load_round_trip_preserves_documents(tmp_path) -> None:
    store = VectorStore(dim=2)
    documents = [
        Document("hello world", {"source": "a.txt", "n": 1}),
        Document("goodbye world", {"source": "b.txt", "n": 2}),
    ]
    store.add([[1.0, 0.0], [0.0, 1.0]], ["a", "b"], documents)
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.get("a").page_content == "hello world"
    assert loaded.get("a").metadata == {"source": "a.txt", "n": 1}
    assert loaded.get("b").page_content == "goodbye world"
    assert loaded.get("b").metadata == {"source": "b.txt", "n": 2}
