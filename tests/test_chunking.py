"""Tests for recursive character text splitting."""

from cortex.rag import Document, RecursiveCharacterTextSplitter


def test_empty_text_returns_empty_list() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    assert splitter.split_text("") == []


def test_overlap_preserved_between_adjacent_chunks() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=3)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 10 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = 0
        for size in range(1, min(len(previous), len(current)) + 1):
            if previous[-size:] == current[:size]:
                shared = size
        assert shared == 3


def test_metadata_chunk_index_and_count_are_set() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=6, chunk_overlap=0)
    document = Document(
        "机器学习。深度学习。自然语言处理。",
        {"source": "a.txt", "filename": "a.txt", "filetype": "txt"},
    )
    chunks = splitter.split_documents([document])
    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_count"] == len(chunks)
        assert chunk.metadata["source"] == "a.txt"
        assert chunk.metadata["filename"] == "a.txt"
        assert chunk.metadata["filetype"] == "txt"


def test_cjk_text_splits_at_sentence_separators() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0)
    text = "第一句。第二句。第三句。"
    chunks = splitter.split_text(text)
    assert len(chunks) == 3
    assert all(len(chunk) <= 5 for chunk in chunks)
