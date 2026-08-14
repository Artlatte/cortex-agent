"""In-memory cosine-similarity vector store with numpy persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cortex.rag.document import Document


class VectorStore:
    """Stores embedding vectors alongside documents and supports cosine search."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: list[list[float]] = []
        self._ids: list[str] = []
        self._documents: list[Document] = []
        self._id_index: dict[str, int] = {}

    def add(
        self,
        vectors: list[list[float]],
        ids: list[str],
        documents: list[Document],
    ) -> None:
        """Add documents, replacing any existing entry with the same id."""
        for vector, doc_id, document in zip(vectors, ids, documents, strict=True):
            if doc_id in self._id_index:
                index = self._id_index[doc_id]
                self._vectors[index] = list(vector)
                self._documents[index] = document
            else:
                self._id_index[doc_id] = len(self._ids)
                self._ids.append(doc_id)
                self._vectors.append(list(vector))
                self._documents.append(document)

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Return ``(doc_id, cosine_similarity)`` pairs ordered descending."""
        if not self._vectors:
            return []
        matrix = np.asarray(self._vectors, dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        query_norm = query / (np.linalg.norm(query) + 1e-12)
        similarities = matrix_norm @ query_norm
        order = np.argsort(-similarities, kind="stable")
        top = order[:top_k]
        return [(self._ids[int(i)], float(similarities[int(i)])) for i in top]

    def get(self, doc_id: str) -> Document:
        """Return the document for ``doc_id`` or raise :class:`KeyError`."""
        index = self._id_index.get(doc_id)
        if index is None:
            raise KeyError(doc_id)
        return self._documents[index]

    def items(self) -> list[tuple[str, Document]]:
        """Return ``(doc_id, Document)`` pairs in insertion order."""
        return list(zip(self._ids, self._documents, strict=True))

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, directory: str | Path) -> None:
        """Persist vectors and documents to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", np.asarray(self._vectors, dtype=np.float32))
        meta = {
            "dim": self.dim,
            "ids": list(self._ids),
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self._documents
            ],
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        """Reconstruct a :class:`VectorStore` previously written with :meth:`save`."""
        directory = Path(directory)
        array = np.load(directory / "vectors.npy")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        dim = int(meta.get("dim", array.shape[1] if array.ndim > 1 else 0))
        store = cls(dim)
        ids = [str(item) for item in meta["ids"]]
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in meta["documents"]
        ]
        vectors = array.tolist() if array.size else []
        store.add(vectors, ids, documents)
        return store
