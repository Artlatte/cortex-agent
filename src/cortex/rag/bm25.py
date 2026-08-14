"""A small Okapi BM25 full-text index over tokenized text."""

from __future__ import annotations

import math
from typing import Any


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf"


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` into lowercase ascii runs and single CJK characters."""
    tokens: list[str] = []
    run: list[str] = []
    for char in text.lower():
        if _is_cjk(char):
            if run:
                tokens.append("".join(run))
                run.clear()
            tokens.append(char)
        elif "a" <= char <= "z" or "0" <= char <= "9":
            run.append(char)
        elif run:
            tokens.append("".join(run))
            run.clear()
    if run:
        tokens.append("".join(run))
    return tokens


class BM25Index:
    """Okapi BM25 ranking with incremental corpus statistics."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, dict[str, Any]] = {}
        self._df: dict[str, int] = {}
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        """Add or replace the document stored under ``doc_id``."""
        tokens = tokenize(text)
        freqs: dict[str, int] = {}
        for token in tokens:
            freqs[token] = freqs.get(token, 0) + 1
        if doc_id in self._docs:
            self._remove_stats(self._docs[doc_id])
        self._docs[doc_id] = {"freqs": freqs, "length": len(tokens)}
        self._total_len += len(tokens)
        for token in freqs:
            self._df[token] = self._df.get(token, 0) + 1

    def remove(self, doc_id: str) -> None:
        """Remove a document; missing ids are a no-op."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return
        self._remove_stats(doc)

    def _remove_stats(self, doc: dict[str, Any]) -> None:
        self._total_len -= doc["length"]
        for token in doc["freqs"]:
            self._df[token] -= 1
            if self._df[token] <= 0:
                del self._df[token]

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        n = len(self._docs)
        if df == 0 or n == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: list[str], doc: dict[str, Any], avgdl: float) -> float:
        if avgdl == 0.0:
            return 0.0
        doc_len = doc["length"]
        denominator_factor = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
        score = 0.0
        for token in query_tokens:
            freq = doc["freqs"].get(token, 0)
            if freq == 0:
                continue
            score += self._idf(token) * (freq * (self.k1 + 1.0)) / (freq + denominator_factor)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, score)`` pairs ordered by score."""
        query_tokens = tokenize(query)
        if not self._docs:
            return []
        avgdl = self._total_len / len(self._docs)
        scored: list[tuple[str, float, int]] = []
        for position, (doc_id, doc) in enumerate(self._docs.items()):
            score = self._score(query_tokens, doc, avgdl) if query_tokens else 0.0
            scored.append((doc_id, score, position))
        scored.sort(key=lambda item: (-item[1], item[2]))
        return [(doc_id, score) for doc_id, score, _ in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._docs)
