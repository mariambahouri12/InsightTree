"""Implémentation du LexicalIndexPort via BM25 (rank_bm25)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from application.ports.lexical_index_port import LexicalIndexPort
from domain.entities import Chunk, Evidence
from domain.enums import RetrievalMethod

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25LexicalIndex(LexicalIndexPort):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query_text: str, top_k: int) -> list[Evidence]:
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = _tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)
        max_score = max(scores) if len(scores) else 0.0

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[Evidence] = []
        for idx in ranked_indices:
            raw_score = scores[idx]
            if raw_score <= 0:
                continue
            normalized_score = raw_score / max_score if max_score > 0 else 0.0
            chunk = self._chunks[idx]
            results.append(
                Evidence(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=normalized_score,
                    source_metadata=chunk.metadata,
                    retrieval_method=RetrievalMethod.LEXICAL,
                )
            )
        return results
