from __future__ import annotations

import math

from application.ports.vector_store_port import VectorStorePort
from domain.entities import Chunk, Evidence
from domain.enums import RetrievalMethod


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:

        self._chunks.extend(
            chunk
            for chunk in chunks
            if chunk.embedding is not None
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[Evidence]:

        if not self._chunks or not query_embedding:
            return []

        scored = []

        for chunk in self._chunks:

            if chunk.embedding is None:
                continue

            score = self._cosine(
                query_embedding,
                chunk.embedding,
            )

            scored.append(
                (score, chunk)
            )

        scored.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [
            Evidence(
                text=chunk.text,
                score=max(0.0, score),
                source_metadata=dict(chunk.metadata),
                retrieval_method=RetrievalMethod.DENSE,
                chunk_id=chunk.id,
                embedding=chunk.embedding,
            )
            for score, chunk in scored[:top_k]
            if score > 0
        ]

    @staticmethod
    def _cosine(
        a: list[float],
        b: list[float],
    ) -> float:

        if (
            not a
            or not b
            or len(a) != len(b)
        ):
            return 0.0

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        na = math.sqrt(
            sum(x * x for x in a)
        )

        nb = math.sqrt(
            sum(y * y for y in b)
        )

        if not na or not nb:
            return 0.0

        return dot / (na * nb)