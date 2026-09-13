from __future__ import annotations

import math

from domain.entities import Chunk, Evidence
from domain.enums import RetrievalMethod
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_evidence,
    vlog_kv,
    vlog_subsection,
)
from application.ports.vector_store_port import VectorStorePort


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:

        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.embedding is not None
        ]

        self._chunks.extend(
            valid_chunks
        )

        vlog_subsection(
            "[VECTOR STORE] Index chunks"
        )

        vlog_kv(
            "received_chunks",
            len(chunks),
        )

        vlog_kv(
            "indexed_chunks",
            len(valid_chunks),
        )

        vlog_kv(
            "total_chunks_in_store",
            len(self._chunks),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[Evidence]:

        vlog_subsection(
            "[DENSE RETRIEVAL] Vector search"
        )

        vlog_kv(
            "indexed_chunks",
            len(self._chunks),
        )

        vlog_kv(
            "top_k",
            top_k,
        )

        vlog_kv(
            "query_dimension",
            len(query_embedding)
            if query_embedding
            else 0,
        )

        if (
            not self._chunks
            or not query_embedding
        ):
            vlog(
                "[DENSE RETRIEVAL] "
                "No searchable chunks."
            )
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

        results = [
            Evidence(
                text=chunk.text,
                score=max(
                    0.0,
                    score,
                ),
                source_metadata=dict(
                    chunk.metadata
                ),
                retrieval_method=(
                    RetrievalMethod.DENSE
                ),
                chunk_id=chunk.id,
                embedding=chunk.embedding,
            )
            for score, chunk in scored[:top_k]
            if score > 0
        ]

        vlog_kv(
            "results",
            len(results),
        )

        for index, evidence in enumerate(
            results,
            start=1,
        ):
            vlog_evidence(
                evidence,
                index=index,
                max_chars=700,
            )

        return results

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
            sum(
                x * x
                for x in a
            )
        )

        nb = math.sqrt(
            sum(
                y * y
                for y in b
            )
        )

        if not na or not nb:
            return 0.0

        return dot / (na * nb)