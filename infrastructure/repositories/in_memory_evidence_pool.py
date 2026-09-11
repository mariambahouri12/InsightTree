from __future__ import annotations

import math

from application.ports.embedding_port import EmbeddingPort
from application.ports.evidence_pool_port import EvidencePoolPort
from domain.entities import Evidence


class InMemoryEvidencePool(EvidencePoolPort):
    def __init__(self, embedding_port: EmbeddingPort) -> None:
        self._embed = embedding_port
        self._evidence: list[Evidence] = []
        self._keys: set[tuple[str | None, str]] = set()

    def add(self, evidence: list[Evidence]) -> None:
        for item in evidence:
            key = (item.chunk_id, item.text.strip())
            if key in self._keys:
                continue
            if item.embedding is None:
                item.embedding = self._embed.embed(item.text)
            self._evidence.append(item)
            self._keys.add(key)

    def find_relevant(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[Evidence]:
        if not query_embedding:
            return []

        scored = []
        for evidence in self._evidence:
            if evidence.embedding is None:
                continue
            similarity = self._cosine(query_embedding, evidence.embedding)
            if similarity >= min_score:
                scored.append((similarity, evidence))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def all(self) -> list[Evidence]:
        return list(self._evidence)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0
