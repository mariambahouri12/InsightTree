from __future__ import annotations
from abc import ABC, abstractmethod
from domain.entities import Evidence


class EvidencePoolPort(ABC):
    @abstractmethod
    def add(self, evidence: list[Evidence]) -> None:
        ...

    @abstractmethod
    def find_relevant(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[Evidence]:
        ...

    @abstractmethod
    def all(self) -> list[Evidence]:
        ...
