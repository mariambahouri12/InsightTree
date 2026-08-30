from __future__ import annotations

from abc import ABC, abstractmethod

from domain.entities import Chunk, Evidence


class VectorStorePort(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Evidence]:
        ...
