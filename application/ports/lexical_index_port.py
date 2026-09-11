from __future__ import annotations
from abc import ABC, abstractmethod
from domain.entities import Chunk, Evidence


class LexicalIndexPort(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def search(self, query_text: str, top_k: int) -> list[Evidence]:
        ...
