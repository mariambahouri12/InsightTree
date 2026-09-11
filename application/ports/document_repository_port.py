from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from domain.entities import Document


class DocumentRepositoryPort(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        ...

    @abstractmethod
    def get(self, document_id: str) -> Optional[Document]:
        ...

    @abstractmethod
    def all(self) -> list[Document]:
        ...
