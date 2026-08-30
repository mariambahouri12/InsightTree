from __future__ import annotations

from abc import ABC, abstractmethod

from domain.entities import Document


class DocumentRepositoryPort(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        ...

    @abstractmethod
    def get(self, document_id: str) -> Document:
        ...

    @abstractmethod
    def list_all(self) -> list[Document]:
        ...
