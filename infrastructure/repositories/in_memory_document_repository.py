from __future__ import annotations
from typing import Optional

from application.ports.document_repository_port import DocumentRepositoryPort
from domain.entities import Document


class InMemoryDocumentRepository(DocumentRepositoryPort):
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> None:
        self._documents[document.id] = document

    def get(self, document_id: str) -> Optional[Document]:
        return self._documents.get(document_id)

    def all(self) -> list[Document]:
        return list(self._documents.values())
