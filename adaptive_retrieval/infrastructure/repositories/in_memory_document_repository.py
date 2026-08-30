from __future__ import annotations

from application.ports.document_repository_port import DocumentRepositoryPort
from domain.entities import Document


class InMemoryDocumentRepository(DocumentRepositoryPort):
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> None:
        self._documents[document.id] = document

    def get(self, document_id: str) -> Document:
        return self._documents[document_id]

    def list_all(self) -> list[Document]:
        return list(self._documents.values())
