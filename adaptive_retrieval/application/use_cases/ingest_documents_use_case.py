"""Use case : ingestion et indexation de documents hétérogènes
(PDF, DOCX, XLSX, CSV, ...).

Étapes (cf. diagramme UML "Ingest and index documents"):
  1. Parser les documents
  2. Extraire le texte
  3. Découper en chunks
  4. Générer les embeddings
  5. Construire l'index lexical
  6. Stocker les métadonnées
"""
from __future__ import annotations

from application.ports.document_parser_port import DocumentParserPort
from application.ports.document_repository_port import DocumentRepositoryPort
from application.ports.embedding_port import EmbeddingPort
from application.ports.lexical_index_port import LexicalIndexPort
from application.ports.vector_store_port import VectorStorePort
from domain.entities import Chunk, Document
from infrastructure.chunking.text_chunker import TextChunker


class IngestDocumentsUseCase:
    def __init__(
        self,
        parser_factory,
        chunker: TextChunker,
        embedding_port: EmbeddingPort,
        vector_store: VectorStorePort,
        lexical_index: LexicalIndexPort,
        document_repository: DocumentRepositoryPort,
    ) -> None:
        self._parser_factory = parser_factory
        self._chunker = chunker
        self._embed = embedding_port
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._repository = document_repository

    def execute(self, file_paths: list[str]) -> list[Document]:
        documents: list[Document] = []
        for file_path in file_paths:
            document = self._ingest_one(file_path)
            documents.append(document)
        return documents

    def _ingest_one(self, file_path: str) -> Document:
        parser: DocumentParserPort = self._parser_factory.get_parser(file_path)
        document = parser.parse(file_path)
        self._repository.save(document)

        chunks: list[Chunk] = self._chunker.chunk_document(document)
        if not chunks:
            return document

        embeddings = self._embed.embed_batch([c.text for c in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self._vector_store.add_chunks(chunks)
        self._lexical_index.add_chunks(chunks)
        return document
