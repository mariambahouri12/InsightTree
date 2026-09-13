from __future__ import annotations

from application.ports.document_parser_port import (
    DocumentParserPort,
)
from application.ports.document_repository_port import (
    DocumentRepositoryPort,
)
from application.ports.embedding_port import (
    EmbeddingPort,
)
from application.ports.lexical_index_port import (
    LexicalIndexPort,
)
from application.ports.vector_store_port import (
    VectorStorePort,
)
from domain.entities import Chunk, Document
from infrastructure.chunking.text_chunker import (
    TextChunker,
)
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_kv,
    vlog_section,
    vlog_subsection,
)


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

        self._parser_factory = (
            parser_factory
        )

        self._chunker = chunker

        self._embed = embedding_port

        self._vector_store = (
            vector_store
        )

        self._lexical_index = (
            lexical_index
        )

        self._repository = (
            document_repository
        )

    def execute(
        self,
        file_paths: list[str],
    ) -> list[Document]:

        vlog_section(
            "[INGESTION] Document ingestion"
        )

        vlog_kv(
            "files",
            file_paths,
        )

        documents = []

        for path in file_paths:
            documents.append(
                self._ingest_one(path)
            )

        vlog_subsection(
            "[INGESTION] Complete"
        )

        vlog_kv(
            "documents",
            len(documents),
        )

        return documents

    def _ingest_one(
        self,
        file_path: str,
    ) -> Document:

        vlog_section(
            f"[INGEST] {file_path}"
        )

        # ------------------------------------------------------------
        # PARSER
        # ------------------------------------------------------------

        vlog_subsection(
            "[1] PARSING"
        )

        parser: DocumentParserPort = (
            self._parser_factory.get_parser(
                file_path
            )
        )

        vlog_kv(
            "parser",
            type(parser).__name__,
        )

        document = parser.parse(
            file_path
        )

        vlog_kv(
            "document_id",
            document.id,
        )

        vlog_kv(
            "document_type",
            document.doc_type.value,
        )

        vlog_kv(
            "source_path",
            document.source_path,
        )

        vlog_kv(
            "raw_text_length",
            len(document.raw_text),
        )

        # ------------------------------------------------------------
        # REPOSITORY
        # ------------------------------------------------------------

        vlog_subsection(
            "[2] DOCUMENT REPOSITORY"
        )

        self._repository.save(
            document
        )

        vlog(
            "[REPOSITORY] Document saved."
        )

        # ------------------------------------------------------------
        # CHUNKING
        # ------------------------------------------------------------

        chunks: list[Chunk] = (
            self._chunker.chunk_document(
                document
            )
        )

        vlog_subsection(
            "[3] CHUNKING RESULT"
        )

        vlog_kv(
            "chunks",
            len(chunks),
        )

        if not chunks:
            vlog(
                "[INGEST] No chunks generated."
            )
            return document

        # ------------------------------------------------------------
        # EMBEDDINGS
        # ------------------------------------------------------------

        vlog_subsection(
            "[4] EMBEDDINGS"
        )

        embeddings = (
            self._embed.embed_batch(
                [
                    chunk.text
                    for chunk in chunks
                ]
            )
        )

        vlog_kv(
            "chunks",
            len(chunks),
        )

        vlog_kv(
            "embeddings",
            len(embeddings),
        )

        if len(embeddings) != len(
            chunks
        ):
            raise RuntimeError(
                "Embedding count does not "
                "match chunk count."
            )

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            chunk.embedding = embedding

        # ------------------------------------------------------------
        # VECTOR STORE
        # ------------------------------------------------------------

        vlog_subsection(
            "[5] VECTOR STORE"
        )

        self._vector_store.add_chunks(
            chunks
        )

        vlog(
            "[VECTOR STORE] Chunks indexed."
        )

        # ------------------------------------------------------------
        # BM25
        # ------------------------------------------------------------

        vlog_subsection(
            "[6] BM25 INDEX"
        )

        self._lexical_index.add_chunks(
            chunks
        )

        vlog(
            "[BM25] Chunks indexed."
        )

        # ------------------------------------------------------------
        # COMPLETE
        # ------------------------------------------------------------

        vlog_section(
            f"[INGEST COMPLETE] {file_path}"
        )

        vlog_kv(
            "document_id",
            document.id,
        )

        vlog_kv(
            "chunks_indexed",
            len(chunks),
        )

        return document