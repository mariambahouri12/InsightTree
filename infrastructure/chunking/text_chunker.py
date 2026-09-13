from __future__ import annotations

import re

from config.settings import ChunkingSettings
from domain.entities import Chunk, Document
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_kv,
    vlog_subsection,
    vlog_text,
)


_WORD_RE = re.compile(r"\S+")


class TextChunker:
    def __init__(
        self,
        settings: ChunkingSettings,
    ) -> None:
        self._settings = settings

    def chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:

        vlog_subsection(
            "[CHUNKING] Text chunking"
        )

        words = _WORD_RE.findall(
            document.raw_text
        )

        vlog_kv(
            "source",
            document.source_path,
        )

        vlog_kv(
            "document_id",
            document.id,
        )

        vlog_kv(
            "total_words",
            len(words),
        )

        vlog_kv(
            "chunk_size",
            self._settings.chunk_size_tokens,
        )

        vlog_kv(
            "overlap",
            self._settings.chunk_overlap_tokens,
        )

        if not words:
            vlog(
                "[CHUNKING] Document is empty."
            )
            return []

        size = max(
            1,
            self._settings.chunk_size_tokens,
        )

        overlap = min(
            max(
                0,
                self._settings.chunk_overlap_tokens,
            ),
            size - 1,
        )

        step = max(
            1,
            size - overlap,
        )

        chunks = []

        for start in range(
            0,
            len(words),
            step,
        ):
            window = words[
                start:start + size
            ]

            if not window:
                break

            chunks.append(
                Chunk(
                    document_id=document.id,
                    text=" ".join(window),
                    metadata={
                        **document.metadata,
                        "source_path": document.source_path,
                        "chunk_index": len(chunks),
                    },
                )
            )

            if (
                start + size
                >= len(words)
            ):
                break

        vlog_kv(
            "chunks_created",
            len(chunks),
        )

        # Show every chunk when verbose tracing is enabled.
        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            vlog_text(
                f"chunk #{index} "
                f"(id={chunk.id})",
                chunk.text,
                max_chars=700,
                indent=2,
            )

        return chunks