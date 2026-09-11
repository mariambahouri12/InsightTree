from __future__ import annotations
import re

from config.settings import ChunkingSettings
from domain.entities import Chunk, Document

_WORD_RE = re.compile(r"\S+")


class TextChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings

    def chunk_document(self, document: Document) -> list[Chunk]:
        words = _WORD_RE.findall(document.raw_text)
        if not words:
            return []

        size = max(1, self._settings.chunk_size_tokens)
        overlap = min(max(0, self._settings.chunk_overlap_tokens), size - 1)
        step = max(1, size - overlap)

        chunks = []
        for start in range(0, len(words), step):
            window = words[start:start + size]
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
            if start + size >= len(words):
                break
        return chunks
