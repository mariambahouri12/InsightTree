"""Découpage d'un document en chunks de taille contrôlée avec chevauchement,
pour préserver le contexte local entre chunks adjacents."""
from __future__ import annotations

from config.settings import ChunkingSettings
from domain.entities import Chunk, Document


class TextChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self._chunk_size = settings.chunk_size_tokens
        self._overlap = settings.chunk_overlap_tokens

    def chunk_document(self, document: Document) -> list[Chunk]:
        words = document.raw_text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = max(self._chunk_size - self._overlap, 1)
        index = 0
        chunk_number = 0

        while index < len(words):
            window = words[index : index + self._chunk_size]
            text = " ".join(window)
            metadata = {
                **document.metadata,
                "source_path": document.source_path,
                "chunk_number": chunk_number,
            }
            chunks.append(Chunk(document_id=document.id, text=text, metadata=metadata))
            chunk_number += 1
            index += step

        return chunks
