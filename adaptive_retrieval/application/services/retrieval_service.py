"""Récupération hybride : recherche dense (sémantique) + lexicale (BM25),
avec fusion des scores et filtrage optionnel par métadonnées."""
from __future__ import annotations

from typing import Any, Optional

from application.ports.embedding_port import EmbeddingPort
from application.ports.lexical_index_port import LexicalIndexPort
from application.ports.vector_store_port import VectorStorePort
from config.settings import RetrievalSettings
from domain.entities import Evidence
from domain.enums import RetrievalMethod


class RetrievalService:
    def __init__(
        self,
        embedding_port: EmbeddingPort,
        vector_store: VectorStorePort,
        lexical_index: LexicalIndexPort,
        settings: RetrievalSettings,
    ) -> None:
        self._embed = embedding_port
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._settings = settings

    def retrieve(
        self,
        query_text: str,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:
        query_embedding = self._embed.embed(query_text)

        dense_hits = self._vector_store.search(query_embedding, self._settings.top_k_dense)
        lexical_hits = self._lexical_index.search(query_text, self._settings.top_k_lexical)

        if metadata_filter:
            dense_hits = self._apply_metadata_filter(dense_hits, metadata_filter)
            lexical_hits = self._apply_metadata_filter(lexical_hits, metadata_filter)

        merged = self._fuse(dense_hits, lexical_hits)
        merged.sort(key=lambda e: e.score, reverse=True)
        return merged[: self._settings.top_k_final]

    @staticmethod
    def _apply_metadata_filter(
        evidence_list: list[Evidence], metadata_filter: dict[str, Any]
    ) -> list[Evidence]:
        def matches(e: Evidence) -> bool:
            return all(e.source_metadata.get(k) == v for k, v in metadata_filter.items())

        return [e for e in evidence_list if matches(e)]

    def _fuse(self, dense: list[Evidence], lexical: list[Evidence]) -> list[Evidence]:
        """Fusion pondérée par chunk_id (les scores dense/lexical sont
        normalisés en amont par leurs adaptateurs respectifs, dans [0,1])."""
        by_chunk: dict[str, Evidence] = {}
        scores: dict[str, float] = {}

        for e in dense:
            by_chunk[e.chunk_id] = e
            scores[e.chunk_id] = scores.get(e.chunk_id, 0.0) + e.score * self._settings.dense_weight

        for e in lexical:
            scores[e.chunk_id] = scores.get(e.chunk_id, 0.0) + e.score * self._settings.lexical_weight
            if e.chunk_id not in by_chunk:
                by_chunk[e.chunk_id] = e

        fused: list[Evidence] = []
        for chunk_id, base_evidence in by_chunk.items():
            fused.append(
                Evidence(
                    chunk_id=chunk_id,
                    text=base_evidence.text,
                    score=scores[chunk_id],
                    source_metadata=base_evidence.source_metadata,
                    retrieval_method=RetrievalMethod.HYBRID,
                )
            )
        return fused
