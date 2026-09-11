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

    def embed_query(
        self,
        query_text: str,
    ) -> list[float]:
        return self._embed.embed(query_text)

    def retrieve(
        self,
        query_text: str,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:

        query_embedding = self.embed_query(query_text)

        dense = self._vector_store.search(
            query_embedding,
            self._settings.top_k_dense,
        )

        lexical = self._lexical_index.search(
            query_text,
            self._settings.top_k_lexical,
        )

        if metadata_filter:
            dense = self._apply_metadata_filter(
                dense,
                metadata_filter,
            )

            lexical = self._apply_metadata_filter(
                lexical,
                metadata_filter,
            )

        merged = self._fuse(
            dense,
            lexical,
        )

        merged.sort(
            key=lambda e: e.score,
            reverse=True,
        )

        return merged[: self._settings.top_k_final]

    @staticmethod
    def _apply_metadata_filter(
        evidence_list: list[Evidence],
        metadata_filter: dict[str, Any],
    ) -> list[Evidence]:

        return [
            e
            for e in evidence_list
            if all(
                e.source_metadata.get(k) == v
                for k, v in metadata_filter.items()
            )
        ]

    def _fuse(
        self,
        dense: list[Evidence],
        lexical: list[Evidence],
    ) -> list[Evidence]:

        by_chunk: dict[str, Evidence] = {}
        scores: dict[str, float] = {}

        for evidence in dense + lexical:

            key = (
                evidence.chunk_id
                or f"text:{hash(evidence.text)}"
            )

            if key not in by_chunk:
                by_chunk[key] = evidence

            if evidence.retrieval_method == RetrievalMethod.DENSE:
                weight = self._settings.dense_weight
            else:
                weight = self._settings.lexical_weight

            scores[key] = (
                scores.get(key, 0.0)
                + evidence.score * weight
            )

        result: list[Evidence] = []

        for key, evidence in by_chunk.items():

            result.append(
                Evidence(
                    chunk_id=(
                        key
                        if not key.startswith("text:")
                        else None
                    ),
                    text=evidence.text,
                    score=scores[key],
                    source_metadata=dict(
                        evidence.source_metadata
                    ),
                    retrieval_method=RetrievalMethod.HYBRID,
                    embedding=evidence.embedding,
                    source_reliability=evidence.source_reliability,
                )
            )

        return result