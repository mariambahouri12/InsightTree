"""Implémentation simple et dépendance-légère du VectorStorePort :
stockage en mémoire + similarité cosinus via numpy.

Pour un usage en production avec de gros volumes, remplacer cette classe
par un adaptateur vers Chroma / Qdrant / FAISS tout en respectant le même
port (VectorStorePort) — c'est tout l'intérêt de l'architecture hexagonale.
"""
from __future__ import annotations

import numpy as np

from application.ports.vector_store_port import VectorStorePort
from domain.entities import Chunk, Evidence
from domain.enums import RetrievalMethod


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._matrix: np.ndarray | None = None
        self._chunk_ids: list[str] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Le chunk {chunk.id} n'a pas d'embedding calculé.")
            self._chunks[chunk.id] = chunk
        self._rebuild_matrix()

    def search(self, query_embedding: list[float], top_k: int) -> list[Evidence]:
        if self._matrix is None or len(self._chunk_ids) == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        similarities = self._matrix @ query_vec
        top_indices = np.argsort(-similarities)[:top_k]

        results: list[Evidence] = []
        for idx in top_indices:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            score = float(similarities[idx])
            if score <= 0:
                continue
            results.append(
                Evidence(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=max(min(score, 1.0), 0.0),
                    source_metadata=chunk.metadata,
                    retrieval_method=RetrievalMethod.DENSE,
                )
            )
        return results

    def _rebuild_matrix(self) -> None:
        self._chunk_ids = list(self._chunks.keys())
        vectors = []
        for chunk_id in self._chunk_ids:
            vec = np.array(self._chunks[chunk_id].embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        self._matrix = np.vstack(vectors) if vectors else None
