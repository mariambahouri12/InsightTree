from __future__ import annotations

from typing import Any, Optional

from application.ports.embedding_port import EmbeddingPort
from application.ports.lexical_index_port import LexicalIndexPort
from application.ports.vector_store_port import VectorStorePort
from config.settings import RetrievalSettings
from domain.entities import Evidence
from domain.enums import RetrievalMethod
from infrastructure.logging.verbose_logger import vlog


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

    # =================================================================
    # QUERY EMBEDDING
    # =================================================================

    def embed_query(
        self,
        query_text: str,
    ) -> list[float]:

        vlog("")
        vlog("[Retrieval] **Query embedding generation...")

        embedding = self._embed.embed(query_text)

        vlog(
            f"[Retrieval] Embedding généré "
            f"(dimension={len(embedding)})"
        )

        return embedding

    # =================================================================
    # RETRIEVAL
    # =================================================================

    def retrieve(
        self,
        query_text: str,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:

        vlog("")
        vlog("=" * 72)
        vlog("[Retrieval] HYBRID RETRIEVAL")
        vlog("=" * 72)

        vlog(
            f"[Retrieval] Query: {query_text}"
        )

        vlog(
            f"[Retrieval] Configuration: "
            f"dense_top_k={self._settings.top_k_dense}, "
            f"lexical_top_k={self._settings.top_k_lexical}, "
            f"final_top_k={self._settings.top_k_final}"
        )

        if metadata_filter:
            vlog(
                f"[Retrieval] Metadata filter: "
                f"{metadata_filter}"
            )
        else:
            vlog(
                "[Retrieval] Metadata filter: none"
            )

        # -------------------------------------------------------------
        # QUERY EMBEDDING
        # -------------------------------------------------------------

        query_embedding = self.embed_query(
            query_text
        )

        # -------------------------------------------------------------
        # DENSE RETRIEVAL
        # -------------------------------------------------------------

        vlog("")
        vlog(
            "[Retrieval] STEP 1/4 — DENSE RETRIEVAL"
        )

        dense = self._vector_store.search(
            query_embedding,
            self._settings.top_k_dense,
        )

        vlog(
            f"[Retrieval] Dense candidates: "
            f"{len(dense)}"
        )

        self._log_candidates(
            dense,
            method="DENSE",
        )

        # -------------------------------------------------------------
        # LEXICAL / BM25 RETRIEVAL
        # -------------------------------------------------------------

        vlog("")
        vlog(
            "[Retrieval] STEP 2/4 — LEXICAL / BM25 RETRIEVAL"
        )

        lexical = self._lexical_index.search(
            query_text,
            self._settings.top_k_lexical,
        )

        vlog(
            f"[Retrieval] Lexical candidates: "
            f"{len(lexical)}"
        )

        self._log_candidates(
            lexical,
            method="LEXICAL / BM25",
        )

        # -------------------------------------------------------------
        # METADATA FILTER
        # -------------------------------------------------------------

        vlog("")
        vlog(
            "[Retrieval] STEP 3/4 — METADATA FILTER"
        )

        if metadata_filter:

            dense_before = len(dense)
            lexical_before = len(lexical)

            dense = self._apply_metadata_filter(
                dense,
                metadata_filter,
            )

            lexical = self._apply_metadata_filter(
                lexical,
                metadata_filter,
            )

            vlog(
                f"[Retrieval] Dense: "
                f"{dense_before} -> {len(dense)}"
            )

            vlog(
                f"[Retrieval] Lexical: "
                f"{lexical_before} -> {len(lexical)}"
            )

        else:
            vlog(
                "[Retrieval] Aucun filtre metadata appliqué."
            )

        # -------------------------------------------------------------
        # RRF FUSION
        # -------------------------------------------------------------

        vlog("")
        vlog(
            "[Retrieval] STEP 4/4 — RRF FUSION"
        )

        merged = self._fuse(
            dense,
            lexical,
        )

        merged.sort(
            key=lambda e: e.score,
            reverse=True,
        )

        vlog(
            f"[Retrieval] Fused candidates: "
            f"{len(merged)}"
        )

        self._log_fused_results(
            merged
        )

        # -------------------------------------------------------------
        # FINAL TOP-K
        # -------------------------------------------------------------

        final_results = merged[
            : self._settings.top_k_final
        ]

        vlog("")
        vlog(
            "[Retrieval] FINAL TOP-K"
        )

        vlog(
            f"[Retrieval] Returning "
            f"{len(final_results)} evidence(s)"
        )

        self._log_final_results(
            final_results
        )

        vlog("=" * 72)
        vlog("")

        return final_results

    # =================================================================
    # METADATA FILTER
    # =================================================================

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

    # =================================================================
    # RRF FUSION
    # =================================================================

    def _fuse(
        self,
        dense: list[Evidence],
        lexical: list[Evidence],
    ) -> list[Evidence]:

        rrf_k = 60

        vlog(
            f"[Retrieval][RRF] k={rrf_k}"
        )

        by_chunk: dict[str, Evidence] = {}
        scores: dict[str, float] = {}

        # -------------------------------------------------------------
        # DENSE CONTRIBUTION
        # -------------------------------------------------------------

        for rank, evidence in enumerate(
            dense,
            start=1,
        ):

            key = (
                evidence.chunk_id
                or f"text:{hash(evidence.text)}"
            )

            contribution = (
                1.0
                / (rrf_k + rank)
            )

            if key not in by_chunk:
                by_chunk[key] = evidence

            scores[key] = (
                scores.get(key, 0.0)
                + contribution
            )

            vlog(
                f"[Retrieval][RRF][Dense] "
                f"rank={rank} "
                f"chunk={key} "
                f"contribution={contribution:.6f}"
            )

        # -------------------------------------------------------------
        # LEXICAL CONTRIBUTION
        # -------------------------------------------------------------

        for rank, evidence in enumerate(
            lexical,
            start=1,
        ):

            key = (
                evidence.chunk_id
                or f"text:{hash(evidence.text)}"
            )

            contribution = (
                1.0
                / (rrf_k + rank)
            )

            if key not in by_chunk:
                by_chunk[key] = evidence

            scores[key] = (
                scores.get(key, 0.0)
                + contribution
            )

            vlog(
                f"[Retrieval][RRF][Lexical] "
                f"rank={rank} "
                f"chunk={key} "
                f"contribution={contribution:.6f}"
            )

        # -------------------------------------------------------------
        # BUILD HYBRID EVIDENCE
        # -------------------------------------------------------------

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

    # =================================================================
    # LOG DENSE / LEXICAL CANDIDATES
    # =================================================================

    @staticmethod
    def _log_candidates(
        evidence_list: list[Evidence],
        method: str,
    ) -> None:

        if not evidence_list:
            vlog(
                f"[Retrieval][{method}] "
                "No candidates."
            )
            return

        for rank, evidence in enumerate(
            evidence_list,
            start=1,
        ):

            chunk_id = (
                evidence.chunk_id
                or "<no-chunk-id>"
            )

            text = (
                evidence.text
                .replace("\n", " ")
                .strip()
            )

            if len(text) > 500:
                text = (
                    text[:500]
                    + "... [truncated]"
                )

            vlog(
                f"[Retrieval][{method}] "
                f"rank={rank} "
                f"score={evidence.score:.6f} "
                f"chunk={chunk_id}"
            )

            vlog(
                f"  citation: "
                f"{evidence.citation_label()}"
            )

            vlog(
                f"  text: {text}"
            )

    # =================================================================
    # LOG RRF RESULTS
    # =================================================================

    @staticmethod
    def _log_fused_results(
        evidence_list: list[Evidence],
    ) -> None:

        if not evidence_list:
            vlog(
                "[Retrieval][RRF] No fused results."
            )
            return

        sorted_results = sorted(
            evidence_list,
            key=lambda e: e.score,
            reverse=True,
        )

        for rank, evidence in enumerate(
            sorted_results,
            start=1,
        ):

            chunk_id = (
                evidence.chunk_id
                or "<no-chunk-id>"
            )

            text = (
                evidence.text
                .replace("\n", " ")
                .strip()
            )

            if len(text) > 400:
                text = (
                    text[:400]
                    + "... [truncated]"
                )

            vlog(
                f"[Retrieval][RRF] "
                f"rank={rank} "
                f"rrf_score={evidence.score:.6f} "
                f"chunk={chunk_id}"
            )

            vlog(
                f"  text: {text}"
            )

    # =================================================================
    # LOG FINAL RESULTS
    # =================================================================

    @staticmethod
    def _log_final_results(
        evidence_list: list[Evidence],
    ) -> None:

        if not evidence_list:
            vlog(
                "[Retrieval][FINAL] "
                "No evidence returned."
            )
            return

        for rank, evidence in enumerate(
            evidence_list,
            start=1,
        ):

            chunk_id = (
                evidence.chunk_id
                or "<no-chunk-id>"
            )

            text = (
                evidence.text
                .replace("\n", " ")
                .strip()
            )

            if len(text) > 600:
                text = (
                    text[:600]
                    + "... [truncated]"
                )

            vlog(
                f"[Retrieval][FINAL] "
                f"rank={rank} "
                f"score={evidence.score:.6f} "
                f"chunk={chunk_id}"
            )

            vlog(
                f"  citation: "
                f"{evidence.citation_label()}"
            )

            vlog(
                f"  source_reliability: "
                f"{evidence.source_reliability:.3f}"
            )

            vlog(
                f"  text: {text}"
            )