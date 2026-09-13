from __future__ import annotations

import time

import requests

from application.ports.embedding_port import EmbeddingPort
from config.settings import OllamaSettings
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_kv,
    vlog_subsection,
)


class OllamaEmbeddingAdapter(EmbeddingPort):
    def __init__(
        self,
        settings: OllamaSettings,
    ) -> None:
        self._settings = settings

    def embed(
        self,
        text: str,
    ) -> list[float]:

        embeddings = self.embed_batch(
            [text]
        )

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embedding."
            )

        return embeddings[0]

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        vlog_subsection(
            "[EMBEDDING] Ollama"
        )

        vlog_kv(
            "model",
            self._settings.embedding_model,
        )

        vlog_kv(
            "input_count",
            len(texts),
        )

        vlog_kv(
            "base_url",
            self._settings.base_url,
        )

        start = time.perf_counter()

        try:
            response = requests.post(
                f"{self._settings.base_url}/api/embed",
                json={
                    "model": self._settings.embedding_model,
                    "input": texts,
                },
                timeout=(
                    self._settings.request_timeout_s
                ),
            )

            response.raise_for_status()

            data = response.json()

            embeddings = data.get(
                "embeddings"
            )

            if not isinstance(
                embeddings,
                list,
            ):
                raise ValueError(
                    "Ollama response has no valid "
                    "'embeddings' field."
                )

            elapsed = (
                time.perf_counter()
                - start
            )

            dimension = (
                len(embeddings[0])
                if embeddings
                and isinstance(
                    embeddings[0],
                    list,
                )
                else 0
            )

            vlog_kv(
                "output_count",
                len(embeddings),
            )

            vlog_kv(
                "dimension",
                dimension,
            )

            vlog_kv(
                "duration",
                f"{elapsed:.3f}s",
            )

            return embeddings

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start
            )

            vlog(
                "[EMBEDDING] ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            vlog_kv(
                "duration_before_error",
                f"{elapsed:.3f}s",
            )

            raise