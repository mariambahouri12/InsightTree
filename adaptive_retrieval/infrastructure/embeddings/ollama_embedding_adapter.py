"""Implémentation de l'EmbeddingPort via Ollama.

Par défaut on utilise un modèle d'embedding dédié (ex: nomic-embed-text),
plus adapté et moins coûteux qu'un modèle de chat pour cette tâche. Il
suffit de faire `ollama pull nomic-embed-text` (ou de changer
OLLAMA_EMBEDDING_MODEL pour pointer vers qwen3:8b si vous préférez tout
faire avec un seul modèle).
"""
from __future__ import annotations

import requests

from application.ports.embedding_port import EmbeddingPort
from config.settings import OllamaSettings


class OllamaEmbeddingAdapter(EmbeddingPort):
    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._endpoint = f"{settings.base_url.rstrip('/')}/api/embed"

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._settings.embedding_model, "input": texts}
        response = requests.post(self._endpoint, json=payload, timeout=self._settings.request_timeout_s)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Aucun embedding retourné par Ollama pour le modèle {self._settings.embedding_model}")
        return embeddings
