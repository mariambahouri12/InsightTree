"""Configuration centrale de l'application.

Toutes les valeurs sont surchargeables via variables d'environnement,
ce qui évite de coder en dur des paramètres d'infrastructure dans le
domaine ou l'application (Dependency Inversion / 12-factor).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.environ.get("OLLAMA_LLM_MODEL", "qwen3:8b")
    embedding_model: str = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    request_timeout_s: int = _env_int("OLLAMA_TIMEOUT_S", 300)
    temperature: float = _env_float("OLLAMA_TEMPERATURE", 0.2)
    # qwen3 supporte un mode "thinking" ; on le désactive par défaut pour la
    # génération de requêtes/JSON structuré (plus rapide, plus déterministe).
    enable_thinking: bool = os.environ.get("OLLAMA_ENABLE_THINKING", "false").lower() == "true"


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size_tokens: int = _env_int("CHUNK_SIZE_TOKENS", 400)
    chunk_overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", 60)


@dataclass(frozen=True)
class RetrievalSettings:
    top_k_dense: int = _env_int("TOP_K_DENSE", 8)
    top_k_lexical: int = _env_int("TOP_K_LEXICAL", 8)
    top_k_final: int = _env_int("TOP_K_FINAL", 6)
    dense_weight: float = _env_float("DENSE_WEIGHT", 0.6)
    lexical_weight: float = _env_float("LEXICAL_WEIGHT", 0.4)


@dataclass(frozen=True)
class TreeSearchSettings:
    """Paramètres qui gouvernent la boucle d'exploration adaptative de l'arbre."""

    max_tree_depth: int = _env_int("MAX_TREE_DEPTH", 4)
    max_iterations: int = _env_int("MAX_ITERATIONS", 25)
    max_children_per_branch: int = _env_int("MAX_CHILDREN_PER_BRANCH", 3)
    relevance_threshold: float = _env_float("RELEVANCE_THRESHOLD", 0.35)
    information_gain_threshold: float = _env_float("INFO_GAIN_THRESHOLD", 0.15)
    global_sufficiency_confidence: float = _env_float("GLOBAL_SUFFICIENCY_CONFIDENCE", 0.75)
    min_independent_branches_supported: int = _env_int("MIN_SUPPORTED_BRANCHES", 2)


@dataclass(frozen=True)
class Settings:
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    tree_search: TreeSearchSettings = field(default_factory=TreeSearchSettings)


def get_settings() -> Settings:
    return Settings()
