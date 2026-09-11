from __future__ import annotations

from dataclasses import dataclass
import os


def _env_int(
    name: str,
    default: int,
) -> int:
    try:
        return int(
            os.getenv(
                name,
                default,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _env_float(
    name: str,
    default: float,
) -> float:
    try:
        return float(
            os.getenv(
                name,
                default,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _env_bool(
    name: str,
    default: bool,
) -> bool:

    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    llm_model: str = os.getenv(
        "OLLAMA_LLM_MODEL",
        "qwen3:8b",
    )

    embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text",
    )

    temperature: float = _env_float(
        "OLLAMA_TEMPERATURE",
        0.1,
    )

    request_timeout_s: float = _env_float(
        "OLLAMA_TIMEOUT_S",
        120.0,
    )

    enable_thinking: bool = _env_bool(
        "OLLAMA_ENABLE_THINKING",
        False,
    )


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size_tokens: int = _env_int(
        "CHUNK_SIZE",
        500,
    )

    chunk_overlap_tokens: int = _env_int(
        "CHUNK_OVERLAP",
        80,
    )


@dataclass(frozen=True)
class RetrievalSettings:
    top_k_dense: int = _env_int(
        "TOP_K_DENSE",
        4,
    )

    top_k_lexical: int = _env_int(
        "TOP_K_LEXICAL",
        4,
    )

    top_k_final: int = _env_int(
        "TOP_K_FINAL",
        6,
    )

    dense_weight: float = _env_float(
        "DENSE_WEIGHT",
        0.6,
    )

    lexical_weight: float = _env_float(
        "LEXICAL_WEIGHT",
        0.4,
    )


@dataclass(frozen=True)
class EvidenceActionEngineSettings:
    max_action_steps: int = _env_int(
        "MAX_ACTION_STEPS",
        3,
    )

    reuse_similarity_threshold: float = _env_float(
        "REUSE_SIMILARITY_THRESHOLD",
        0.80,
    )

    reuse_top_k: int = _env_int(
        "REUSE_TOP_K",
        3,
    )


@dataclass(frozen=True)
class TreeSearchSettings:
    max_initial_branches: int = _env_int(
        "MAX_INITIAL_BRANCHES",
        4,
    )

    max_tree_depth: int = _env_int(
        "MAX_TREE_DEPTH",
        4,
    )

    max_iterations: int = _env_int(
        "MAX_ITERATIONS",
        25,
    )

    max_expansion_rounds: int = _env_int(
        "MAX_EXPANSION_ROUNDS",
        3,
    )

    max_children_per_branch: int = _env_int(
        "MAX_CHILDREN_PER_BRANCH",
        3,
    )

    relevance_threshold: float = _env_float(
        "RELEVANCE_THRESHOLD",
        0.30,
    )

    information_gain_threshold: float = _env_float(
        "INFO_GAIN_THRESHOLD",
        0.15,
    )

    global_sufficiency_confidence: float = _env_float(
        "GLOBAL_SUFFICIENCY_CONFIDENCE",
        0.75,
    )


@dataclass(frozen=True)
class Settings:
    ollama: OllamaSettings = OllamaSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    action_engine: EvidenceActionEngineSettings = (
        EvidenceActionEngineSettings()
    )
    tree_search: TreeSearchSettings = (
        TreeSearchSettings()
    )


def get_settings() -> Settings:
    return Settings()