"""Port (interface) vers un modèle de langage. Toute implémentation
(Ollama, OpenAI-compatible, etc.) doit respecter ce contrat."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Retourne une réponse texte libre."""

    @abstractmethod
    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict[str, Any]:
        """Retourne une réponse structurée (JSON). L'implémentation est
        responsable de forcer/valider le format JSON."""
