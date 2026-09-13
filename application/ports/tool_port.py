"""
Port (interface) pour un outil exécutable par l'Evidence Action Engine.

Un outil est distinct de la récupération hybride (dense/BM25) :
c'est du code déterministe ou spécialisé (calculatrice, requête sur
données structurées, appel API externe, etc.) invoqué explicitement
par le planner quand USE_TOOL est décidé.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from domain.entities import Evidence, ToolResult


class ToolPort(ABC):
    name: str
    description: str
    requires_data: bool

    """
    Si True, l'Evidence Action Engine doit d'abord récupérer des
    données (retrieval) et les fournir en entrée de l'outil.

    `tool_input` permet à l'Engine de transmettre une entrée structurée
    déjà préparée par un service applicatif.

    Exemple pour la calculatrice :

        {
            "operation": "subtract",
            "values": [1250, 1180]
        }

    L'outil n'a alors pas besoin de comprendre la question utilisateur.
    """

    @abstractmethod
    def execute(
        self,
        information_need: str,
        input_data: Optional[list[Evidence]] = None,
        tool_input: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        ...


class ToolRegistryPort(ABC):

    @abstractmethod
    def list_tools(self) -> list[ToolPort]:
        ...

    @abstractmethod
    def get_tool(self, name: str) -> Optional[ToolPort]:
        ...