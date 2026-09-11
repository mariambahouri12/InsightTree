from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from domain.entities import Evidence, ToolResult


class ToolPort(ABC):
    name: str
    description: str
    requires_data: bool

    @abstractmethod
    def execute(
        self,
        information_need: str,
        input_data: Optional[list[Evidence]] = None,
    ) -> ToolResult:
        ...


class ToolRegistryPort(ABC):
    @abstractmethod
    def list_tools(self) -> list[ToolPort]:
        ...

    @abstractmethod
    def get_tool(self, name: str) -> Optional[ToolPort]:
        ...
