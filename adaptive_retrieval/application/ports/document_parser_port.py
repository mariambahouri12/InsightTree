from __future__ import annotations

from abc import ABC, abstractmethod

from domain.entities import Document


class DocumentParserPort(ABC):
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        ...

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        ...
