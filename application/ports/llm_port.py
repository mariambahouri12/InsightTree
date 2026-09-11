from __future__ import annotations
from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        ...

    @abstractmethod
    def generate_json(self, prompt: str, system: str = "") -> dict:
        ...
