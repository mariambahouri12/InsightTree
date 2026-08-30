"""Implémentation du LLMPort via Ollama (modèle local qwen3:8b).

Ollama expose une API HTTP locale (par défaut http://localhost:11434).
Prérequis :
    ollama pull qwen3:8b
    ollama serve   # généralement déjà lancé en tâche de fond
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import requests

from application.ports.llm_port import LLMPort
from config.settings import OllamaSettings

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# qwen3 peut émettre un bloc de raisonnement <think>...</think> avant la
# réponse finale même quand "thinking" n'est pas explicitement demandé.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLMAdapter(LLMPort):
    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._endpoint = f"{settings.base_url.rstrip('/')}/api/chat"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        raw = self._chat(prompt, system, force_json=False)
        return self._strip_think(raw).strip()

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict[str, Any]:
        json_system = (system or "") + "\nTu dois répondre EXCLUSIVEMENT avec un objet JSON valide, sans texte autour."
        raw = self._chat(prompt, json_system, force_json=True)
        cleaned = self._strip_think(raw).strip()
        return self._parse_json(cleaned)

    # ------------------------------------------------------------------
    def _chat(self, prompt: str, system: Optional[str], force_json: bool) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._settings.temperature},
            "think": self._settings.enable_thinking,
        }
        if force_json:
            payload["format"] = "json"

        response = requests.post(self._endpoint, json=payload, timeout=self._settings.request_timeout_s)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _strip_think(text: str) -> str:
        return _THINK_BLOCK_RE.sub("", text)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = _JSON_FENCE_RE.search(text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Dernier recours : extraire le premier bloc { ... } équilibré.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {}
