from __future__ import annotations

import json
import re

import requests

from application.ports.llm_port import LLMPort
from config.settings import OllamaSettings

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class OllamaLLMAdapter(LLMPort):
    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings

    def generate(self, prompt: str, system: str = "") -> str:
        return self._chat(prompt, system)

    def generate_json(self, prompt: str, system: str = "") -> dict:
        raw = self._chat(prompt, system, force_json=True)
        return self._parse_json(raw)

    def _chat(self, prompt: str, system: str, force_json: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._settings.temperature},
        }
        if not self._settings.enable_thinking:
            payload["think"] = False
        if force_json:
            payload["format"] = "json"

        response = requests.post(
            f"{self._settings.base_url}/api/chat",
            json=payload,
            timeout=self._settings.request_timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("message") or {}).get("content", "").strip()

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = _JSON_FENCE_RE.sub("", raw).strip()
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return {}
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
