from __future__ import annotations

import json
import re
import time

import requests

from application.ports.llm_port import LLMPort
from config.settings import OllamaSettings
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_json,
    vlog_kv,
    vlog_subsection,
    vlog_text,
)


_JSON_FENCE_RE = re.compile(
    r"^```(?:json)?\s*|\s*```$",
    re.MULTILINE,
)


class OllamaLLMAdapter(LLMPort):

    def __init__(
        self,
        settings: OllamaSettings,
    ) -> None:
        self._settings = settings

    def generate(
        self,
        prompt: str,
        system: str = "",
    ) -> str:

        return self._chat(
            prompt,
            system,
        )

    def generate_json(
        self,
        prompt: str,
        system: str = "",
    ) -> dict:

        raw = self._chat(
            prompt,
            system,
            force_json=True,
        )

        return self._parse_json(
            raw
        )

    def _chat(
        self,
        prompt: str,
        system: str,
        force_json: bool = False,
    ) -> str:

        vlog_subsection(
            "[LLM] Ollama call"
        )

        vlog_kv(
            "model",
            self._settings.llm_model,
        )

        vlog_kv(
            "temperature",
            self._settings.temperature,
        )

        vlog_kv(
            "force_json",
            force_json,
        )

        vlog_kv(
            "thinking_enabled",
            self._settings.enable_thinking,
        )

        vlog_text(
            "system_prompt",
            system,
            max_chars=2500,
        )

        vlog_text(
            "user_prompt",
            prompt,
            max_chars=5000,
        )

        messages = []

        if system:
            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    self._settings.temperature
                ),
            },
        }

        if not self._settings.enable_thinking:
            payload["think"] = False

        if force_json:
            payload["format"] = "json"

        start = time.perf_counter()

        try:

            response = requests.post(
                f"{self._settings.base_url}/api/chat",
                json=payload,
                timeout=(
                    self._settings
                    .request_timeout_s
                ),
            )

            response.raise_for_status()

            data = response.json()

            output = (
                (data.get("message") or {})
                .get("content", "")
                .strip()
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            vlog_kv(
                "duration",
                f"{elapsed:.3f}s",
            )

            vlog_kv(
                "output_chars",
                len(output),
            )

            vlog_text(
                "raw_llm_output",
                output,
                max_chars=5000,
            )

            return output

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start
            )

            vlog(
                f"[LLM] ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            vlog_kv(
                "duration_before_error",
                f"{elapsed:.3f}s",
            )

            raise

    @staticmethod
    def _parse_json(
        raw: str,
    ) -> dict:

        cleaned = (
            _JSON_FENCE_RE
            .sub("", raw)
            .strip()
        )

        try:

            value = json.loads(
                cleaned
            )

            if isinstance(
                value,
                dict,
            ):

                vlog_json(
                    "[LLM] Parsed JSON",
                    value,
                )

                return value

            return {}

        except json.JSONDecodeError:

            # Fallback: find the first JSON object.
            match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            if not match:
                return {}

            try:

                value = json.loads(
                    match.group(0)
                )

                if isinstance(
                    value,
                    dict,
                ):

                    vlog_json(
                        "[LLM] Parsed JSON fallback",
                        value,
                    )

                    return value

                return {}

            except json.JSONDecodeError:

                vlog(
                    "[LLM] JSON parsing failed."
                )

                return {}