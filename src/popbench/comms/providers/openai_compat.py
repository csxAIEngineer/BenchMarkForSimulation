"""OpenAI-compatible chat completions, including the DeepSeek endpoint."""

from __future__ import annotations

import os
from typing import Any

from popbench.comms.errors import CommsError, CommsTransportError

DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/v1"


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60,
    ) -> None:
        self.api_key = api_key if api_key is not None else _api_key()
        self.base_url = (base_url or _base_url()).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise CommsError("missing OPENAI_API_KEY or DEEPSEEK_API_KEY")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float | None = None,
    ) -> str:
        import httpx

        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise CommsTransportError(f"chat completion failed: {exc}") from exc
        except ValueError as exc:
            raise CommsTransportError("chat completion returned non-JSON") from exc
        try:
            return str(payload["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise CommsTransportError("chat completion is missing message content") from exc


def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()


def _base_url() -> str:
    openai = os.environ.get("OPENAI_BASE_URL", "").strip()
    if openai:
        return openai
    if os.environ.get("DEEPSEEK_API_KEY", "").strip() and not os.environ.get("OPENAI_API_KEY", "").strip():
        return os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_URL).strip() or DEFAULT_DEEPSEEK_URL
    deepseek = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    return deepseek or DEFAULT_OPENAI_URL
