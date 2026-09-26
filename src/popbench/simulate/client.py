"""DeepSeek chat client.

DeepSeek speaks the Chat Completions format. Calls are cached on disk by
model, persona id, and item id. Scoring reads the raw response strings back
from that cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Choice:
    """Structured model answer. `raw` is the unmodified response string."""

    answer: str
    rationale: str | None
    raw: str


def configured_model() -> str:
    """Model name from `DEEPSEEK_MODEL` or the project `.env`. No API key required."""
    load_env_file(_default_env_path())
    model = os.environ.get(ENV_MODEL, DEFAULT_MODEL).strip()
    return model or DEFAULT_MODEL


def model_settings_from_env() -> ModelSettings:
    """Read DeepSeek settings from the process environment and the project `.env`."""
    load_env_file(_default_env_path())
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    if not api_key:
        raise RuntimeError("missing environment variable: DEEPSEEK_API_KEY")
    base_url = os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).strip().rstrip("/")
    return ModelSettings(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        model=configured_model(),
    )


def model_from_run(meta: dict[str, object] | None) -> str:
    """Prefer the model stored with a run, then the configured model name."""
    if meta:
        recorded = meta.get("model")
        if isinstance(recorded, str) and recorded.strip() and recorded.strip().casefold() != "unknown":
            return recorded.strip()
    return configured_model()


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines. Existing process environment wins."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _default_env_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return Path.cwd() / ".env"


def response_cache_key(model: str, persona_id: str, item_id: str) -> str:
    """Stable cache key for one model call. The stored value is the raw string."""
    material = "\0".join((model, persona_id, item_id))
    return hashlib.sha256(material.encode()).hexdigest()


def chat(settings: ModelSettings, messages: list[dict[str, str]]) -> str:
    """Send one chat completion and return the raw assistant string."""
    import httpx

    response = httpx.post(
        f"{settings.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.api_key}"},
        json={
            "model": settings.model,
            "temperature": 0,
            "messages": messages,
            "thinking": {"type": "disabled"},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def complete(settings: ModelSettings, persona_prompt: str, item_prompt: str) -> str:
    """Ask the chat model for a JSON object with `answer` and optional `rationale`."""
    return chat(
        settings,
        [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": item_prompt},
        ],
    )


def parse_choice(raw: str, options: tuple[str, ...]) -> Choice:
    """Pull `answer` out of a model string and match it to one option."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response is not JSON")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict) or "answer" not in payload:
        raise ValueError("model response has no answer")
    answer = str(payload["answer"]).strip()
    rationale = payload.get("rationale")
    if rationale is not None:
        rationale = " ".join(str(rationale).split())
        if not rationale:
            rationale = None
    matched = _match_option(answer, options)
    return Choice(answer=matched, rationale=rationale, raw=raw)


def _match_option(answer: str, options: tuple[str, ...]) -> str:
    if not options:
        return answer
    for option in options:
        if answer == option:
            return option
    folded = {option.casefold(): option for option in options}
    if answer.casefold() in folded:
        return folded[answer.casefold()]
    numbered = re.fullmatch(r"(\d+)", answer)
    if numbered:
        index = int(numbered.group(1))
        if 1 <= index <= len(options):
            return options[index - 1]
    contained = [option for option in options if option.casefold() in answer.casefold()]
    if len(contained) == 1:
        return contained[0]
    raise ValueError(f"answer is not one of the options: {answer}")


class ModelError(RuntimeError):
    """A model call failed after retries."""


def chat_with_retry(
    settings: ModelSettings,
    messages: list[dict[str, str]],
    *,
    attempts: int = 6,
) -> str:
    """Call the chat model, retrying rate limits and server errors."""
    import time

    import httpx

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return chat(settings, messages)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            code = exc.response.status_code
            if code != 429 and code < 500:
                raise ModelError(f"model request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(min(60, 2**attempt))
    raise ModelError(f"model request failed after retries: {last_error}")
