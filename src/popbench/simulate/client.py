"""Chat client for the models under test.

Calls use the Chat Completions shape. The compare models share the CR API.
Calls are cached on disk by model,
persona id, and item id. Scoring reads the raw response strings back from
that cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# Fixed set for `popbench compare`. These ids must exist in cr_api.TEXT_MODELS.
COMPARE_MODELS = ("gpt-5.5", "glm-5.3", "deepseek-v4-pro")


def _prefer_repo_cr_api() -> None:
    """Import the checkout's cr_api package, not a copy installed elsewhere."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "cr_api" / "client.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_prefer_repo_cr_api()
from cr_api.client import BASE_URL as CR_BASE_URL
from cr_api.client import CRClient, CRError, TEXT_MODELS


class SettingsError(RuntimeError):
    """Model API settings are missing or the model id is unknown."""


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float | None = 0
    max_tokens: int | None = None
    token_field: str = "max_tokens"
    extra_body: dict[str, Any] = field(default_factory=dict)
    transport: str = "deepseek"


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
        raise SettingsError("missing environment variable: DEEPSEEK_API_KEY")
    base_url = os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).strip().rstrip("/")
    return ModelSettings(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        model=configured_model(),
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
        transport="deepseek",
    )


def _cr_text_models() -> set[str]:
    return {model for models in TEXT_MODELS.values() for model in models}


def model_catalog() -> dict[str, dict[str, Any]]:
    """gpt-5.5, glm-5.3, and deepseek-v4-pro are called through cr_api.CRClient.

    The client posts to `{BASE_URL}/v1/chat/completions` with the model id.
    GLM and DeepSeek think before answering, so max_tokens stays unset.
    """
    known = _cr_text_models()
    missing = [model for model in COMPARE_MODELS if model not in known]
    if missing:
        raise SettingsError("cr_api has no text model: " + ", ".join(missing))
    shared = {
        "key_envs": ("CR_API_KEY",),
        "base_url_env": "CR_BASE_URL",
        "default_base_url": CR_BASE_URL,
        "temperature": None,
        "max_tokens": None,
        "token_field": "max_tokens",
        "extra_body": {},
        "transport": "cr",
    }
    return {model: {**shared, "model": model} for model in COMPARE_MODELS}


def canonical_compare_model(name: str) -> str | None:
    """Map a user label onto a compare-model slug, or None if it is not one."""
    text = name.strip().lower().replace(" ", "").replace("_", "-")
    compact = text.replace("-", "").replace(".", "")
    aliases = {
        "gpt55": "gpt-5.5",
        "glm53": "glm-5.3",
        "deepseekv4pro": "deepseek-v4-pro",
    }
    if compact in aliases:
        return aliases[compact]
    if text in model_catalog():
        return text
    return None


def run_slug(model: str) -> str:
    """Filesystem-safe run directory name for a model id."""
    slug = canonical_compare_model(model) or model.strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-")
    return cleaned or "model"


def model_settings_for(name: str) -> ModelSettings:
    """Settings for one named model.

    The three compare models use cr_api.CRClient. Any other name is sent to
    the DeepSeek endpoint, with that name as the model id.
    """
    load_env_file(_default_env_path())
    text = name.strip()
    if not text:
        raise SettingsError("model name is empty")
    slug = canonical_compare_model(text)
    if slug is None:
        base = model_settings_from_env()
        return replace(base, model=text)
    spec = model_catalog()[slug]
    api_key = ""
    for env_name in spec["key_envs"]:
        api_key = os.environ.get(env_name, "").strip()
        if api_key:
            break
    if not api_key:
        names = " or ".join(spec["key_envs"])
        raise SettingsError(f"missing environment variable: {names}")
    if spec["transport"] == "cr":
        base_url = CR_BASE_URL
    else:
        base_url = os.environ.get(spec["base_url_env"], "").strip().rstrip("/") or spec["default_base_url"]
    return ModelSettings(
        base_url=base_url,
        api_key=api_key,
        model=spec["model"],
        temperature=spec["temperature"],
        max_tokens=spec["max_tokens"],
        token_field=spec["token_field"],
        extra_body=dict(spec["extra_body"]),
        transport=spec["transport"],
    )


def request_record(settings: ModelSettings) -> dict[str, Any]:
    """Request fields stored in a run. The API key stays out."""
    return {
        "transport": settings.transport,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "token_field": settings.token_field,
        "extra_body": settings.extra_body,
    }


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


def completion_payload(settings: ModelSettings, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Chat Completions body for one model. Provider-specific fields come from settings."""
    body: dict[str, Any] = {"model": settings.model, "messages": messages}
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.max_tokens is not None:
        body[settings.token_field] = settings.max_tokens
    body.update(settings.extra_body)
    return body


def assistant_text(payload: dict[str, Any]) -> str:
    """Final answer string from a Chat Completions payload."""
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError("model response has no choices") from exc
    if not isinstance(message, dict):
        raise ModelError("model response has no message")
    text = _coerce_content(message.get("content"))
    if text.strip():
        return text
    if message.get("reasoning_content"):
        raise ModelError("model returned reasoning without an answer; increase max tokens")
    raise ModelError("model response has no content")


def _coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return str(content)


def chat(settings: ModelSettings, messages: list[dict[str, str]]) -> str:
    """Send one chat completion and return the raw assistant string."""
    if settings.transport == "cr":
        return _chat_via_cr(settings, messages)
    import httpx

    response = httpx.post(
        f"{settings.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.api_key}"},
        json=completion_payload(settings, messages),
        timeout=180,
    )
    response.raise_for_status()
    return assistant_text(response.json())


def _chat_via_cr(settings: ModelSettings, messages: list[dict[str, str]]) -> str:
    """Call gpt-5.5, glm-5.3, or deepseek-v4-pro through cr_api.CRClient."""
    client = CRClient(api_key=settings.api_key)
    try:
        result = client.chat(
            messages,
            model=settings.model,
            max_tokens=settings.max_tokens,
            timeout=300,
        )
    except CRError as exc:
        retryable = exc.status is None or exc.status == 429 or exc.status >= 500
        raise ModelError(str(exc), status=exc.status, retryable=retryable) from exc
    text = result.content or ""
    if text.strip():
        return text
    if result.reasoning_content:
        raise ModelError("model returned reasoning without an answer; increase max tokens")
    raise ModelError("model response has no content")


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
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
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
    """A model call failed."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


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
        except ModelError as exc:
            if not exc.retryable:
                raise
            last_error = exc
        except httpx.HTTPStatusError as exc:
            last_error = exc
            code = exc.response.status_code
            if code != 429 and code < 500:
                detail = exc.response.text[:500].replace("\n", " ")
                raise ModelError(f"model request failed ({code}): {detail}") from exc
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(min(60, 2**attempt))
    raise ModelError(f"model request failed after retries: {last_error}", retryable=True)
