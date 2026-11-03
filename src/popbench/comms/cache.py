"""Disk cache for one provider call. Keys never include API secrets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from popbench.schema import PromptPackage


def cache_key(
    package: PromptPackage,
    *,
    provider: str,
    model: str,
    temperature: float | None,
) -> str:
    return cache_key_messages(
        package.messages(),
        provider=provider,
        model=model,
        temperature=temperature,
        response_mode=package.response_mode,
    )


def cache_key_messages(
    messages: list[dict[str, str]],
    *,
    provider: str,
    model: str,
    temperature: float | None,
    response_mode: str,
) -> str:
    """Hash one provider call. A batched request is one key, not one key per question."""
    material = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_mode": response_mode,
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_cache(directory: Path, key: str) -> str | None:
    path = directory / f"{key}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "raw" not in payload:
        return None
    return str(payload["raw"])


def write_cache(directory: Path, key: str, raw: str, parsed: dict[str, Any] | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"raw": raw, "parsed": parsed}
    (directory / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
