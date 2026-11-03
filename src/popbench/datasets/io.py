"""JSONL cache helpers shared by dataset loaders."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class DatasetError(RuntimeError):
    """A dataset cache is missing, incomplete, or has the wrong shape."""


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(f"cache file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_number} is not JSON") from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"{path}:{line_number} must be an object")
        rows.append(payload)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_jsonish(value: Any) -> Any:
    """Decode a JSON string, including the double-encoded strings some exports use."""
    current = value
    for _ in range(2):
        if not isinstance(current, str):
            return current
        text = current.strip()
        if not text:
            return None
        try:
            current = json.loads(text)
        except json.JSONDecodeError:
            return current
    return current


def require_positive_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise DatasetError("limit must be a positive integer")
