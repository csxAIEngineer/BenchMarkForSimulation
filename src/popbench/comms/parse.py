"""Parse a model reply into JSON. Transport code stays in the providers."""

from __future__ import annotations

import json
import re
from typing import Any


class ParseError(ValueError):
    """The model reply did not contain a JSON object."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ParseError("model reply is empty")
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _slice_object(text)
    if not isinstance(payload, dict):
        raise ParseError("model reply JSON must be an object")
    return payload


def _slice_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ParseError("model reply does not contain a JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ParseError("model reply contains malformed JSON") from exc
