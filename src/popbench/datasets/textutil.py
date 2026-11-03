"""Small text helpers for survey JSON. Not an LLM client."""

from __future__ import annotations

import re
from typing import Any


def strip_html(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def answer_label(question: dict[str, Any]) -> str:
    """Readable selected answer. Empty when the export has no answer."""
    answers = question.get("Answers") or {}
    if not isinstance(answers, dict):
        return ""
    selected = answers.get("SelectedText")
    if isinstance(selected, list):
        parts = [strip_html(item) for item in selected if strip_html(item)]
        return "; ".join(parts)
    if isinstance(selected, str) and selected.strip():
        return strip_html(selected)
    text = answers.get("Text")
    if isinstance(text, str) and text.strip():
        return strip_html(text)
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict):
                parts.extend(strip_html(piece) for piece in item.values() if strip_html(piece))
            else:
                cleaned = strip_html(item)
                if cleaned:
                    parts.append(cleaned)
        return "; ".join(parts)
    values = answers.get("Values")
    if isinstance(values, list) and values:
        return ", ".join(strip_html(item) for item in values if strip_html(item))
    return ""


def option_labels(question: dict[str, Any]) -> tuple[str, ...]:
    raw = question.get("Options") or question.get("Columns") or ()
    if not isinstance(raw, list):
        return ()
    labels = tuple(strip_html(item) for item in raw if strip_html(item))
    return labels
