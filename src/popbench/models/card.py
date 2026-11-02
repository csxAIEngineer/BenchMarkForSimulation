"""Compact persona cards shared by the individual methods.

The shape matches ``atus_expanded``: a ``## Basics`` block, an optional
``## Life history`` block, and an optional routine section. ``group_prompt``
does not use this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_FRIENDLY_LABELS = (
    ("age", "Age"),
    ("sex", "Sex"),
    ("gender", "Sex"),
    ("state", "State"),
    ("state_name", "State"),
    ("education", "Education"),
    ("education_level", "Education"),
    ("marital_status", "Marital status"),
    ("race", "Race"),
    ("occupation", "Occupation"),
    ("income", "Income"),
    ("income_over_50k", "Income over $50,000"),
    ("work_hours", "Usual weekly work hours"),
    ("household_size", "Household size"),
    ("political", "Political leaning"),
)

_LABEL_BY_KEY = dict(_FRIENDLY_LABELS)


def render_card(
    basics: Sequence[tuple[str, str]],
    life: Sequence[tuple[str, str]] | None = None,
    extra: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> str:
    """Join labeled lines into the shared markdown card."""
    chunks: list[str] = []
    _append_section(chunks, "Basics", [f"{label}: {value}" for label, value in basics])
    _append_section(chunks, "Life history", [f"{label}: {value}" for label, value in life or ()])
    for title, lines in extra or ():
        _append_section(chunks, title, list(lines))
    return "\n".join(chunks).strip()


def demographic_facts(
    demographics: Mapping[str, Any] | None,
    demographic_text: str | None = None,
) -> list[tuple[str, str]]:
    """Short labeled facts. Question stems are compressed; essays are dropped."""
    facts: list[tuple[str, str]] = []
    seen: set[str] = set()
    fields = demographics or {}
    for key, label in _FRIENDLY_LABELS:
        if key not in fields:
            continue
        value = _fact_value(key, fields.get(key))
        if not value:
            continue
        _add_fact(facts, seen, label, value)
    for label, value in _parse_demographic_text(demographic_text or ""):
        _add_fact(facts, seen, label, value)
    for key, raw in fields.items():
        if key in _LABEL_BY_KEY or _looks_like_question_id(str(key)):
            continue
        value = _clip(str(raw).strip(), 80)
        if not value:
            continue
        _add_fact(facts, seen, _humanize(str(key)), value)
    return facts


def condense_notes(text: str, *, max_lines: int = 3, max_chars: int = 140) -> list[str]:
    """Keep a few short sentences. The rest of a long narrative is left out."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not sentences:
        sentences = [cleaned]
    notes: list[str] = []
    for sentence in sentences:
        clipped = _clip(sentence, max_chars)
        if clipped in notes:
            continue
        notes.append(clipped)
        if len(notes) >= max_lines:
            break
    return notes


def _append_section(chunks: list[str], title: str, lines: list[str]) -> None:
    kept = [line for line in lines if line.strip()]
    if not kept:
        return
    if chunks:
        chunks.append("")
    chunks.append(f"## {title}")
    chunks.extend(kept)


def _add_fact(facts: list[tuple[str, str]], seen: set[str], label: str, value: str) -> None:
    folded = label.casefold()
    if not value or _covered(seen, folded):
        return
    seen.add(folded)
    facts.append((label, value))


def _covered(seen: set[str], folded: str) -> bool:
    for existing in seen:
        if folded == existing:
            return True
        if folded.startswith(existing + " ") or existing.startswith(folded + " "):
            return True
    return False


def _parse_demographic_text(text: str) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    pending: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        answer = re.match(r"^answer:\s*(.+)$", line, flags=re.IGNORECASE)
        if answer and pending:
            facts.append((_short_label(pending), _clip(answer.group(1).strip(), 80)))
            pending = None
            continue
        labeled = re.match(r"^([^:]{1,48}):\s+(.+)$", line)
        if labeled and not line.lower().startswith("answer:"):
            facts.append((labeled.group(1).strip(), _clip(labeled.group(2).strip(), 80)))
            pending = None
            continue
        pending = line
    return [(label, value) for label, value in facts if label and value]


def _fact_value(key: str, raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if key == "income_over_50k":
        return "yes" if text.casefold() in {"1", "true", "yes", "y"} else "no"
    return _clip(text, 80)


def _short_label(question: str) -> str:
    text = re.sub(r"\s+", " ", question).strip().rstrip("?").strip()
    text = re.sub(
        r"^(what is your|what are your|what's your|how old are you|"
        r"please (select|indicate|tell us))\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    words = text.split()
    if len(words) > 5:
        text = " ".join(words[:5])
    return text[:40].strip() or "Detail"


def _humanize(key: str) -> str:
    return key.replace("_", " ").strip().capitalize() or "Detail"


def _looks_like_question_id(key: str) -> bool:
    folded = key.strip()
    if not folded:
        return True
    if " " in folded:
        return True
    return bool(re.match(r"^(qid|q)\d+", folded, flags=re.IGNORECASE))


def _clip(text: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
