"""Compact persona card used by ``atus_expanded``.

Sections are ``## Basics``, optional ``## Life history``, and an optional
routine block.
"""

from __future__ import annotations

from collections.abc import Sequence


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


def _append_section(chunks: list[str], title: str, lines: list[str]) -> None:
    kept = [line for line in lines if line.strip()]
    if not kept:
        return
    if chunks:
        chunks.append("")
    chunks.append(f"## {title}")
    chunks.extend(kept)
