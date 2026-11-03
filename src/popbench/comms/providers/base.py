"""Provider interface. Implementations hide HTTP details from persona methods."""

from __future__ import annotations

from typing import Protocol


class Provider(Protocol):
    name: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float | None = None,
    ) -> str:
        """Return the raw assistant string."""
