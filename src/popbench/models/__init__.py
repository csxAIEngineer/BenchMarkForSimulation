"""Persona methods. Each subdirectory is one way to build an LLM input package.

Methods read a dataset case and return a ``PromptPackage``. They do not send
HTTP requests. ``atus_expanded`` is the card. ``atus_expanded_skill`` adds the
survey-twin instructions on top of that same card.
"""

from __future__ import annotations

from collections.abc import Callable

from popbench.schema import Case, PromptPackage
from popbench.models.atus_expanded import NAME as ATUS_EXPANDED
from popbench.models.atus_expanded import build as build_atus_expanded
from popbench.models.atus_expanded_skill import NAME as ATUS_EXPANDED_SKILL
from popbench.models.atus_expanded_skill import build as build_atus_expanded_skill

Builder = Callable[..., PromptPackage]

METHODS: dict[str, Builder] = {
    ATUS_EXPANDED: build_atus_expanded,
    ATUS_EXPANDED_SKILL: build_atus_expanded_skill,
}


def resolve_method(name: str) -> Builder:
    builder = METHODS.get(name)
    if builder is None:
        known = ", ".join(sorted(METHODS))
        raise KeyError(f"unknown persona method {name!r}. Known methods: {known}")
    return builder


def build_prompt(name: str, case: Case, **kwargs: object) -> PromptPackage:
    return resolve_method(name)(case, **kwargs)


__all__ = ["METHODS", "build_prompt", "resolve_method"]
