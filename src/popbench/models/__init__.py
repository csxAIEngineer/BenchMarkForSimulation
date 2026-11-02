"""Persona methods. Each subdirectory is one way to build an LLM input package.

Methods read a dataset case and return a ``PromptPackage``. They do not send
HTTP requests. The only method is ``atus_expanded``.
"""

from __future__ import annotations

from collections.abc import Callable

from popbench.schema import Case, PromptPackage
from popbench.models.atus_expanded import NAME as ATUS_EXPANDED
from popbench.models.atus_expanded import build as build_atus_expanded

Builder = Callable[..., PromptPackage]

METHODS: dict[str, Builder] = {
    ATUS_EXPANDED: build_atus_expanded,
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
