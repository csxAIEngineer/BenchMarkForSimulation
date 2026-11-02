"""Persona methods. Each subdirectory is one way to build an LLM input package.

Methods read a dataset case and return a ``PromptPackage``. They do not send
HTTP requests.
"""

from __future__ import annotations

from collections.abc import Callable

from popbench.schema import Case, PromptPackage
from popbench.models.atus_expanded import NAME as ATUS_EXPANDED
from popbench.models.atus_expanded import build as build_atus_expanded
from popbench.models.demographics_only import NAME as DEMOGRAPHICS_ONLY
from popbench.models.demographics_only import build as build_demographics_only
from popbench.models.full_persona import NAME as FULL_PERSONA
from popbench.models.full_persona import build as build_full_persona
from popbench.models.group_prompt import NAME as GROUP_PROMPT
from popbench.models.group_prompt import build as build_group_prompt
from popbench.models.persona_summary import NAME as PERSONA_SUMMARY
from popbench.models.persona_summary import build as build_persona_summary

Builder = Callable[..., PromptPackage]

METHODS: dict[str, Builder] = {
    DEMOGRAPHICS_ONLY: build_demographics_only,
    FULL_PERSONA: build_full_persona,
    PERSONA_SUMMARY: build_persona_summary,
    ATUS_EXPANDED: build_atus_expanded,
    GROUP_PROMPT: build_group_prompt,
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
