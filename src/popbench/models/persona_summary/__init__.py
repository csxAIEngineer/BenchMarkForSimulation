"""Compact card for a Twin-2K persona summary.

Basics reuse demographic facts already on the respondent. The summary itself
is split into short Life history lines instead of one free paragraph.
"""

from popbench.models._prompt import build_package
from popbench.models.card import condense_notes, demographic_facts, render_card
from popbench.models.errors import PersonaMethodError
from popbench.schema import Case, PromptPackage

NAME = "persona_summary"


def build(case: Case) -> PromptPackage:
    if case.subject.unit != "individual":
        raise PersonaMethodError(
            "persona_summary is an individual summary. It does not summarize a SimBench group into a person."
        )
    summary = (case.subject.persona_summary or "").strip()
    if not summary:
        raise PersonaMethodError(
            f"{case.subject.id} has no persona_summary. "
            "That field comes from Twin-2K-500 full_persona."
        )
    basics = demographic_facts(case.subject.demographics, case.subject.demographic_text)
    life = [("Summary", note) for note in condense_notes(summary, max_lines=4, max_chars=160)]
    if not basics and not life:
        raise PersonaMethodError(f"{case.subject.id} persona_summary did not yield a card")
    persona = render_card(basics, life)
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
