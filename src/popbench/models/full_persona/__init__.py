"""Compact card from Twin-2K demographics plus a short slice of persona_text.

The published persona essay is not pasted through. Basics come from structured
demographic fields. Life history is a few clipped sentences from persona_text.
"""

from popbench.models._prompt import build_package
from popbench.models.card import condense_notes, demographic_facts, render_card
from popbench.models.errors import PersonaMethodError
from popbench.schema import Case, PromptPackage

NAME = "full_persona"


def build(case: Case) -> PromptPackage:
    if case.subject.unit != "individual":
        raise PersonaMethodError(
            "full_persona is an individual Twin-2K card. SimBench stays a group distribution."
        )
    narrative = (case.subject.persona_text or "").strip()
    if not narrative:
        raise PersonaMethodError(
            f"{case.subject.id} has no persona_text. "
            "Load Twin-2K-500 full_persona, or attach it onto a Mega-Study row."
        )
    basics = demographic_facts(case.subject.demographics, case.subject.demographic_text)
    life = [("Note", note) for note in condense_notes(narrative)]
    if not basics and not life:
        raise PersonaMethodError(f"{case.subject.id} persona_text did not yield a card")
    persona = render_card(basics, life)
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
