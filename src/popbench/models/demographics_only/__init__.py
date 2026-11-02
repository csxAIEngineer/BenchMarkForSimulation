"""Demographic facts only, rendered as a compact Basics card.

Question stems are shortened to labels. This method does not write a life
history and does not invent facts that are not already on the respondent.
"""

from popbench.models._prompt import build_package
from popbench.models.card import demographic_facts, render_card
from popbench.models.errors import PersonaMethodError
from popbench.schema import Case, PromptPackage

NAME = "demographics_only"


def build(case: Case) -> PromptPackage:
    if case.subject.unit != "individual":
        raise PersonaMethodError(
            "demographics_only describes one respondent. "
            "SimBench groups use the group_prompt method."
        )
    facts = demographic_facts(case.subject.demographics, case.subject.demographic_text)
    if not facts:
        raise PersonaMethodError(
            f"{case.subject.id} has no demographic block. "
            "Fetch Twin-2K-500 or attach those personas before using demographics_only."
        )
    persona = render_card(facts)
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
