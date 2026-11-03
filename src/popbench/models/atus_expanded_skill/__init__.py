"""atus_expanded card plus the human-like survey-twin skill.

The compact card is built by ``render_atus_persona``. This module appends
rank anchors and the skill text. It does not call an API. Cross-person
order follows stated preferences and prior answers. Income, education,
and age stay soft cues. An id hash does not draw behavior axes.
"""

from popbench.models._prompt import build_package
from popbench.models.atus_expanded import render_atus_persona
from popbench.models.atus_expanded_skill.skill_text import SKILL_SYSTEM, SKILL_UNDER_PERSONA
from popbench.models.atus_expanded_skill.tendency import tendency_card
from popbench.schema import Case, PromptPackage

NAME = "atus_expanded_skill"


def build(case: Case, *, seed: int = 0) -> PromptPackage:
    card = render_atus_persona(case, seed=seed)
    tendency = tendency_card(
        case.subject.id,
        case.subject.demographics,
        seed=seed,
        demographic_text=case.subject.demographic_text,
    )
    persona = "\n\n".join([card, tendency, SKILL_UNDER_PERSONA.strip()])
    return build_package(
        case,
        method=NAME,
        persona=persona,
        response_mode="answer",
        system=SKILL_SYSTEM,
        outer_sample=True,
    )
