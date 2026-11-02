"""Compact card for the Canadian BDM lettuce survey.

Basics are the survey demographics and habits. Life history is a short seeded
expansion of those same fields. Bids and the purchase outcome are not read.
"""

from popbench.datasets.bdm_lettuce.fields import (
    BACKGROUND_KEYS,
    OUTCOME_COLUMNS,
    SURVEY_FIELDS,
)
from popbench.models._prompt import build_package
from popbench.models.bdm_survey_persona.expand import expand_bdm
from popbench.models.card import render_card
from popbench.models.errors import PersonaMethodError
from popbench.schema import Case, PromptPackage

NAME = "bdm_survey_persona"

_LABELS = {key: label for _column, key, label in SURVEY_FIELDS}


def build(case: Case, *, seed: int = 0) -> PromptPackage:
    if case.dataset != "bdm_lettuce" or case.subject.unit != "individual":
        raise PersonaMethodError(
            "bdm_survey_persona is only for individual rows in the Canadian BDM lettuce survey. "
            "It does not invent a shopper and it does not apply to SimBench groups."
        )
    demographics = {
        key: value.strip()
        for key, value in case.subject.demographics.items()
        if key in BACKGROUND_KEYS and key not in OUTCOME_COLUMNS and str(value).strip()
    }
    if not demographics:
        raise PersonaMethodError(
            f"{case.subject.id} has no survey background. "
            "bdm_survey_persona does not invent a respondent."
        )
    basics = [(_LABELS[key], demographics[key]) for key in BACKGROUND_KEYS if key in demographics]
    life = expand_bdm(subject_id=case.subject.id, demographics=demographics, seed=seed)
    persona = render_card(basics, life)
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
