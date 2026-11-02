"""Persona card for the Canadian BDM lettuce survey.

The card is the respondent's demographics and shopping habits. It does not
read willingness-to-pay bids or the purchase outcome.
"""

from popbench.datasets.bdm_lettuce.fields import OUTCOME_COLUMNS, render_background
from popbench.models._prompt import build_package
from popbench.models.errors import PersonaMethodError
from popbench.schema import Case, PromptPackage

NAME = "bdm_survey_persona"


def build(case: Case) -> PromptPackage:
    if case.dataset != "bdm_lettuce" or case.subject.unit != "individual":
        raise PersonaMethodError(
            "bdm_survey_persona is only for individual rows in the Canadian BDM lettuce survey. "
            "It does not invent a shopper and it does not apply to SimBench groups."
        )
    demographics = {
        key: value
        for key, value in case.subject.demographics.items()
        if key not in OUTCOME_COLUMNS
    }
    persona = render_background(demographics)
    if not persona:
        raise PersonaMethodError(
            f"{case.subject.id} has no survey background. "
            "bdm_survey_persona does not invent a respondent."
        )
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
