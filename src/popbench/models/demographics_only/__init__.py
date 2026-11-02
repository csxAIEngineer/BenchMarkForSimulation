"""Demographic block only, as in the Twin-2K demographics-only specification."""

from popbench.schema import Case, PromptPackage
from popbench.models._prompt import build_package
from popbench.models.errors import PersonaMethodError

NAME = "demographics_only"


def build(case: Case) -> PromptPackage:
    if case.subject.unit != "individual":
        raise PersonaMethodError(
            "demographics_only describes one respondent. "
            "SimBench groups use the group_prompt method."
        )
    persona = (case.subject.demographic_text or "").strip()
    if not persona:
        persona = _from_fields(case.subject.demographics)
    if not persona:
        raise PersonaMethodError(
            f"{case.subject.id} has no demographic block. "
            "Fetch Twin-2K-500 or attach those personas before using demographics_only."
        )
    return build_package(case, method=NAME, persona=persona, response_mode="answer")


def _from_fields(fields: object) -> str:
    if not isinstance(fields, dict) or not fields:
        return ""
    return "\n".join(f"{key}: {value}" for key, value in fields.items())
