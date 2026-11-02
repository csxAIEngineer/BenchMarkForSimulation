"""ATUS-style expansion of one existing person's demographics and time use."""

from popbench.schema import Case, PromptPackage
from popbench.models._prompt import build_package
from popbench.models.atus_expanded.expand import expand_persona
from popbench.models.errors import PersonaMethodError

NAME = "atus_expanded"


def build(case: Case, *, seed: int = 0) -> PromptPackage:
    if case.subject.unit != "individual":
        raise PersonaMethodError(
            "atus_expanded expands one existing person. "
            "It does not turn a SimBench group distribution into individuals."
        )
    if case.dataset == "simbench":
        raise PersonaMethodError("atus_expanded refuses SimBench rows")
    demographics = dict(case.subject.demographics)
    if not demographics and not (case.subject.demographic_text or "").strip():
        raise PersonaMethodError(
            f"{case.subject.id} has no demographics to expand. "
            "atus_expanded does not invent a respondent."
        )
    persona = expand_persona(
        subject_id=case.subject.id,
        demographics=demographics,
        routine=case.subject.routine,
        seed=seed,
    )
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
