"""Full persona text from Twin-2K-500 ``persona_text``."""

from popbench.schema import Case, PromptPackage
from popbench.models._prompt import build_package
from popbench.models.errors import PersonaMethodError

NAME = "full_persona"


def build(case: Case) -> PromptPackage:
    if case.subject.unit != "individual":
        raise PersonaMethodError(
            "full_persona is an individual Twin-2K card. SimBench stays a group distribution."
        )
    persona = (case.subject.persona_text or "").strip()
    if not persona:
        raise PersonaMethodError(
            f"{case.subject.id} has no persona_text. "
            "Load Twin-2K-500 full_persona, or attach it onto a Mega-Study row."
        )
    return build_package(case, method=NAME, persona=persona, response_mode="answer")
