"""Published persona summary, without the raw answer transcript."""

from popbench.schema import Case, PromptPackage
from popbench.models._prompt import build_package
from popbench.models.errors import PersonaMethodError

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
    return build_package(case, method=NAME, persona=summary, response_mode="answer")
