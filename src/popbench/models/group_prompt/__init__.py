"""Use the published SimBench group prompt. Do not synthesize members."""

from popbench.schema import Case, PromptPackage
from popbench.models._prompt import build_package
from popbench.models.errors import PersonaMethodError

NAME = "group_prompt"


def build(case: Case) -> PromptPackage:
    if case.subject.unit != "group":
        raise PersonaMethodError(
            "group_prompt is for SimBench aggregates. Individual datasets use a persona method."
        )
    prompt = (case.subject.group_prompt or "").strip()
    if not prompt:
        raise PersonaMethodError(f"{case.subject.id} is missing the published group prompt")
    return build_package(case, method=NAME, persona=prompt, response_mode="distribution")
