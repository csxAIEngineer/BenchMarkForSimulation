"""Assemble a prompt package. Gold is intentionally not a parameter."""

from __future__ import annotations

from popbench.schema import Case, PromptPackage

ANSWER_SYSTEM = (
    "You answer as the person described in the persona. "
    "Stay consistent with that description. "
    'Reply with a JSON object {"answer": "...", "rationale": "..."} and no other text.'
)

DISTRIBUTION_SYSTEM = (
    "You simulate the response distribution of the described group. "
    "Do not invent an individual respondent. "
    'Reply with a JSON object {"distribution": {"option": weight}} '
    "whose weights are non-negative and sum to 1."
)


def build_package(
    case: Case,
    *,
    method: str,
    persona: str,
    response_mode: str,
    system: str | None = None,
    outer_sample: bool = False,
) -> PromptPackage:
    if response_mode not in {"answer", "distribution"}:
        raise ValueError(f"unknown response mode: {response_mode}")
    text = persona.strip()
    if not text:
        raise ValueError(f"{method} produced an empty persona")
    if system is None:
        system = ANSWER_SYSTEM if response_mode == "answer" else DISTRIBUTION_SYSTEM
    question_body = _question_body(case)
    user = "\n".join(["Persona:", text, "", "Question:", question_body])
    return PromptPackage(
        dataset=case.dataset,
        subject_id=case.subject.id,
        question_id=case.question.id,
        method=method,
        system=system,
        user=user,
        response_mode=response_mode,  # type: ignore[arg-type]
        persona=text,
        question_body=question_body,
        options=tuple(case.question.options),
        kind=case.question.kind,
        outer_sample=outer_sample,
    )


def _question_body(case: Case) -> str:
    lines = [case.question.prompt.strip()]
    if case.question.options and "Options:" not in case.question.prompt:
        lines.extend(["", "Options:", *[f"- {option}" for option in case.question.options]])
    return "\n".join(lines)
