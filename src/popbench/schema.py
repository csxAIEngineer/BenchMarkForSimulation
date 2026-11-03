"""Records shared by the dataset, persona, and communication layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Unit = Literal["individual", "group"]
AnswerKind = Literal["multiple_choice", "likert", "numeric", "distribution", "text"]

ANSWER_KINDS = ("multiple_choice", "likert", "numeric", "distribution", "text")
UNITS = ("individual", "group")


class SchemaError(ValueError):
    """A record violates the shared benchmark schema."""


@dataclass(frozen=True)
class Question:
    """One item shown to the model. Gold stays on ``Gold``, not here."""

    id: str
    prompt: str
    options: tuple[str, ...] = ()
    kind: AnswerKind = "text"

    def __post_init__(self) -> None:
        if not self.id:
            raise SchemaError("question id is required")
        if not self.prompt.strip():
            raise SchemaError(f"question {self.id} is missing a prompt")
        if self.kind not in ANSWER_KINDS:
            raise SchemaError(f"unknown question kind: {self.kind}")


@dataclass(frozen=True)
class Gold:
    """Held-out human target. Persona methods must not read this object."""

    answer: str | None = None
    distribution: tuple[tuple[str, float], ...] | None = None
    group_size: int | None = None
    payload: str | None = None

    def __post_init__(self) -> None:
        if self.distribution is not None:
            for label, weight in self.distribution:
                if not str(label):
                    raise SchemaError("distribution labels must be non-empty")
                if weight < 0:
                    raise SchemaError("distribution weights cannot be negative")
        if self.group_size is not None and self.group_size < 0:
            raise SchemaError("group_size cannot be negative")

    def distribution_dict(self) -> dict[str, float] | None:
        if self.distribution is None:
            return None
        return {label: weight for label, weight in self.distribution}


@dataclass(frozen=True)
class Subject:
    """One simulation unit: a Twin-2K respondent or a SimBench group.

    ``unit="group"`` is a published aggregate. It is not a stand-in for a
    made-up person.
    """

    id: str
    dataset: str
    unit: Unit
    demographics: Mapping[str, str]
    persona_text: str | None = None
    persona_summary: str | None = None
    demographic_text: str | None = None
    group_prompt: str | None = None
    routine: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise SchemaError("subject id is required")
        if not self.dataset:
            raise SchemaError("subject dataset is required")
        if self.unit not in UNITS:
            raise SchemaError(f"unknown subject unit: {self.unit}")
        if self.unit == "group" and not (self.group_prompt or "").strip():
            raise SchemaError(f"group subject {self.id} is missing group_prompt")


@dataclass(frozen=True)
class Case:
    """One subject paired with one question."""

    dataset: str
    subject: Subject
    question: Question
    gold: Gold | None = None

    def __post_init__(self) -> None:
        if self.subject.dataset != self.dataset:
            raise SchemaError("case dataset does not match the subject")


@dataclass(frozen=True)
class DatasetBundle:
    """Unified output of a dataset loader."""

    name: str
    unit: Unit
    source: str
    revision: str | None
    subjects: tuple[Subject, ...]
    cases: tuple[Case, ...]

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise SchemaError(f"unknown bundle unit: {self.unit}")
        for case in self.cases:
            if case.dataset != self.name:
                raise SchemaError(f"{case.dataset} case stored on {self.name}")
            if case.subject.unit != self.unit:
                raise SchemaError(
                    f"{self.name} is a {self.unit} dataset but {case.subject.id} is {case.subject.unit}"
                )

    def limited(self, limit: int | None) -> DatasetBundle:
        if limit is None:
            return self
        if limit < 1:
            raise SchemaError("limit must be a positive integer")
        cases = self.cases[:limit]
        seen: list[str] = []
        subjects: list[Subject] = []
        for case in cases:
            if case.subject.id not in seen:
                seen.append(case.subject.id)
                subjects.append(case.subject)
        return DatasetBundle(
            name=self.name,
            unit=self.unit,
            source=self.source,
            revision=self.revision,
            subjects=tuple(subjects),
            cases=cases,
        )


@dataclass(frozen=True)
class PromptPackage:
    """Persona plus question, ready for the communication layer.

    The simulation layer builds this object and does not call an HTTP client.
    """

    dataset: str
    subject_id: str
    question_id: str
    method: str
    system: str
    user: str
    response_mode: Literal["answer", "distribution"]
    persona: str = ""
    question_body: str = ""
    options: tuple[str, ...] = ()
    kind: str = ""
    outer_sample: bool = False

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subject_id": self.subject_id,
            "question_id": self.question_id,
            "method": self.method,
            "system": self.system,
            "user": self.user,
            "response_mode": self.response_mode,
            "persona": self.persona,
            "question_body": self.question_body,
            "options": list(self.options),
            "kind": self.kind,
            "outer_sample": self.outer_sample,
        }


@dataclass(frozen=True)
class SimulationResult:
    """One model reply, including the raw string and the parsed JSON."""

    subject_id: str
    question_id: str
    method: str
    provider: str
    model: str
    raw: str
    parsed: Mapping[str, Any] | None
    cached: bool
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "question_id": self.question_id,
            "method": self.method,
            "provider": self.provider,
            "model": self.model,
            "raw": self.raw,
            "parsed": None if self.parsed is None else dict(self.parsed),
            "cached": self.cached,
            "parse_error": self.parse_error,
        }
