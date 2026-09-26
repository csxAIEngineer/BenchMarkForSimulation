"""Records shared by the data layer, the simulator, and the scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AnswerKind = Literal["multiple_choice", "likert", "binary", "numeric"]
Suite = Literal["decision", "sentiment"]

# Published Twin-2K-500 sample size (U.S. adults).
TWIN2K_N = 2058

ANSWER_KINDS = ("multiple_choice", "likert", "binary", "numeric")
SUITES = ("decision", "sentiment")

# Demographic margins scored inside the sentiment suite.
SUBGROUP_FIELDS = ("age", "sex", "education", "party", "income")

# Twin-2K-50 is the 44-item Big Five Inventory plus six green-consumption items.
TWIN2K50_TURNS = 50
TWIN2K50_INSTRUMENT = "bfi44_plus_green6"


@dataclass(frozen=True)
class Item:
    """One scored question.

    `options` is empty for numeric items. `human_shares` and `country` are
    optional so a later country-level or group-level poll can use the same type
    without a Twin-2K respondent row.
    """

    id: str
    prompt: str
    options: tuple[str, ...]
    kind: AnswerKind
    wave: str
    suite: Suite
    subgroup_columns: tuple[str, ...] = ()
    country: str | None = None
    human_shares: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("item id is required")
        if self.kind not in ANSWER_KINDS:
            raise ValueError(f"unknown item kind: {self.kind}")
        if self.suite not in SUITES:
            raise ValueError(f"unknown suite: {self.suite}")
        if self.human_shares is not None and self.options:
            if len(self.human_shares) != len(self.options):
                raise ValueError("human_shares must have one weight per option")
            if any(weight < 0 for weight in self.human_shares):
                raise ValueError("human_shares cannot be negative")


@dataclass(frozen=True)
class PersonaCard:
    """The text the model sees when it answers as one person.

    Demographic fields are filled for Nemotron adults. Twin-2K cards carry
    only the narrative, which holds the held-out answers out of the prompt.
    """

    id: str
    condition: str
    text: str
    age: int | None = None
    age_band: str | None = None
    sex: str | None = None
    state: str | None = None
    education_level: str | None = None
    occupation: str | None = None
    marital_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {"id": self.id, "condition": self.condition}
        if self.age is not None:
            raw["age"] = self.age
            raw["age_band"] = self.age_band
            raw["sex"] = self.sex
            raw["state"] = self.state
            raw["education_level"] = self.education_level
            raw["occupation"] = self.occupation
            raw["marital_status"] = self.marital_status
        raw["text"] = self.text
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PersonaCard:
        return cls(
            id=str(raw["id"]),
            condition=str(raw["condition"]),
            text=str(raw["text"]),
            age=None if raw.get("age") is None else int(raw["age"]),
            age_band=raw.get("age_band"),
            sex=raw.get("sex"),
            state=raw.get("state"),
            education_level=raw.get("education_level"),
            occupation=raw.get("occupation"),
            marital_status=raw.get("marital_status"),
        )


@dataclass(frozen=True)
class InterviewTurn:
    """One closed-ended turn. Gold fields are for scoring, not for the prompt."""

    turn: int
    item_id: str
    scale: str
    prompt: str
    options: tuple[str, ...]
    reverse: bool
    gold_answer: str | None = None
    gold_code: int | None = None
    label_text: str | None = None
    include_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "turn": self.turn,
            "item_id": self.item_id,
            "scale": self.scale,
            "prompt": self.prompt,
            "options": list(self.options),
            "gold_answer": self.gold_answer,
            "gold_code": self.gold_code,
        }
        if self.include_label:
            raw["label_text"] = self.label_text
        raw["reverse"] = self.reverse
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InterviewTurn:
        return cls(
            turn=int(raw["turn"]),
            item_id=str(raw["item_id"]),
            scale=str(raw["scale"]),
            prompt=str(raw["prompt"]),
            options=tuple(raw["options"]),
            reverse=bool(raw["reverse"]),
            gold_answer=raw.get("gold_answer"),
            gold_code=None if raw.get("gold_code") is None else int(raw["gold_code"]),
            label_text=raw.get("label_text"),
            include_label="label_text" in raw,
        )


@dataclass(frozen=True)
class InterviewRecord:
    """One person, the persona text, and the twin-2k-50 turns they will answer."""

    id: str
    region: str
    language: str
    source: str
    instrument: str
    persona: PersonaCard
    turns: tuple[InterviewTurn, ...]
    scale_scores: tuple[tuple[str, float], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "id": self.id,
            "region": self.region,
            "language": self.language,
            "source": self.source,
            "instrument": self.instrument,
            "persona": self.persona.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
        }
        if self.scale_scores is not None:
            raw["scale_scores"] = {key: value for key, value in self.scale_scores}
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> InterviewRecord:
        scores = raw.get("scale_scores")
        return cls(
            id=str(raw["id"]),
            region=str(raw["region"]),
            language=str(raw["language"]),
            source=str(raw["source"]),
            instrument=str(raw["instrument"]),
            persona=PersonaCard.from_dict(raw["persona"]),
            turns=tuple(InterviewTurn.from_dict(turn) for turn in raw["turns"]),
            scale_scores=None
            if scores is None
            else tuple((str(key), float(value)) for key, value in scores.items()),
        )
