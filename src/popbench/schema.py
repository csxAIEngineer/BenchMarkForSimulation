"""Closed-ended survey item shared by every suite and later dataset adapter."""

from dataclasses import dataclass
from typing import Literal

AnswerKind = Literal["multiple_choice", "likert", "binary", "numeric"]
Suite = Literal["decision", "sentiment"]

# Published Twin-2K-500 sample size (U.S. adults).
TWIN2K_N = 2058

ANSWER_KINDS = ("multiple_choice", "likert", "binary", "numeric")
SUITES = ("decision", "sentiment")

# Demographic margins scored inside the sentiment suite.
SUBGROUP_FIELDS = ("age", "sex", "education", "party", "income")


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
