"""Seeded life history for one Canadian BDM shopper.

Templates are chosen from age, sex, income, education, household size, and
shopping or cooking habits. Willingness-to-pay bids are not an input.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping

_CHILDHOOD = {
    "young": (
        "Grew up recently enough that grocery trips were still a family errand.",
        "Spent school-age years in a household where meals were planned around a budget.",
    ),
    "mid": (
        "Grew up with ordinary weeknight meals and a standing grocery stop.",
        "Childhood kitchens were practical; leftovers and sales both mattered.",
    ),
    "older": (
        "Grew up when produce shopping meant a weekly market or supermarket run.",
        "Learned early to stretch a grocery list across a full household.",
    ),
}

_SCHOOLING = {
    "bachelor_or_higher": (
        "Finished a university credential and still uses that training at work.",
        "Completed higher education and kept a structured weekday after it.",
    ),
    "some_college": (
        "Spent some time in college or a certificate program, then moved into work.",
        "Started post-secondary study and kept learning mostly on the job.",
    ),
    "high_school": (
        "Finished high school and went into local work rather than a long degree.",
        "Graduated high school and built skills through jobs more than campus.",
    ),
    "less_than_high_school": (
        "Left school early and learned the work routine without a diploma.",
    ),
    "unknown": (
        "Schooling is mixed; most of the useful training happened at work.",
    ),
}

_WORK = {
    "high": (
        "Holds a higher-income job and still does the household shopping around it.",
        "Work pays well enough that grocery choices are about preference as well as price.",
    ),
    "low": (
        "Works around a tighter budget; grocery trips watch the total.",
        "Income leaves less slack, so produce purchases stay deliberate.",
    ),
    "mid": (
        "Has a middle-income routine: steady work, ordinary grocery weeks.",
        "Pay covers the basics; specialty produce is an occasional choice.",
    ),
}

_ALONE = (
    "Lives alone and sets the grocery list without negotiating a larger household.",
    "A one-person household means cooking and shopping are the same person's job.",
)
_SHARED = (
    "Shares the dwelling; meals and grocery runs are coordinated with others.",
    "Household size means the cart has to cover more than one appetite.",
)


def expand_bdm(*, subject_id: str, demographics: Mapping[str, str], seed: int = 0) -> list[tuple[str, str]]:
    """Life-history lines. Missing background fields are skipped, not invented."""
    age = _age(demographics.get("age"))
    sex = (demographics.get("sex") or "").strip()
    education = (demographics.get("education") or "").strip()
    income = (demographics.get("income") or "").strip()
    household = (demographics.get("household_size") or "").strip()
    grocery = (demographics.get("primary_grocery_shopper") or "").strip()
    cook = (demographics.get("primary_cook") or "").strip()
    political = (demographics.get("political") or "").strip()
    grows = (demographics.get("grows_own_food") or "").strip()
    greens = (demographics.get("leafy_green_meals") or "").strip()
    rng = _rng(seed, subject_id, sex, education, income, household)
    lines: list[tuple[str, str]] = []
    if age is not None:
        lines.append(("Childhood", _pick(rng, _CHILDHOOD[_age_band(age)])))
    if education:
        lines.append(("Schooling", _pick(rng, _SCHOOLING[_education_bucket(education)])))
    if income:
        lines.append(("Work path", _pick(rng, _WORK[_income_band(income)])))
    if household:
        lines.append(("Household", _household_line(rng, household)))
    shopping = _shopping_line(grocery, cook)
    if shopping:
        lines.append(("Shopping", shopping))
    habits = _habits_line(grows, greens, political)
    if habits:
        lines.append(("Habits", habits))
    return lines


def _household_line(rng: random.Random, household: str) -> str:
    if household.strip() in {"1", "one"}:
        return _pick(rng, _ALONE)
    return _pick(rng, _SHARED) + f" Household size: {household}."


def _shopping_line(grocery: str, cook: str) -> str:
    parts: list[str] = []
    if grocery:
        if _yes(grocery):
            parts.append("Does the household's main grocery shopping.")
        elif _no(grocery):
            parts.append("Someone else usually does the grocery shopping.")
    if cook:
        if _yes(cook):
            parts.append("Cooks most of the household meals.")
        elif _no(cook):
            parts.append("Is not the household's main cook.")
    return " ".join(parts)


def _habits_line(grows: str, greens: str, political: str) -> str:
    parts: list[str] = []
    if grows:
        if _yes(grows):
            parts.append("Grows some food at home.")
        elif _no(grows):
            parts.append("Does not grow food at home.")
    if greens:
        parts.append(f"Leafy-green meals: {greens}.")
    if political:
        parts.append(f"Politics: {political}.")
    return " ".join(parts)


def _age(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    return int(match.group(0))


def _age_band(age: int) -> str:
    if age < 30:
        return "young"
    if age < 55:
        return "mid"
    return "older"


def _education_bucket(value: str) -> str:
    text = value.casefold()
    if "some university" in text or "college" in text:
        return "some_college"
    if any(token in text for token in ("graduate", "university", "professional", "bachelor")):
        return "bachelor_or_higher"
    if "high school" in text:
        return "high_school"
    if "less than" in text or "no formal" in text:
        return "less_than_high_school"
    return "unknown"


def _income_band(value: str) -> str:
    numbers = [int(piece.replace(",", "")) for piece in re.findall(r"\d[\d,]*", value)]
    if not numbers:
        return "mid"
    anchor = numbers[0]
    if anchor >= 75_000:
        return "high"
    if anchor < 35_000:
        return "low"
    return "mid"


def _yes(value: str) -> bool:
    return value.strip().casefold() in {"yes", "y", "true", "1"}


def _no(value: str) -> bool:
    return value.strip().casefold() in {"no", "n", "false", "0"}


def _rng(seed: int, subject_id: str, *parts: str) -> random.Random:
    material = ":".join([str(seed), subject_id, *parts])
    digest = hashlib.sha256(material.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pick(rng: random.Random, options: tuple[str, ...]) -> str:
    return options[rng.randrange(len(options))]
