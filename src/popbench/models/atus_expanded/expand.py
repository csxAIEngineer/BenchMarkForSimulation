"""Seeded life-history card around observed demographics.

The paragraphs are templates keyed by education, income, and marital status.
They elaborate one existing person. They are not a new respondent, and this
module refuses group rows.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from typing import Any

_CHILDHOOD = (
    "Grew up in a small town where weekends meant yard work and church potlucks.",
    "Raised in a dense suburb; after-school time was split between homework and siblings.",
    "Spent childhood near extended family; holidays were crowded and loud.",
    "Moved once as a kid for a parent's job and learned to make friends quickly.",
    "Grew up in a quiet household that valued keeping to a routine.",
)

_SCHOOLING = {
    "less_than_high_school": (
        "Left school early to help with bills and never finished a diploma track.",
        "Tried adult-ed classes later but did not complete a credential.",
    ),
    "high_school": (
        "Finished high school and went straight into local work instead of college.",
        "Graduated high school, briefly considered trade school, then took a steady job.",
    ),
    "some_college": (
        "Started community college, paused after a few semesters, and may go back.",
        "Earned certificates between jobs but never completed a bachelor's degree.",
    ),
    "bachelor_or_higher": (
        "Completed a bachelor's degree and still leans on that training at work.",
        "Finished college later than peers and treats the degree as a hard-won reset.",
    ),
    "unknown": ("Education path is uneven; learning mostly happened on the job.",),
}

_WORK_HIGH = (
    "Built a full-time career with raises tied to long hours and reliability.",
    "Moved into a higher-paying role after a stretch of overtime and skill-building.",
    "Keeps a demanding schedule; work often crowds evenings.",
)
_WORK_LOW = (
    "Pieces together part-time or gig work around other obligations.",
    "Has had uneven employment; some months are thin on hours.",
    "Prioritizes flexible shifts even when pay stays modest.",
)
_WORK_MID = (
    "Holds a regular job with ordinary hours and few dramatic promotions.",
    "Stayed in the same field for years; changes come slowly.",
)

_FAMILY = {
    "Married": (
        "Shares a household with a spouse and coordinates chores around both schedules.",
        "Married life means joint budgets, shared errands, and evening check-ins.",
    ),
    "Never married": (
        "Lives independently; close friends fill some of the roles a partner might.",
        "Has stayed single and keeps a household routine built around personal habits.",
    ),
    "Divorced": (
        "After a divorce, rebuilt a quieter home routine and clearer boundaries.",
        "Post-divorce years meant relearning solo logistics and smaller gatherings.",
    ),
    "Separated": ("Currently separated; day-to-day plans are in flux and often negotiated.",),
    "Widowed": ("Lost a spouse and adjusted to a house that feels too large on some days.",),
    "Unknown": ("Family life is private; weekdays are mostly self-managed.",),
}

_MOVES = (
    "Settled in the current state for work and stayed longer than planned.",
    "Followed family to the current state and gradually put down roots.",
    "Came for school or a first job, then the place became home.",
    "Relocated within the last decade and is still learning the local rhythms.",
)

_TURNING = (
    "A caregiving stretch reordered the week and still shows up in daily minutes.",
    "A job change lengthened the commute and shortened leisure.",
    "A health scare in the family pushed personal care and rest higher on the list.",
    "Remote or flexible work briefly expanded free time, then responsibilities crept back.",
    "Money stress led to an extra shift and less evening social time.",
)


def expand_persona(
    *,
    subject_id: str,
    demographics: Mapping[str, Any],
    routine: Mapping[str, float] | None,
    seed: int = 0,
) -> str:
    rng = _rng(seed, subject_id)
    age = demographics.get("age", "unknown")
    sex = demographics.get("sex") or "Unknown"
    state = demographics.get("state") or demographics.get("state_name")
    education = str(demographics.get("education") or "unknown")
    marital = str(demographics.get("marital_status") or "Unknown")
    race = demographics.get("race") or "Unknown"
    income_over_50k = _truthy(demographics.get("income_over_50k"))
    work_hours = _hours(demographics.get("work_hours"))
    childhood = _pick(rng, _CHILDHOOD)
    schooling = _pick(rng, _SCHOOLING.get(education, _SCHOOLING["unknown"]))
    work_path = _work_line(rng, income_over_50k=income_over_50k, work_hours=work_hours)
    family = _pick(rng, _FAMILY.get(marital, _FAMILY["Unknown"]))
    moves = _pick(rng, _MOVES)
    if state:
        moves = f"{moves} Current home state: {state}."
    turning = _turning_line(rng, routine)
    return _compose(
        age=age,
        sex=str(sex),
        state=None if state is None else str(state),
        education=education,
        marital=marital,
        race=str(race),
        income_over_50k=income_over_50k,
        work_hours=work_hours,
        childhood=childhood,
        schooling=schooling,
        work_path=work_path,
        family=family,
        moves=moves,
        turning=turning,
        routine=dict(routine or {}),
    )


def _compose(
    *,
    age: object,
    sex: str,
    state: str | None,
    education: str,
    marital: str,
    race: str,
    income_over_50k: bool,
    work_hours: float | None,
    childhood: str,
    schooling: str,
    work_path: str,
    family: str,
    moves: str,
    turning: str,
    routine: dict[str, float],
) -> str:
    income = "over $50,000" if income_over_50k else "$50,000 or less"
    hours = "unknown" if work_hours is None else f"{work_hours:.0f}"
    lines = [
        "## Basics",
        f"Age: {age}",
        f"Sex: {sex}",
        f"State: {state or 'Unknown'}",
        f"Education: {education}",
        f"Marital status: {marital}",
        f"Race: {race}",
        f"Personal income: {income}",
        f"Usual weekly work hours: {hours}",
        "",
        "## Life history",
        f"Childhood: {childhood}",
        f"Schooling: {schooling}",
        f"Work path: {work_path}",
        f"Family: {family}",
        f"Moves: {moves}",
        f"Turning point: {turning}",
    ]
    if routine:
        lines.extend(["", "## Typical day (ATUS minutes)"])
        ranked = sorted(routine.items(), key=lambda item: (-item[1], item[0]))
        lines.extend(f"- {key}: {minutes:.0f} minutes" for key, minutes in ranked)
    return "\n".join(lines).strip()


def _rng(seed: int, subject_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pick(rng: random.Random, options: tuple[str, ...]) -> str:
    return options[rng.randrange(len(options))]


def _work_line(rng: random.Random, *, income_over_50k: bool, work_hours: float | None) -> str:
    hours = work_hours
    if income_over_50k and (hours is None or hours >= 35):
        line = _pick(rng, _WORK_HIGH)
    elif (hours is not None and hours < 30) or not income_over_50k:
        line = _pick(rng, _WORK_LOW)
    else:
        line = _pick(rng, _WORK_MID)
    if hours is None:
        return f"{line} Weekly hours vary."
    return f"{line} Typical weekly hours: {hours:.0f}."


def _turning_line(rng: random.Random, routine: Mapping[str, float] | None) -> str:
    base = _pick(rng, _TURNING)
    if not routine:
        return base
    top_key, top_minutes = max(routine.items(), key=lambda item: float(item[1]))
    return (
        f"{base} Time-use note: {top_key} averages about {float(top_minutes):.0f} "
        "minutes on a diary day."
    )


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _hours(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)  # type: ignore[arg-type]
