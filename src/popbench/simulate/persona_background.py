"""Seeded life-history expansion for simulate persona prompt engineering.

Basic ACS/ATUS demographics stay fixed. A deterministic RNG (run seed +
visitor id) fills childhood, schooling, work, family, moves, and a turning
point, then those paragraphs are stitched into the system persona card.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Pools keyed so draws stay consistent with education / marital / income.
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
    "unknown": (
        "Education path is uneven; learning mostly happened on the job.",
    ),
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
    "Separated": (
        "Currently separated; day-to-day plans are in flux and often negotiated.",
    ),
    "Widowed": (
        "Lost a spouse and adjusted to a house that feels too large on some days.",
    ),
    "Unknown": (
        "Family life is private; weekdays are mostly self-managed.",
    ),
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
    visitor_id: str,
    demographics: dict[str, Any],
    routine: dict[str, float] | None,
    base_text: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Return background slots plus a full persona card for the system prompt."""
    rng = _rng(seed, visitor_id)
    age = demographics.get("age")
    sex = demographics.get("sex") or "Unknown"
    state = demographics.get("state") or demographics.get("state_name")
    # Visitor rows keep state on the parent object; callers may pass it in.
    education = str(demographics.get("education") or "unknown")
    marital = str(demographics.get("marital_status") or "Unknown")
    race = demographics.get("race") or "Unknown"
    income_over_50k = bool(demographics.get("income_over_50k"))
    work_hours = demographics.get("work_hours")

    childhood = _pick(rng, _CHILDHOOD)
    schooling = _pick(rng, _SCHOOLING.get(education, _SCHOOLING["unknown"]))
    work_path = _work_line(rng, income_over_50k=income_over_50k, work_hours=work_hours)
    family = _pick(rng, _FAMILY.get(marital, _FAMILY["Unknown"]))
    moves = _pick(rng, _MOVES)
    if state:
        moves = f"{moves} Current home state: {state}."
    turning_point = _turning_line(rng, routine)

    background = {
        "seed": seed,
        "visitor_id": visitor_id,
        "childhood": childhood,
        "schooling": schooling,
        "work_path": work_path,
        "family": family,
        "moves": moves,
        "turning_point": turning_point,
    }
    persona_text = _compose_card(
        base_text=base_text,
        age=age,
        sex=sex,
        state=state,
        education=education,
        marital=marital,
        race=race,
        income_over_50k=income_over_50k,
        work_hours=work_hours,
        background=background,
        routine=routine or {},
    )
    return {"background": background, "persona_text": persona_text}


def _rng(seed: int, visitor_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{visitor_id}".encode()).digest()
    # NumPy SeedSequence wants up to 624 uint32 words; 8 from SHA-256 is enough.
    words = np.frombuffer(digest, dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words))


def _pick(rng: np.random.Generator, options: tuple[str, ...]) -> str:
    return str(options[int(rng.integers(0, len(options)))])


def _work_line(
    rng: np.random.Generator,
    *,
    income_over_50k: bool,
    work_hours: float | None,
) -> str:
    hours = None if work_hours is None else float(work_hours)
    if income_over_50k and (hours is None or hours >= 35):
        line = _pick(rng, _WORK_HIGH)
    elif (hours is not None and hours < 30) or not income_over_50k:
        line = _pick(rng, _WORK_LOW)
    else:
        line = _pick(rng, _WORK_MID)
    if hours is not None:
        line = f"{line} Typical weekly hours: {hours:.0f}."
    else:
        line = f"{line} Weekly hours vary."
    return line


def _turning_line(rng: np.random.Generator, routine: dict[str, float] | None) -> str:
    base = _pick(rng, _TURNING)
    if not routine:
        return base
    ranked = sorted(routine.items(), key=lambda item: -float(item[1]))
    if not ranked:
        return base
    top_key, top_minutes = ranked[0]
    return (
        f"{base} Time-use note: {top_key} averages about {top_minutes:.0f} "
        "minutes on a diary day."
    )


def _compose_card(
    *,
    base_text: str,
    age: Any,
    sex: str,
    state: str | None,
    education: str,
    marital: str,
    race: str,
    income_over_50k: bool,
    work_hours: float | None,
    background: dict[str, Any],
    routine: dict[str, float],
) -> str:
    income = "over $50,000" if income_over_50k else "$50,000 or less"
    hours = "unknown" if work_hours is None else f"{float(work_hours):.0f}"
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
        f"Childhood: {background['childhood']}",
        f"Schooling: {background['schooling']}",
        f"Work path: {background['work_path']}",
        f"Family: {background['family']}",
        f"Moves: {background['moves']}",
        f"Turning point: {background['turning_point']}",
        "",
        "## Typical day (ATUS minutes)",
    ]
    for key, minutes in sorted(routine.items(), key=lambda item: -item[1]):
        lines.append(f"- {key}: {minutes:.0f} minutes")
    # Keep any narrative already on the card only if basics were missing;
    # prefer the structured expansion above for simulate.
    _ = base_text
    return "\n".join(lines).strip()
