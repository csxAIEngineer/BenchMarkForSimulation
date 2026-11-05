"""Seeded life-history card around observed demographics.

The paragraphs are templates keyed by education, income, and marital status.
They elaborate one existing person. They are not a new respondent, and this
module refuses group rows.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping
from typing import Any

from popbench.datasets.twin2k_500.demographics import (
    weekly_hours_from_employment,
    with_canonical_demographics,
)
from popbench.models.card import render_card

# Survey outcomes. They are never copied onto the card.
_OUTCOME_KEYS = {
    "local_field",
    "local_greenhouse",
    "local_vertical",
    "nonlocal_field",
    "nonlocal_greenhouse",
    "nonlocal_vertical",
    "conclusion_purchased_product",
    "conclusion_selected_product",
    "conclusion_generated_price",
}

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
    "Living with a partner": (
        "Shares a home with a partner without a marriage, and splits day-to-day costs.",
        "Living with a partner means two schedules and one grocery list.",
    ),
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
    fields = _adapt_demographics(demographics)
    rng = _rng(seed, subject_id)
    age = fields.get("age", "unknown")
    sex = fields.get("sex") or "Unknown"
    state = fields.get("state") or fields.get("state_name")
    region = fields.get("region")
    education = str(fields.get("education") or "unknown")
    education_bucket = str(fields.get("education_bucket") or education)
    marital = str(fields.get("marital_status") or "Unknown")
    race = fields.get("race") or "Unknown"
    income_over_50k = _truthy(fields.get("income_over_50k"))
    work_hours = _hours(fields.get("work_hours"))
    childhood = _pick(rng, _CHILDHOOD)
    schooling = _pick(rng, _SCHOOLING.get(education_bucket, _SCHOOLING["unknown"]))
    work_path = _work_line(rng, income_over_50k=income_over_50k, work_hours=work_hours)
    family = _pick(rng, _FAMILY.get(_family_key(marital), _FAMILY["Unknown"]))
    moves = _place_sentence(_pick(rng, _MOVES), state=state, region=region)
    turning = _turning_line(rng, routine)
    return _compose(
        age=age,
        sex=str(sex),
        state=None if state is None else str(state),
        region=None if region is None else str(region),
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
        extras=_extra_basics(
            fields,
            region_already_shown=state is None and bool(region),
        ),
    )


def _compose(
    *,
    age: object,
    sex: str,
    state: str | None,
    region: str | None,
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
    extras: tuple[tuple[str, str], ...] = (),
) -> str:
    income = "over $50,000" if income_over_50k else "$50,000 or less"
    hours = "unknown" if work_hours is None else f"{work_hours:.0f}"
    place_label, place_value = _place_line(state, region)
    basics = (
        ("Age", str(age)),
        ("Sex", sex),
        (place_label, place_value),
        ("Education", education),
        ("Marital status", marital),
        ("Race", race),
        ("Personal income", income),
        ("Usual weekly work hours", hours),
        *extras,
    )
    life = (
        ("Childhood", childhood),
        ("Schooling", schooling),
        ("Work path", work_path),
        ("Family", family),
        ("Moves", moves),
        ("Turning point", turning),
    )
    extra: list[tuple[str, list[str]]] = []
    if routine:
        ranked = sorted(routine.items(), key=lambda item: (-item[1], item[0]))
        extra.append(
            (
                "Typical day (ATUS minutes)",
                [f"- {key}: {minutes:.0f} minutes" for key, minutes in ranked],
            )
        )
    return render_card(basics, life, extra)


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


def _adapt_demographics(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Map survey fields onto the keys this card already prints.

    BDM stores household income and education as phrases. Twin-2K stores the
    same facts as QID11–QID24. Those phrases pick the same schooling and
    income templates as the ATUS buckets. Outcome columns are dropped.
    """
    # The Twin-2K loader already canonicalizes QID rows. Pricing code can also
    # pass the raw QID dict straight onto the subject, so the card maps again.
    # Keys that are already set are left as they are.
    adapted: dict[str, Any] = {
        key: value
        for key, value in with_canonical_demographics(raw).items()
        if key not in _OUTCOME_KEYS
    }
    _fill_derived_fields(adapted)
    return adapted


def _fill_derived_fields(adapted: dict[str, Any]) -> None:
    education = adapted.get("education")
    if isinstance(education, str) and education:
        adapted["education_bucket"] = _education_bucket(education)
    if "work_hours" not in adapted:
        hours = weekly_hours_from_employment(adapted.get("employment"))
        if hours is not None:
            adapted["work_hours"] = f"{hours:.0f}"
    if "income_over_50k" not in adapted and adapted.get("income"):
        flag = _income_over_50k(str(adapted["income"]))
        adapted["income_over_50k"] = "true" if flag else "false"


def _extra_basics(
    demographics: Mapping[str, Any],
    *,
    region_already_shown: bool,
) -> tuple[tuple[str, str], ...]:
    labels = (
        ("income", "Household income"),
        ("household_size", "Household size"),
        ("political", "Political leaning"),
        ("political_party", "Political party"),
        ("employment", "Employment"),
        ("citizenship", "US citizen"),
        ("religion", "Religion"),
        ("worship_attendance", "Religious attendance"),
        ("primary_grocery_shopper", "Primary grocery shopper"),
        ("primary_cook", "Primary cook"),
    )
    if not region_already_shown:
        labels = (("region", "Region"),) + labels
    rows: list[tuple[str, str]] = []
    for key, label in labels:
        value = demographics.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            rows.append((label, text))
    return tuple(rows)


def _education_bucket(value: str) -> str:
    if value in _SCHOOLING:
        return value
    text = value.casefold().replace("’", "'")
    if "less than" in text or "no formal" in text:
        return "less_than_high_school"
    if "some college" in text or "some university" in text or "associate" in text:
        return "some_college"
    if "high school" in text:
        return "high_school"
    if any(
        token in text
        for token in (
            "bachelor",
            "college graduate",
            "postgrad",
            "graduate",
            "university",
            "professional",
            "master",
            "doctorate",
            "doctoral",
        )
    ):
        return "bachelor_or_higher"
    if "college" in text:
        return "some_college"
    return "unknown"


def _family_key(value: str) -> str:
    if value in _FAMILY:
        return value
    return {
        "never been married": "Never married",
        "never married": "Never married",
        "living with a partner": "Living with a partner",
        "married": "Married",
        "divorced": "Divorced",
        "separated": "Separated",
        "widowed": "Widowed",
    }.get(value.casefold(), "Unknown")


def _place_line(state: str | None, region: str | None) -> tuple[str, str]:
    if state:
        return "State", state
    if region:
        return "Region", region
    return "State", "Unknown"


def _place_sentence(moves: str, *, state: object, region: object) -> str:
    if state:
        return f"{moves} Current home state: {state}."
    if region:
        return f"{moves} Current census region: {region}."
    return moves


def _income_over_50k(value: str) -> bool:
    numbers = [int(piece.replace(",", "")) for piece in re.findall(r"\d[\d,]*", value)]
    if not numbers:
        return False
    return numbers[0] >= 50_000


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _hours(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None
