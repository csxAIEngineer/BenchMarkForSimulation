"""Map Twin-2K Qualtrics demographic answers onto persona-card fields.

Pricing and wave-1 rows store the demographic block as QID11–QID24
(region, sex, age band, education, race, citizenship, marital status,
religion, worship, party, household income, ideology, household size,
employment). The persona card already reads age, sex, education, race,
marital status, household income, and employment. This adapter fills
those keys when the record does not already carry them, and it keeps
the original QID answers for callers that still look up QID21.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Qualtrics id, card field. Order matches the Twin-2K wave-1 demographic block.
TWIN2K_DEMOGRAPHIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("QID11", "region"),
    ("QID12", "sex"),
    ("QID13", "age"),
    ("QID14", "education"),
    ("QID15", "race"),
    ("QID16", "citizenship"),
    ("QID17", "marital_status"),
    ("QID18", "religion"),
    ("QID19", "worship_attendance"),
    ("QID20", "political_party"),
    ("QID21", "income"),
    ("QID22", "political"),
    ("QID23", "household_size"),
    ("QID24", "employment"),
)

# Usual paid week implied by the employment label. Self-employment is not here.
_FULL_TIME_WEEKLY_HOURS = 40.0
_PART_TIME_WEEKLY_HOURS = 20.0
_NONWORK_HOURS = {
    "unemployed": 0.0,
    "retired": 0.0,
    "student": 0.0,
    "home-maker": 0.0,
    "homemaker": 0.0,
    "home maker": 0.0,
}


def with_canonical_demographics(raw: Mapping[str, Any]) -> dict[str, str]:
    """Copy ``raw`` and add card keys for Twin-2K answers that are missing.

    An existing canonical value wins. Household income is also left alone
    when ``income_over_50k`` is already set, and employment is left alone
    when usual hours are already set, so an ATUS or BDM row is not
    overwritten by a stray QID.
    """
    adapted = _copy_text(raw)
    for qid, field in TWIN2K_DEMOGRAPHIC_FIELDS:
        if _field_occupied(adapted, field):
            continue
        answer = adapted.get(qid)
        if answer:
            adapted[field] = answer
    return adapted


def weekly_hours_from_employment(value: object) -> float | None:
    """Usual weekly hours implied by a Twin-2K employment label.

    Full-time is 40, part-time is 20, and statuses outside paid work are 0.
    Self-employment and unrecognized labels return ``None``.
    """
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    nonwork = _nonwork_hours(text)
    if nonwork is not None:
        return nonwork
    if "part-time" in text or "part time" in text:
        return _PART_TIME_WEEKLY_HOURS
    if "full-time" in text or "full time" in text:
        return _FULL_TIME_WEEKLY_HOURS
    return None


def _nonwork_hours(text: str) -> float | None:
    """Hours for a label that names time outside paid work.

    Checked before the full-time and part-time phrases so \"full-time student\"
    is not read as a 40-hour job.
    """
    if text in _NONWORK_HOURS:
        return _NONWORK_HOURS[text]
    for label, hours in _NONWORK_HOURS.items():
        if text.endswith(label) or f" {label}" in text:
            return hours
    return None


def _copy_text(raw: Mapping[str, Any]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            copied[str(key)] = text
    return copied


def _field_occupied(fields: Mapping[str, str], field: str) -> bool:
    """True when a canonical answer is already present.

    ``income_over_50k`` may be the string ``\"false\"``. Presence matters:
    an explicit ATUS flag, either way, keeps a QID from replacing it.
    """
    if field == "income":
        return "income" in fields or "income_over_50k" in fields
    if field == "employment":
        return "employment" in fields or "work_hours" in fields
    return field in fields
