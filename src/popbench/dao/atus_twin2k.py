"""ATUS adults as persona cards, answering a short Twin-2K multiple-choice set.

People are drawn from the ATUS 2023 activity summary. The persona card uses
that respondent's demographics and diary minutes, plus a seeded life history.
The questions are six Twin-2K agreement items. Scoring uses Twin-2K option
shares on those same items. ATUS respondents are a different sample, so the
panel carries no per-person gold answer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from popbench.dao.interview import (
    OPTIONS,
    InterviewError,
    build_instrument,
    human_baseline,
)
from popbench.dao.io import write_jsonl
from popbench.dao.personas import age_band
from popbench.dao.schema import InterviewRecord, InterviewTurn, PersonaCard
from popbench.dao.visitors import ATUS_MAJOR, SEX_LABEL, VisitorError, atus_dir

# One forward BFI-44 item per Big Five scale, in scale order, then the first
# green-consumption item. Numbers are 1-based positions inside QID25.
PROBE_BFI_NUMBERS = (1, 7, 3, 4, 5)
PROBE_N_TURNS = len(PROBE_BFI_NUMBERS) + 1
PROBE_INSTRUMENT = "twin2k_probe6"

PEEDUCA_BANDS = (
    (31, 38, "less_than_high_school"),
    (39, 39, "high_school"),
    (40, 42, "some_college"),
    (43, 46, "bachelor_or_higher"),
)
RACE_LABEL = {
    1: "White",
    2: "Black",
    3: "American Indian or Alaska Native",
    4: "Asian",
    5: "Native Hawaiian or Other Pacific Islander",
}
LABOR_LABEL = {
    1: "Employed, at work",
    2: "Employed, absent",
    3: "Unemployed, on layoff",
    4: "Unemployed, looking",
    5: "Not in labor force",
}
PARTNER_LABEL = {1: "spouse", 2: "unmarried partner", 3: "none"}
METRO_LABEL = {1: "metropolitan", 2: "nonmetropolitan", 3: "not identified"}
WEEKDAY = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}


def probe_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "atus_probe" / "v0"


def select_probe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Six closed items: one forward Big Five item per scale, then one green item."""
    by_turn = {int(item["turn"]): item for item in items}
    chosen: list[dict[str, Any]] = []
    for number in PROBE_BFI_NUMBERS:
        item = by_turn.get(number)
        if item is None:
            raise InterviewError(f"BFI item {number} is missing from the instrument")
        if item.get("reverse"):
            raise InterviewError(f"BFI item {number} is reverse-keyed; pick a forward item")
        chosen.append(dict(item))
    green = [item for item in items if item.get("scale") == "green_values"]
    if not green:
        raise InterviewError("the instrument has no green-consumption item")
    chosen.append(dict(green[0]))
    if len(chosen) != PROBE_N_TURNS:
        raise InterviewError(f"expected {PROBE_N_TURNS} probe items, found {len(chosen)}")
    for turn, item in enumerate(chosen, start=1):
        item["source_turn"] = int(item["turn"])
        item["turn"] = turn
    return chosen


def diary_minutes(row: pd.Series) -> dict[str, float]:
    """This respondent's diary minutes on each major activity."""
    values: dict[str, float] = {}
    for code, label in ATUS_MAJOR.items():
        columns = [
            column
            for column in row.index
            if isinstance(column, str)
            and column.startswith("t")
            and len(column) >= 3
            and column[1:3] == code
            and column[1:].isdigit()
        ]
        if not columns:
            continue
        total = pd.to_numeric(row[columns], errors="coerce").fillna(0).sum()
        values[label] = float(total)
    return values


def sample_atus_adults(frame: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    """Stratified ATUS adults, age band × sex, weighted by TUFINLWGT."""
    if n < 1:
        raise ValueError("n must be positive")
    required = {"TUCASEID", "TEAGE", "TESEX", "TUFINLWGT"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ATUS summary is missing columns: {', '.join(sorted(missing))}")
    adults = frame.loc[pd.to_numeric(frame["TEAGE"], errors="coerce") >= 18].copy()
    adults["TEAGE"] = pd.to_numeric(adults["TEAGE"], errors="coerce")
    adults["age_band"] = adults["TEAGE"].map(lambda age: age_band(int(age)))
    adults["sex"] = pd.to_numeric(adults["TESEX"], errors="coerce").map(
        lambda value: None if pd.isna(value) else SEX_LABEL.get(int(value))
    )
    adults = adults.loc[adults["age_band"].notna() & adults["sex"].notna()]
    if adults.empty:
        raise ValueError("ATUS summary has no adults")
    adults["stratum"] = adults["age_band"].astype(str) + "|" + adults["sex"].astype(str)
    adults = adults.sort_values(["stratum", "TUCASEID"])
    weights = {
        key: float(group["TUFINLWGT"].clip(lower=0).sum())
        for key, group in adults.groupby("stratum", sort=True)
    }
    caps = {key: int(len(group)) for key, group in adults.groupby("stratum", sort=True)}
    quotas = _quotas(weights, caps, n)
    rng = np.random.default_rng(seed)
    picked: list[pd.DataFrame] = []
    for key in sorted(quotas):
        group = adults.loc[adults["stratum"] == key]
        weight = pd.to_numeric(group["TUFINLWGT"], errors="coerce").fillna(0).clip(lower=0)
        indices = _pps_indices(weight.to_numpy(dtype=float), quotas[key], rng)
        picked.append(group.iloc[indices])
    return pd.concat(picked, ignore_index=True)


def build_atus_twin2k(
    data_dir: Path,
    *,
    n: int = 50,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the ATUS persona panel, the six Twin-2K items, and the human shares."""
    from popbench.dao.twin2k import DatasetError, load_twin2k
    from popbench.simulate.persona_background import expand_persona

    if n < 1:
        raise InterviewError("n must be positive")
    summary_path = atus_dir(data_dir) / "sum" / "atussum_2023.dat"
    if not summary_path.is_file():
        raise VisitorError(
            f"ATUS activity summary is missing: {summary_path}. "
            "Run `popbench fetch --visitors-only`."
        )
    try:
        dataset = load_twin2k(data_dir, download=False)
    except DatasetError as exc:
        raise InterviewError(str(exc)) from exc

    items = select_probe_items(build_instrument(dataset.catalog))
    baseline, _pids = human_baseline(dataset.wave1_3, items)
    atus = pd.read_csv(summary_path)
    try:
        sampled = sample_atus_adults(atus, n=n, seed=seed)
    except ValueError as exc:
        raise VisitorError(str(exc)) from exc

    personas: list[dict[str, Any]] = []
    records: list[InterviewRecord] = []
    turns = _probe_turns(items)
    for index, row in sampled.iterrows():
        person_id = f"atus-{seed}-{index:04d}"
        demographics, household = _demographics(row)
        routine = diary_minutes(row)
        expanded = expand_persona(
            visitor_id=person_id,
            demographics=demographics,
            routine=routine,
            base_text="",
            seed=seed,
            diary_day=True,
        )
        text = expanded["persona_text"] + "\n\n" + _household_block(household)
        personas.append(
            {
                "id": person_id,
                "demographics": demographics,
                "household": household,
                "routine": routine,
                "background": expanded["background"],
                "text": text,
            }
        )
        records.append(
            InterviewRecord(
                id=person_id,
                region="US",
                language="en",
                source="ATUS-2023",
                instrument=PROBE_INSTRUMENT,
                persona=PersonaCard(
                    id=person_id,
                    condition="atus_life_history",
                    text=text,
                    age=int(demographics["age"]),
                    age_band=str(demographics["age_band"]),
                    sex=str(demographics["sex"]),
                    state=None,
                    education_level=str(demographics["education"]),
                    occupation=household.get("labor_force"),
                    marital_status=str(demographics["marital_status"]),
                ),
                turns=turns,
            )
        )

    out = Path(out_dir) if out_dir is not None else probe_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_path = out / f"atus_n{n}_seed{seed}.jsonl"
    write_jsonl(panel_path, [record.to_dict() for record in records])
    write_jsonl(out / f"personas_n{n}_seed{seed}.jsonl", personas)
    (out / "items.json").write_text(json.dumps(items, indent=2) + "\n")
    (out / "human_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    manifest = {
        "instrument": PROBE_INSTRUMENT,
        "n_turns": PROBE_N_TURNS,
        "n_people": n,
        "seed": seed,
        "people": "ATUS 2023 adults, stratified by age band and sex, weighted by TUFINLWGT",
        "questions": "One forward BFI-44 item per Big Five scale, plus the first green-consumption item",
        "baseline": "Twin-2K option shares on these items. No per-person gold; ATUS respondents are not Twin-2K respondents.",
        "panel": panel_path.name,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "records": records,
        "personas": personas,
        "items": items,
        "baseline": baseline,
        "panel_path": panel_path,
        "out_dir": out,
    }


def _probe_turns(items: list[dict[str, Any]]) -> tuple[InterviewTurn, ...]:
    return tuple(
        InterviewTurn(
            turn=int(item["turn"]),
            item_id=str(item["item_id"]),
            scale=str(item["scale"]),
            prompt=str(item["prompt"]),
            options=OPTIONS,
            reverse=bool(item["reverse"]),
            gold_answer=None,
            gold_code=None,
        )
        for item in items
    )


def _demographics(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    age = int(row["TEAGE"])
    sex = str(row["sex"])
    education = _education(row.get("PEEDUCA"))
    race = _race(row.get("PTDTRACE"))
    hours = _nonnegative(row.get("TEHRUSLT"))
    weekly_cents = _nonnegative(row.get("TRERNWA"))
    weekly_dollars = None if weekly_cents is None else weekly_cents / 100
    income_over_50k = bool(weekly_dollars is not None and weekly_dollars * 52 >= 50_000)
    partner = _coded(row.get("TRSPPRES"), PARTNER_LABEL)
    marital = "Married" if partner == "spouse" else "Unknown"
    demographics = {
        "age": age,
        "age_band": str(row["age_band"]),
        "sex": sex,
        "education": education,
        "marital_status": marital,
        "race": race,
        "income_over_50k": income_over_50k,
        "work_hours": hours,
    }
    children = _nonnegative(row.get("TRCHILDNUM"))
    household = {
        "labor_force": _coded(row.get("TELFS"), LABOR_LABEL),
        "spouse_or_partner": partner,
        "children": None if children is None else int(children),
        "metropolitan": _coded(row.get("GTMETSTA"), METRO_LABEL),
        "diary_day": _coded(row.get("TUDIARYDAY"), WEEKDAY),
        "weekly_earnings_dollars": weekly_dollars,
    }
    return demographics, household


def _household_block(household: dict[str, Any]) -> str:
    earnings = household.get("weekly_earnings_dollars")
    earnings_text = "unknown" if earnings is None else f"${earnings:.0f}"
    children = household.get("children")
    children_text = "unknown" if children is None else str(children)
    lines = [
        "## Household (ATUS respondent)",
        f"Labor force: {household.get('labor_force') or 'unknown'}",
        f"Spouse or partner in household: {household.get('spouse_or_partner') or 'unknown'}",
        f"Children in household: {children_text}",
        f"Weekly earnings: {earnings_text}",
        f"Metropolitan status: {household.get('metropolitan') or 'unknown'}",
        f"Diary day: {household.get('diary_day') or 'unknown'}",
    ]
    return "\n".join(lines)


def _education(value: Any) -> str:
    code = _nonnegative(value)
    if code is None:
        return "unknown"
    number = int(code)
    for low, high, name in PEEDUCA_BANDS:
        if low <= number <= high:
            return name
    return "unknown"


def _race(value: Any) -> str:
    code = _nonnegative(value)
    if code is None:
        return "Unknown"
    number = int(code)
    if number in RACE_LABEL:
        return RACE_LABEL[number]
    if number >= 6:
        return "Two or more races"
    return "Unknown"


def _coded(value: Any, labels: dict[int, str]) -> str | None:
    code = _nonnegative(value)
    if code is None:
        return None
    return labels.get(int(code))


def _nonnegative(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _quotas(weights: dict[str, float], caps: dict[str, int], n: int) -> dict[str, int]:
    keys = sorted(key for key, weight in weights.items() if weight > 0 and caps.get(key, 0) > 0)
    if sum(caps.get(key, 0) for key in keys) < n:
        raise ValueError("not enough ATUS adults to fill the sample")
    total = sum(weights[key] for key in keys)
    raw = {key: n * weights[key] / total for key in keys}
    base = {key: min(caps[key], int(math.floor(raw[key]))) for key in keys}
    leftover = n - sum(base.values())
    order = sorted(keys, key=lambda key: (raw[key] - math.floor(raw[key]), weights[key]), reverse=True)
    while leftover:
        progressed = False
        for key in order:
            if leftover == 0:
                break
            if base[key] < caps[key]:
                base[key] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            raise ValueError("not enough ATUS adults to fill the sample")
    return {key: count for key, count in base.items() if count}


def _pps_indices(weights: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    remaining = np.arange(len(weights))
    current = np.array(weights, dtype=float, copy=True)
    current[current < 0] = 0
    if float(current.sum()) <= 0:
        current[:] = 1
    chosen: list[int] = []
    for _ in range(k):
        pool = current[remaining]
        probs = pool / pool.sum()
        pick = int(rng.choice(len(remaining), p=probs))
        chosen.append(int(remaining[pick]))
        remaining = np.delete(remaining, pick)
    return chosen
