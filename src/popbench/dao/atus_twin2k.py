"""ATUS adults answering Twin-2K items from written persona cards.

People are drawn from the ATUS 2023 activity summary. The persona text is the
card already written for that TUCASEID in ``adults/personas.jsonl``. The
six-item probe and the seven paper categories both use this draw. Scoring uses
Twin-2K response shares. ATUS respondents are a different sample, so the panel
carries no per-person gold answer.
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
from popbench.dao.io import read_jsonl, write_jsonl
from popbench.dao.personas import AGE_BANDS, age_band

# Same four bands as Twin-2K QID13 and the ATUS age split.
AGE_BAND_ORDER = tuple(name for _low, _high, name in AGE_BANDS)
from popbench.dao.schema import InterviewRecord, InterviewTurn, PersonaCard
from popbench.dao.visitors import SEX_LABEL, VisitorError, atus_dir, visitors_dir

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
_CARD_COMMAND = "popbench simulate-population-personas"


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


def adult_persona_cards_path(data_dir: Path) -> Path:
    """Written life-history cards, one row per ATUS adult."""
    return visitors_dir(data_dir) / "by_population" / "adults" / "personas.jsonl"


def sample_atus_adults(frame: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    """Stratified ATUS adults, age band × sex, weighted by TUFINLWGT."""
    if n < 1:
        raise ValueError("n must be positive")
    adults = _labeled_adults(frame)
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


def sample_atus_age_bands(frame: pd.DataFrame, *, n_per_band: int, seed: int) -> pd.DataFrame:
    """Draw ``n_per_band`` adults from each age band, weighted by TUFINLWGT.

    Bands are 18–29, 30–49, 50–64, and 65+. Sampling is without replacement
    inside a band. The same seed returns the same people. Sex is not a stratum.
    """
    if n_per_band < 1:
        raise ValueError("n_per_band must be positive")
    adults = _labeled_adults(frame)
    adults = adults.sort_values(["age_band", "TUCASEID"])
    rng = np.random.default_rng(seed)
    picked: list[pd.DataFrame] = []
    for band in AGE_BAND_ORDER:
        group = adults.loc[adults["age_band"] == band]
        if len(group) < n_per_band:
            raise ValueError(
                f"age band {band} has {len(group)} adults, fewer than {n_per_band}"
            )
        weight = pd.to_numeric(group["TUFINLWGT"], errors="coerce").fillna(0).clip(lower=0)
        indices = _pps_indices(weight.to_numpy(dtype=float), n_per_band, rng)
        picked.append(group.iloc[indices])
    return pd.concat(picked, ignore_index=True)


def _labeled_adults(frame: pd.DataFrame) -> pd.DataFrame:
    """ATUS respondents age 18+ with an age band and a sex label."""
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
    return adults


def load_all_adult_personas(data_dir: Path) -> list[dict[str, Any]]:
    """Every written ATUS adult card, in file order. Does not sample."""
    path = adult_persona_cards_path(data_dir)
    if not path.is_file():
        raise VisitorError(
            f"ATUS adult persona cards are missing: {path}. Run `{_CARD_COMMAND}`."
        )
    personas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in read_jsonl(path):
        person_id = str(card.get("id") or "").strip()
        if not person_id:
            person_id = f"visitor-atus-{_case_key(card.get('TUCASEID'))}"
        if person_id in seen:
            raise VisitorError(f"duplicate persona id {person_id} in {path}")
        seen.add(person_id)
        text = card.get("persona_text")
        if not isinstance(text, str) or not text.strip():
            raise VisitorError(f"{person_id} has no persona card text in {path}")
        demo = card.get("demographics") if isinstance(card.get("demographics"), dict) else {}
        personas.append(
            {
                "id": person_id,
                "TUCASEID": card.get("TUCASEID"),
                "demographics": {
                    "age": demo.get("age"),
                    "age_band": demo.get("age_band"),
                    "sex": demo.get("sex"),
                    "education": demo.get("education_level") or demo.get("education"),
                    "marital_status": demo.get("partner") or demo.get("marital_status"),
                    "employment": demo.get("employment"),
                },
                "text": text,
            }
        )
    if not personas:
        raise VisitorError(f"no persona cards in {path}")
    return personas


def build_atus_personas(
    data_dir: Path,
    *,
    n: int = 50,
    seed: int = 0,
    n_per_age_band: int | None = None,
) -> list[dict[str, Any]]:
    """Sample ATUS adults and attach the written persona card for each TUCASEID.

    The same ``n`` and ``seed`` always return the same people. With
    ``n_per_age_band``, each age band contributes that many adults and ``n``
    is ignored. Card text is ``persona_text`` from ``adults/personas.jsonl``.
    """
    if n_per_age_band is None and n < 1:
        raise InterviewError("n must be positive")
    if n_per_age_band is not None and n_per_age_band < 1:
        raise InterviewError("n_per_age_band must be positive")
    summary_path = atus_dir(data_dir) / "sum" / "atussum_2023.dat"
    if not summary_path.is_file():
        raise VisitorError(
            f"ATUS activity summary is missing: {summary_path}. "
            "Run `popbench fetch --visitors-only`."
        )
    cards_path = adult_persona_cards_path(data_dir)
    cards = _load_adult_cards(cards_path)
    atus = pd.read_csv(summary_path)
    try:
        if n_per_age_band is None:
            sampled = sample_atus_adults(atus, n=n, seed=seed)
        else:
            sampled = sample_atus_age_bands(atus, n_per_band=n_per_age_band, seed=seed)
    except ValueError as exc:
        raise VisitorError(str(exc)) from exc
    personas: list[dict[str, Any]] = []
    for _, row in sampled.iterrows():
        case = _case_key(row["TUCASEID"])
        card = cards.get(case)
        text = None if card is None else card.get("persona_text")
        if card is None or not isinstance(text, str) or not text.strip():
            raise VisitorError(
                f"ATUS adult {case} has no persona card in {cards_path}. "
                f"Run `{_CARD_COMMAND}`."
            )
        person_id = f"visitor-atus-{case}"
        demographics = _card_demographics(card, row)
        personas.append(
            {
                "id": person_id,
                "TUCASEID": int(case) if case.isdigit() else case,
                "demographics": demographics,
                "household": {"labor_force": demographics.get("employment")},
                "text": text,
            }
        )
    return personas


def build_atus_twin2k(
    data_dir: Path,
    *,
    n: int = 50,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the ATUS persona panel, the six Twin-2K items, and the human shares."""
    from popbench.dao.twin2k import DatasetError, load_twin2k

    try:
        dataset = load_twin2k(data_dir, download=False)
    except DatasetError as exc:
        raise InterviewError(str(exc)) from exc

    items = select_probe_items(build_instrument(dataset.catalog))
    baseline, _pids = human_baseline(dataset.wave1_3, items)
    personas = build_atus_personas(data_dir, n=n, seed=seed)
    records: list[InterviewRecord] = []
    turns = _probe_turns(items)
    for person in personas:
        demographics = person["demographics"]
        household = person.get("household") or {}
        age = demographics.get("age")
        records.append(
            InterviewRecord(
                id=person["id"],
                region="US",
                language="en",
                source="ATUS-2023",
                instrument=PROBE_INSTRUMENT,
                persona=PersonaCard(
                    id=person["id"],
                    condition="atus_life_history",
                    text=person["text"],
                    age=None if age is None else int(age),
                    age_band=demographics.get("age_band"),
                    sex=demographics.get("sex"),
                    state=None,
                    education_level=demographics.get("education"),
                    occupation=household.get("labor_force"),
                    marital_status=demographics.get("marital_status"),
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
        "people": (
            "ATUS 2023 adults, stratified by age band and sex, weighted by TUFINLWGT. "
            "Persona text is the written adult card for that TUCASEID."
        ),
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


def _load_adult_cards(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise VisitorError(
            f"ATUS adult persona cards are missing: {path}. Run `{_CARD_COMMAND}`."
        )
    cards: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        case = row.get("TUCASEID")
        if case is None:
            person_id = str(row.get("id") or "")
            prefix = "visitor-atus-"
            if person_id.startswith(prefix):
                case = person_id[len(prefix) :]
        if case is None:
            continue
        cards[_case_key(case)] = row
    return cards


def _card_demographics(card: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    demo = card.get("demographics") if isinstance(card.get("demographics"), dict) else {}
    persona = card.get("persona") if isinstance(card.get("persona"), dict) else {}
    age = demo.get("age", persona.get("age"))
    if age is None:
        age = int(row["TEAGE"])
    else:
        age = int(age)
    education = demo.get("education_level") or persona.get("education_level") or demo.get("education")
    marital = persona.get("marital_status") or demo.get("marital_status") or demo.get("partner")
    employment = demo.get("employment") or persona.get("occupation")
    return {
        "age": age,
        "age_band": str(demo.get("age_band") or persona.get("age_band") or row["age_band"]),
        "sex": str(demo.get("sex") or persona.get("sex") or row["sex"]),
        "education": None if education in (None, "") else str(education),
        "marital_status": None if marital in (None, "") else str(marital),
        "employment": None if employment in (None, "") else str(employment),
    }


def _case_key(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        return text
    try:
        if pd.isna(value):
            raise VisitorError("ATUS row is missing TUCASEID")
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if not value.is_integer():
            raise VisitorError(f"ATUS TUCASEID is not an integer: {value}")
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


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
