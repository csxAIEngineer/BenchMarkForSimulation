"""Bind one fixed Twin-2K slice for economy, experiments, and pricing.

The dataset name is ``twin2K-economy-bechmark``. ``QID118`` through ``QID231``
is the closed tail of the economic-preference block after the trust-game
sender: receiver items ``QID118``–``QID122``, then the dictator game
``QID231``. Thought listings ``QID271``, ``QID272``, and ``QID275`` sit
between those numbers and stay out, as do the other ultimatum and mental-
accounting items that were not named.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from popbench.dao.interview import InterviewError
from popbench.dao.io import write_jsonl
from popbench.dao.survey import (
    CATEGORY_LABELS,
    build_closed_items,
    family_for,
    human_baseline,
    instrument_for,
    is_open_entry,
    items_by_family,
)
from popbench.dao.twin2k import twin2k_dir

DATASET_NAME = "twin2K-economy-bechmark"

# One ultimatum-receiver row, one mental-accounting row, the full QID251
# risk ladder, the trust-game sender, the trust-game receiver, and the dictator game.
ECONOMIC_COLUMNS: tuple[str, ...] = (
    "QID229",
    "QID149",
    *(f"QID251_{index}" for index in range(1, 15)),
    "QID117",
    "QID118",
    "QID119",
    "QID120",
    "QID121",
    "QID122",
    "QID231",
)

# Selected false-consensus rows, plus one item from each named experiment.
EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "QID287_1",
    "QID287_3",
    "QID287_5",
    "QID287_10",
    "QID287_12",
    "QID158",
    "QID177",
    "QID184",
    "QID195",
    "QID178",
    "QID159_1",
)

PRICING_COLUMNS: tuple[str, ...] = tuple(f"QID9_{index}" for index in range(1, 41))

ECONOMY_BENCHMARK_COLUMNS: tuple[str, ...] = (
    ECONOMIC_COLUMNS + EXPERIMENT_COLUMNS + PRICING_COLUMNS
)

_BENCHMARK_FAMILIES = ("economic", "experiments", "pricing")
_WAVE4_FAMILIES = frozenset({"experiments", "pricing"})


def economy_benchmark_dir(data_dir: Path) -> Path:
    return twin2k_dir(data_dir) / DATASET_NAME


def bind_economy_benchmark(
    catalog: list[dict[str, Any]],
    columns: tuple[str, ...] = ECONOMY_BENCHMARK_COLUMNS,
) -> list[dict[str, Any]]:
    """Copy catalog entries trimmed to ``columns``, in that column order."""
    if len(columns) != len(set(columns)):
        raise InterviewError("economy benchmark selects a column more than once")
    owners = _owners(catalog)
    grouped: list[dict[str, Any]] = []
    index_by_entry: dict[int, dict[str, Any]] = {}
    for column in columns:
        entry = _entry_for(owners, column)
        bucket = index_by_entry.get(id(entry))
        if bucket is None:
            bucket = {"entry": entry, "columns": []}
            index_by_entry[id(entry)] = bucket
            grouped.append(bucket)
        bucket["columns"].append(column)
    return [_slice_entry(bucket["entry"], bucket["columns"]) for bucket in grouped]


def write_economy_benchmark(
    data: Any,
    dest: Path,
    columns: tuple[str, ...] = ECONOMY_BENCHMARK_COLUMNS,
) -> dict[str, Any]:
    """Write the fixed catalog and the response columns those items own."""
    entries = bind_economy_benchmark(data.catalog, columns)
    if any(is_open_entry(entry) for entry in entries):
        raise InterviewError("economy benchmark includes an open question")
    records = _column_records(entries)
    flat = [record["column"] for record in records]
    if tuple(flat) != tuple(columns):
        raise InterviewError("bound columns do not match the requested order")
    _same_columns(data.wave1_3, data.wave1_3_labels, "wave1_3 labels")
    _same_columns(data.wave4, data.wave4_labels, "wave4 labels")
    _require_columns(data.wave1_3, flat, "wave1_3")
    wave4_columns = [record["column"] for record in records if record["family"] in _WAVE4_FAMILIES]
    _require_columns(data.wave4, wave4_columns, "wave4")
    _reject_repeated_economic(data.wave4, records)

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "question_catalog.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
    )
    _write_slice(data.wave1_3, flat, dest / "wave1_3_response.csv")
    _write_slice(data.wave1_3_labels, flat, dest / "wave1_3_response_label.csv")
    _write_slice(data.wave4, wave4_columns, dest / "wave4_response.csv")
    _write_slice(data.wave4_labels, wave4_columns, dest / "wave4_response_label.csv")
    summary = _summary(data, entries, records, wave4_columns)
    (dest / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def _owners(catalog: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], str]]:
    owners: dict[str, tuple[dict[str, Any], str]] = {}
    for entry in catalog:
        family = family_for(str(entry.get("BlockName") or ""))
        for column in entry.get("csv_columns") or []:
            if not isinstance(column, str) or not column:
                raise InterviewError(f"{entry.get('QuestionID')} has an empty csv column")
            if column in owners:
                raise InterviewError(f"{column} belongs to more than one catalog entry")
            owners[column] = (entry, family)
    return owners


def _entry_for(
    owners: dict[str, tuple[dict[str, Any], str]],
    column: str,
) -> dict[str, Any]:
    found = owners.get(column)
    if found is None:
        raise InterviewError(f"{column} is not in the Twin-2K catalog")
    entry, family = found
    if family not in _BENCHMARK_FAMILIES:
        raise InterviewError(
            f"{column} is {family}; the economy benchmark only keeps "
            "economic, experiments, and pricing"
        )
    return entry


def _slice_entry(entry: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    owned = [str(column) for column in entry.get("csv_columns") or []]
    positions = [_position(owned, column, entry) for column in columns]
    sliced: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "csv_columns":
            sliced[key] = list(columns)
        elif key in {"Rows", "RowsID"}:
            sliced[key] = _take(value, positions, key, owned, entry)
        else:
            sliced[key] = _copy_value(value)
    return sliced


def _position(owned: list[str], column: str, entry: dict[str, Any]) -> int:
    try:
        return owned.index(column)
    except ValueError as exc:
        raise InterviewError(f"{column} is not a column of {entry.get('QuestionID')}") from exc


def _take(
    values: Any,
    positions: list[int],
    key: str,
    owned: list[str],
    entry: dict[str, Any],
) -> list[Any]:
    if not isinstance(values, list) or len(values) != len(owned):
        raise InterviewError(f"{entry.get('QuestionID')} {key} and csv columns differ")
    return [values[index] for index in positions]


def _copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


def _column_records(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for entry in entries:
        family = family_for(str(entry.get("BlockName") or ""))
        instrument = instrument_for(entry)
        block = str(entry.get("BlockName") or "").strip()
        question_id = str(entry.get("QuestionID") or "")
        for column in entry.get("csv_columns") or []:
            records.append(
                {
                    "column": column,
                    "question_id": question_id,
                    "family": family,
                    "block": block,
                    "instrument": instrument,
                }
            )
    return records


def _summary(
    data: Any,
    entries: list[dict[str, Any]],
    records: list[dict[str, str]],
    wave4_columns: list[str],
) -> dict[str, Any]:
    counts = Counter(record["family"] for record in records)
    return {
        "name": DATASET_NAME,
        "revision": data.revision,
        "n_respondents": int(len(data.wave1_3)),
        "n_questions": len(entries),
        "n_columns": len(records),
        "families": {family: counts.get(family, 0) for family in _BENCHMARK_FAMILIES},
        "wave4_columns": len(wave4_columns),
        "columns": records,
    }


def _same_columns(numeric: pd.DataFrame, labels: pd.DataFrame, name: str) -> None:
    if list(numeric.columns) != list(labels.columns):
        raise InterviewError(f"{name} columns differ from the numeric table")


def _require_columns(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise InterviewError(f"{table} is missing " + ", ".join(missing[:8]))


def _reject_repeated_economic(frame: pd.DataFrame, records: list[dict[str, str]]) -> None:
    present = set(frame.columns)
    extra = [
        record["column"]
        for record in records
        if record["family"] not in _WAVE4_FAMILIES and record["column"] in present
    ]
    if extra:
        raise InterviewError(
            "wave4 repeats a column outside experiments and pricing: " + ", ".join(extra[:8])
        )


def _write_slice(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    frame.loc[:, ["pid", *columns]].to_csv(path, index=False)


def load_economy_benchmark(data_dir: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Read the bound catalog and the wave 1–3 numeric responses."""
    folder = economy_benchmark_dir(data_dir)
    catalog_path = folder / "question_catalog.json"
    wave_path = folder / "wave1_3_response.csv"
    if not catalog_path.is_file() or not wave_path.is_file():
        raise InterviewError(
            f"missing {DATASET_NAME} under {folder}. Run `popbench bind-economy-benchmark`."
        )
    catalog = json.loads(catalog_path.read_text())
    if not isinstance(catalog, list):
        raise InterviewError(f"question catalog must be a list: {catalog_path}")
    return catalog, pd.read_csv(wave_path)


def economy_benchmark_items(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Closed items in the fixed column order."""
    items = build_closed_items(catalog, minimum=1)
    found = tuple(item["item_id"] for item in items)
    if found != ECONOMY_BENCHMARK_COLUMNS:
        raise InterviewError("benchmark catalog did not expand to the fixed columns")
    return items


def build_economy_benchmark_panels(
    data_dir: Path,
    *,
    n: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """ATUS cards answering the fixed slice, one interview per family.

    Later families do not see earlier answers. The human baseline is this
    dataset's wave 1–3 table.
    """
    from popbench.dao.atus_twin2k import build_atus_personas

    catalog, wave = load_economy_benchmark(data_dir)
    grouped = items_by_family(economy_benchmark_items(catalog))
    personas = build_atus_personas(data_dir, n=n, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "personas.jsonl", personas)
    panels = []
    for family in _BENCHMARK_FAMILIES:
        items = grouped[family]
        if not items:
            raise InterviewError(f"{family} has no closed items in {DATASET_NAME}")
        baseline = human_baseline(wave, items)
        records = _benchmark_records(personas, items)
        folder = out_dir / "families" / family
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "items.json").write_text(json.dumps(items, ensure_ascii=False) + "\n")
        (folder / "human_baseline.json").write_text(json.dumps(baseline) + "\n")
        manifest = {
            "dataset": DATASET_NAME,
            "family": family,
            "label": CATEGORY_LABELS[family],
            "n_items": len(items),
            "n_people": len(records),
            "seed": seed,
            "sampling": "age_band_x_sex",
            "baseline": f"{DATASET_NAME} wave1_3 shares",
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        panels.append(
            {
                "family": family,
                "label": CATEGORY_LABELS[family],
                "items": items,
                "baseline": baseline,
                "records": records,
                "dir": folder,
            }
        )
    return {
        "personas": personas,
        "panels": panels,
        "n": len(personas),
        "seed": seed,
        "out_dir": out_dir,
    }


def responses_cover(path: Path, records: list[dict[str, Any]]) -> bool:
    """True when every persona already has one answer for every turn."""
    if not path.is_file() or not records:
        return False
    from popbench.dao.io import read_jsonl

    rows = read_jsonl(path)
    if [row.get("id") for row in rows] != [record["id"] for record in records]:
        return False
    item_ids = [turn["item_id"] for turn in records[0]["turns"]]
    for row in rows:
        answers = row.get("answers")
        if not isinstance(answers, list):
            return False
        if [answer.get("item_id") for answer in answers] != item_ids:
            return False
    return True


def _benchmark_records(
    personas: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    turns = [_benchmark_turn(item) for item in items]
    records = []
    for person in personas:
        demographics = person.get("demographics") or {}
        records.append(
            {
                "id": person["id"],
                "persona": {
                    "id": person["id"],
                    "age_band": demographics.get("age_band"),
                    "sex": demographics.get("sex"),
                    "text": person["text"],
                },
                "turns": turns,
            }
        )
    return records


def _benchmark_turn(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": item["turn"],
        "item_id": item["item_id"],
        "kind": item["kind"],
        "family": item["family"],
        "block": item["block"],
        "instrument": item["instrument"],
        "prompt": item["prompt"],
        "options": list(item["options"]),
    }
