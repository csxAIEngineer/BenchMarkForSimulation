"""Read the public Twin-2K and BDM files into rows and pricing outcomes.

Pricing stimuli stay on the outcome records. They are not returned as card
fields.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from popbench.pricing_basic_info.cards import normalize_cell
from popbench.pricing_basic_info.spec import (
    BDM_BASIC_COLUMNS,
    BDM_DESIGN_COLUMNS,
    BDM_OUTCOME_COLUMNS,
    BDM_PUBLISHED_N,
    ID_COLUMN,
    TWIN_BASIC_COLUMNS,
    TWIN_PRICING_IDS,
    TWIN_PUBLISHED_N,
)
from popbench.datasets.bdm_lettuce.loader import PUBLISHED_N


class SourceError(ValueError):
    """A published file does not match the pinned pricing extract."""


def load_twin_tables(
    demographic_csv: Path,
    wave4_csv: Path,
    catalog_path: Path,
    wave_chunks: Iterable[Path],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, object]]], dict[str, str], dict[str, int]]:
    demographics = _read_csv(demographic_csv)
    wave4 = _read_csv(wave4_csv)
    _expect_rows(demographics, TWIN_PUBLISHED_N, demographic_csv.name)
    _expect_rows(wave4, TWIN_PUBLISHED_N, wave4_csv.name)
    stems = demographic_stems(json.loads(catalog_path.read_text(encoding="utf-8")))
    wave4_by_id = {_pid(row): row for row in wave4}
    demo_by_id = {_pid(row): row for row in demographics}
    if set(wave4_by_id) != set(demo_by_id):
        raise SourceError("wave 4 pids do not match the demographic file")
    json_outcomes = _pricing_from_wave_json(wave_chunks)
    if set(json_outcomes) != set(demo_by_id):
        raise SourceError("wave JSON pids do not match the demographic file")
    equal = 0
    differ = 0
    for subject_id, row in demo_by_id.items():
        earlier = row
        later = wave4_by_id[subject_id]
        for question_id in TWIN_PRICING_IDS:
            if (earlier.get(question_id) or "").strip() == (later.get(question_id) or "").strip():
                equal += 1
            else:
                differ += 1
        items = json_outcomes[subject_id]
        for item in items:
            question_id = str(item["question_id"])
            csv_answer = (later.get(question_id) or "").strip()
            if csv_answer != item["answer"]:
                raise SourceError(
                    f"pid {subject_id} {question_id} wave4 CSV answer does not match wave4 JSON"
                )
    audit = {
        "wave4_vs_wave1_3_qid9_cells_differ": differ,
        "wave4_vs_wave1_3_qid9_cells_equal": equal,
    }
    rows = [demo_by_id[subject_id] for subject_id in sorted(demo_by_id, key=_pid_sort)]
    return rows, json_outcomes, stems, audit


def demographic_stems(catalog: object) -> dict[str, str]:
    if not isinstance(catalog, list):
        raise SourceError("question catalog must be a list")
    stems: dict[str, str] = {}
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        question_id = str(entry.get("QuestionID") or "")
        if entry.get("BlockName") != "Demographics" or question_id not in TWIN_BASIC_COLUMNS:
            continue
        if question_id in stems:
            raise SourceError(f"duplicate demographic stem for {question_id}")
        stems[question_id] = str(entry.get("QuestionText") or "")
    missing = [column for column in TWIN_BASIC_COLUMNS if column not in stems]
    if missing:
        raise SourceError(f"question catalog is missing demographic stems: {missing}")
    return stems


def load_bdm_table(csv_path: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, object]]]]:
    if PUBLISHED_N != BDM_PUBLISHED_N:
        raise SourceError("BDM published N drifted from the loader")
    rows = _read_csv(csv_path)
    _expect_rows(rows, BDM_PUBLISHED_N, csv_path.name)
    fieldnames = set(rows[0])
    required = (ID_COLUMN, *BDM_BASIC_COLUMNS, *BDM_OUTCOME_COLUMNS, *BDM_DESIGN_COLUMNS)
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise SourceError(f"BDM CSV is missing columns: {missing}")
    outcomes: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        subject_id = _pid(row)
        items: list[dict[str, object]] = []
        for column in BDM_OUTCOME_COLUMNS:
            item: dict[str, object] = {
                "answer": (row.get(column) or "").strip(),
                "question_id": column,
                "question_text": "",
                "source_column": column,
            }
            if column == "conclusion_purchased_product":
                item["stimulus"] = {
                    "conclusion_generated_price": (row.get("conclusion_generated_price") or "").strip(),
                    "conclusion_selected_product": (row.get("conclusion_selected_product") or "").strip(),
                }
            items.append(item)
        outcomes[subject_id] = items
    ordered = sorted(rows, key=lambda row: _pid_sort(_pid(row)))
    return ordered, outcomes


def _pricing_from_wave_json(chunks: Iterable[Path]) -> dict[str, list[dict[str, object]]]:
    import pyarrow.parquet as pq

    found: dict[str, list[dict[str, object]]] = {}
    paths = list(chunks)
    if len(paths) != 7:
        raise SourceError(f"expected 7 Twin-2K wave chunks, found {len(paths)}")
    for path in paths:
        print(f"reading {path.name}", flush=True)
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if "pid" not in columns or "wave4_Q_wave4_A" not in columns:
            raise SourceError(f"{path.name} is missing pid or wave4_Q_wave4_A")
        for batch in parquet.iter_batches(batch_size=16, columns=["pid", "wave4_Q_wave4_A"]):
            payload = batch.to_pydict()
            for pid, raw in zip(payload["pid"], payload["wave4_Q_wave4_A"], strict=True):
                subject_id = str(pid).strip()
                if subject_id in found:
                    raise SourceError(f"duplicate pid in wave JSON: {subject_id}")
                found[subject_id] = _qid9_items(raw, subject_id)
    return found


def _qid9_items(raw: object, subject_id: str) -> list[dict[str, object]]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    questions = [node for node in _walk_questions(data) if str(node.get("QuestionID") or "").startswith("QID9_")]
    by_id: dict[str, dict[str, object]] = {}
    for node in questions:
        question_id = str(node.get("QuestionID") or "")
        if question_id in by_id:
            raise SourceError(f"pid {subject_id} repeats {question_id}")
        answers = node.get("Answers")
        if not isinstance(answers, dict):
            raise SourceError(f"pid {subject_id} {question_id} has no Answers object")
        selected = answers.get("SelectedText")
        if not isinstance(selected, str):
            raise SourceError(f"pid {subject_id} {question_id} SelectedText is not a string")
        text = node.get("QuestionText")
        if not isinstance(text, str) or text.strip() == "":
            raise SourceError(f"pid {subject_id} {question_id} has no question text")
        position = answers.get("SelectedByPosition")
        if isinstance(position, bool) or not isinstance(position, int):
            raise SourceError(f"pid {subject_id} {question_id} SelectedByPosition is not an integer")
        by_id[question_id] = {
            "answer": selected,
            "answer_position": position,
            "question_id": question_id,
            "question_text": text,
            "source_column": "wave4_Q_wave4_A",
        }
    expected = list(TWIN_PRICING_IDS)
    if set(by_id) != set(expected):
        raise SourceError(f"pid {subject_id} does not have QID9_1 through QID9_40")
    ordered = [by_id[question_id] for question_id in expected]
    for item in ordered:
        value, status = normalize_cell(item["answer"])
        if status != "present" or value != item["answer"]:
            raise SourceError(f"pid {subject_id} {item['question_id']} answer is blank or padded")
    return ordered


def _walk_questions(node: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "QuestionID" in value and "QuestionText" in value:
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found


def _read_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or ID_COLUMN not in reader.fieldnames:
        raise SourceError(f"{path.name} is missing pid")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in reader:
        row = {key: "" if value is None else str(value) for key, value in raw.items()}
        subject_id = row[ID_COLUMN].strip()
        if not subject_id:
            raise SourceError(f"{path.name} has a blank pid")
        if subject_id in seen:
            raise SourceError(f"{path.name} repeats pid {subject_id}")
        seen.add(subject_id)
        row[ID_COLUMN] = subject_id
        rows.append(row)
    return rows


def _expect_rows(rows: list[dict[str, str]], expected: int, name: str) -> None:
    if len(rows) != expected:
        raise SourceError(f"{name} has {len(rows)} rows, expected {expected}")


def _pid(row: Mapping[str, str]) -> str:
    return str(row[ID_COLUMN]).strip()


def _pid_sort(subject_id: str) -> tuple[int, int | str]:
    if subject_id.isdigit():
        return (0, int(subject_id))
    return (1, subject_id)

