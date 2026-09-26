"""Build the closed-ended Twin-2K item list and the Nemotron panel that answers it.

The paper's "~500 questions" counts some multi-row tasks as one question and
includes open-ended items. This module expands each scorable closed-ended
answer (matrix row, single choice, number, slider) and skips free text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from popbench.dao.interview import InterviewError
from popbench.dao.io import read_jsonl, write_jsonl
from popbench.dao.personas import sample_usa_adults


def survey_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "interview" / "full"


def family_for(block: str) -> str:
    name = block.strip()
    if name == "Demographics":
        return "demographics"
    if name == "Personality":
        return "personality"
    if name == "Cognitive tests":
        return "cognitive"
    if name.startswith("Economic"):
        return "economic"
    if name.startswith("Product"):
        return "pricing"
    if name == "Non-experimental heuristics and biases":
        return "heuristics"
    return "experiments"


def is_open_entry(entry: dict[str, Any]) -> bool:
    if entry.get("is_descriptive") or entry.get("QuestionType") == "DB":
        return True
    settings = entry.get("Settings") or {}
    selector = settings.get("Selector")
    content = settings.get("ContentType")
    if entry.get("QuestionType") == "TE" and selector in {"ML", "FORM"}:
        return True
    if entry.get("QuestionType") == "TE" and selector == "SL" and content != "ValidNumber":
        return True
    return False


ASK_N = 500


def build_closed_items(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One scorable answer per row, in catalog order."""
    items: list[dict[str, Any]] = []
    for entry in catalog:
        if is_open_entry(entry):
            continue
        items.extend(_expand(entry))
    if len(items) < 400:
        raise InterviewError(f"expected several hundred closed items, found {len(items)}")
    for turn, item in enumerate(items, start=1):
        item["turn"] = turn
    return items


def cap_items(items: list[dict[str, Any]], n: int = ASK_N) -> list[dict[str, Any]]:
    """Keep the first n items, and finish the matrix question that contains item n."""
    if len(items) <= n:
        return items
    question = _question_key(items[n - 1]["item_id"])
    end = n
    while end < len(items) and _question_key(items[end]["item_id"]) == question:
        end += 1
    kept = items[:end]
    for turn, item in enumerate(kept, start=1):
        item["turn"] = turn
    return kept


def _question_key(item_id: str) -> str:
    head, sep, tail = item_id.rpartition("_")
    if sep and tail.isdigit():
        return head
    return item_id


def _expand(entry: dict[str, Any]) -> list[dict[str, Any]]:
    question_type = entry.get("QuestionType")
    selector = (entry.get("Settings") or {}).get("Selector")
    if question_type == "Matrix":
        return _matrix_items(entry)
    if question_type == "Slider":
        return _slider_items(entry)
    if selector in {"MAVR", "MAHR"}:
        return [_multi_as_choice(entry)]
    if question_type == "TE":
        return [_numeric_item(entry, entry["csv_columns"][0], _stem(entry))]
    return [_choice_item(entry, entry["csv_columns"][0], _stem(entry), list(entry.get("Options") or []))]


def _matrix_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = entry.get("Rows") or []
    columns = entry.get("csv_columns") or []
    if len(rows) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} rows and csv columns differ")
    selector = (entry.get("Settings") or {}).get("Selector")
    scale = list(entry.get("Columns") or [])
    items = []
    stem = _stem(entry)
    for statement, column in zip(rows, columns):
        if selector == "Bipolar":
            left, right = statement.split(":", 1)
            options = [left.strip(), right.strip()]
            prompt = f"{stem}\n\nChoose one.\nLeft: {options[0]}\nRight: {options[1]}"
        else:
            options = scale
            prompt = f"{stem}\n\n{statement}"
        items.append(_choice_item(entry, column, prompt, options))
    return items


def _slider_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    columns = entry.get("csv_columns") or []
    statements = entry.get("Statements") or []
    if statements and len(statements) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} slider statements and columns differ")
    bounds = entry.get("Range") or {}
    low = bounds.get("Min", 0)
    high = bounds.get("Max", 100)
    stem = _stem(entry)
    items = []
    for index, column in enumerate(columns):
        statement = statements[index].strip() if statements else ""
        extra = f"\n\n{statement}" if statement else ""
        prompt = f"{stem}{extra}\n\nReply with a number from {low} to {high}."
        items.append(_numeric_item(entry, column, prompt))
    return items


def _multi_as_choice(entry: dict[str, Any]) -> dict[str, Any]:
    options = list(entry.get("Options") or [])
    columns = list(entry.get("csv_columns") or [])
    if len(options) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} options and columns differ")
    stem = _stem(entry) or "Which statement fits you best?"
    item = _choice_item(entry, columns[0], stem, options)
    item["item_id"] = entry["QuestionID"]
    item["encoding"] = "first_selected"
    item["columns"] = columns
    return item


def _choice_item(
    entry: dict[str, Any],
    column: str,
    prompt: str,
    options: list[str],
) -> dict[str, Any]:
    if len(options) < 2:
        raise InterviewError(f"{column} needs at least two options")
    return {
        "item_id": column,
        "column": column,
        "columns": [column],
        "encoding": "single",
        "kind": "choice",
        "family": family_for(entry.get("BlockName") or ""),
        "block": (entry.get("BlockName") or "").strip(),
        "prompt": prompt.strip(),
        "options": options,
    }


def _numeric_item(entry: dict[str, Any], column: str, prompt: str) -> dict[str, Any]:
    return {
        "item_id": column,
        "column": column,
        "columns": [column],
        "encoding": "number",
        "kind": "numeric",
        "family": family_for(entry.get("BlockName") or ""),
        "block": (entry.get("BlockName") or "").strip(),
        "prompt": prompt.strip(),
        "options": [],
    }


def _stem(entry: dict[str, Any]) -> str:
    text = (entry.get("QuestionText") or "").strip()
    if not text:
        return ""
    head = text.split("\n\n")[0].strip()
    if len(head) > 900:
        head = head[:900].rsplit(" ", 1)[0]
    return head


def human_baseline(numeric: pd.DataFrame, items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in items:
        if item["kind"] == "choice":
            codes = _choice_codes(numeric, item)
            observed = codes.dropna().astype(int)
            n_options = len(item["options"])
            shares = []
            for code in range(1, n_options + 1):
                shares.append(float((observed == code).mean()) if len(observed) else 0.0)
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "choice",
                    "family": item["family"],
                    "n": int(len(observed)),
                    "shares": shares,
                    "mean_code": float(observed.mean()) if len(observed) else None,
                }
            )
        else:
            values = pd.to_numeric(numeric[item["column"]], errors="coerce").dropna()
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "numeric",
                    "family": item["family"],
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else None,
                }
            )
    return {"n_respondents": int(len(numeric)), "items": rows}


def _choice_codes(frame: pd.DataFrame, item: dict[str, Any]) -> pd.Series:
    if item["encoding"] == "single":
        return pd.to_numeric(frame[item["column"]], errors="coerce")
    block = frame[item["columns"]].apply(pd.to_numeric, errors="coerce")
    selected = block.fillna(0).gt(0)
    any_selected = selected.any(axis=1)
    index = selected.to_numpy().argmax(axis=1) + 1
    codes = pd.Series(index, index=frame.index, dtype="float64")
    return codes.where(any_selected)


def sample_panel(data_dir: Path, items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Nemotron adults, with one turn per closed-ended item and no gold answer."""
    from popbench.dao.personas import PersonaError

    try:
        adults = sample_usa_adults(data_dir, n=n, seed=seed)
    except PersonaError as exc:
        raise InterviewError(str(exc)) from exc
    turns = [
        {
            "turn": item["turn"],
            "item_id": item["item_id"],
            "kind": item["kind"],
            "family": item["family"],
            "block": item["block"],
            "prompt": item["prompt"],
            "options": list(item["options"]),
        }
        for item in items
    ]
    records = []
    for person in adults:
        records.append(
            {
                "id": f"nemotron:{person.id}",
                "persona": {
                    "id": person.id,
                    "age_band": person.age_band,
                    "sex": person.sex,
                    "text": person.text,
                },
                "turns": turns,
            }
        )
    return records


def prepare_survey(
    data_dir: Path,
    n_people: int = 50,
    seed: int = 0,
    ask_n: int = ASK_N,
) -> dict[str, Any]:
    """Load or build the closed-item panel and the human baseline."""
    from popbench.dao.twin2k import load_twin2k

    data = load_twin2k(data_dir, download=False)
    closed = build_closed_items(data.catalog)
    items = cap_items(closed, ask_n)
    out = survey_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_path = out / "human_baseline.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
    else:
        baseline = human_baseline(data.wave1_3, items)
        baseline_path.write_text(json.dumps(baseline) + "\n")
    panel_path = out / "nemotron_panel.jsonl"
    if panel_path.is_file():
        records = read_jsonl(panel_path)
        kept = {item["item_id"] for item in items}
        for record in records:
            selected = [turn for turn in record["turns"] if turn["item_id"] in kept]
            for turn_no, turn in enumerate(selected, start=1):
                turn["turn"] = turn_no
            record["turns"] = selected
    else:
        records = sample_panel(data_dir, items, n=n_people, seed=seed)
        write_jsonl(panel_path, records)
    manifest = {"n_closed": len(closed), "n_asked": len(items), "n_people": len(records)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "items": items,
        "baseline": baseline,
        "records": records,
        "n_closed": len(closed),
        "panel_path": panel_path,
    }


