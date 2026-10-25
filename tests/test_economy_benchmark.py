"""Fixed Twin-2K slice for economic preferences, experiments, and pricing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from popbench.dao.economy_benchmark import (
    DATASET_NAME,
    ECONOMIC_COLUMNS,
    ECONOMY_BENCHMARK_COLUMNS,
    EXPERIMENT_COLUMNS,
    PRICING_COLUMNS,
    bind_economy_benchmark,
    build_economy_benchmark_panels,
    economy_benchmark_dir,
    economy_benchmark_items,
    load_economy_benchmark,
    responses_cover,
    write_economy_benchmark,
)
from popbench.dao.io import write_jsonl
from popbench.dao.interview import InterviewError
from popbench.dao.schema import TWIN2K_N
from popbench.dao.twin2k import Twin2KData, cache_is_complete, load_twin2k

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_columns_are_the_requested_items():
    assert ECONOMY_BENCHMARK_COLUMNS == ECONOMIC_COLUMNS + EXPERIMENT_COLUMNS + PRICING_COLUMNS
    assert ECONOMIC_COLUMNS == (
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
    assert EXPERIMENT_COLUMNS == (
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
    assert PRICING_COLUMNS == tuple(f"QID9_{index}" for index in range(1, 41))
    assert len(ECONOMY_BENCHMARK_COLUMNS) == 74
    excluded = {
        "QID150",
        "QID224",
        "QID225",
        "QID250_1",
        "QID271_1",
        "QID272_1",
        "QID275_1",
        "QID287_2",
        "QID159_2",
        "QID8",
    }
    assert excluded.isdisjoint(ECONOMY_BENCHMARK_COLUMNS)


def test_matrix_rows_follow_the_requested_columns():
    catalog = [
        _matrix(
            "QID287",
            "False consensus",
            ["1", "2", "3", "4", "5", "6", "7", "10", "11", "12"],
        )
    ]
    original = list(catalog[0]["RowsID"])
    entries = bind_economy_benchmark(catalog, ("QID287_12", "QID287_1", "QID287_5"))

    assert entries[0]["RowsID"] == ["12", "1", "5"]
    assert entries[0]["csv_columns"] == ["QID287_12", "QID287_1", "QID287_5"]
    assert entries[0]["Rows"] == ["statement 12", "statement 1", "statement 5"]
    assert catalog[0]["RowsID"] == original


def test_entries_follow_the_first_selected_column():
    catalog = [
        _choice("QID229", "Economic preferences"),
        _choice("QID149", "Economic preferences"),
    ]
    entries = bind_economy_benchmark(catalog, ("QID149", "QID229"))
    assert [entry["QuestionID"] for entry in entries] == ["QID149", "QID229"]


def test_columns_outside_the_three_families_are_rejected():
    catalog = [_choice("QID12", "Demographics")]
    with pytest.raises(InterviewError, match="economic, experiments, and pricing"):
        bind_economy_benchmark(catalog, ("QID12",))


def test_missing_column_is_rejected():
    with pytest.raises(InterviewError, match="QID229"):
        bind_economy_benchmark([], ("QID229",))


def test_write_binds_wave_tables_in_selection_order(tmp_path: Path):
    catalog = [
        _choice("QID229", "Economic preferences"),
        _choice("QID158", "Disease-loss"),
        _choice("QID9_1", "Product Preferences - Pricing"),
    ]
    columns = ("QID229", "QID158", "QID9_1")
    wave1 = pd.DataFrame(
        {"pid": ["a", "b"], "QID9_1": [1, 0], "QID229": [2, 2], "QID158": [3, 4]}
    )
    labels = wave1.copy()
    labels["QID229"] = ["accept", "accept"]
    wave4 = pd.DataFrame({"pid": ["a", "b"], "QID158": [3, 9], "QID9_1": [1, 1]})
    summary = write_economy_benchmark(
        _dataset(catalog, wave1, labels, wave4, wave4.copy()),
        tmp_path,
        columns,
    )

    assert summary["name"] == DATASET_NAME
    assert summary["families"] == {"economic": 1, "experiments": 1, "pricing": 1}
    assert summary["wave4_columns"] == 2
    written = pd.read_csv(tmp_path / "wave1_3_response.csv")
    assert list(written.columns) == ["pid", "QID229", "QID158", "QID9_1"]
    assert written["QID229"].tolist() == [2, 2]
    written_labels = pd.read_csv(tmp_path / "wave1_3_response_label.csv")
    assert written_labels["QID229"].tolist() == ["accept", "accept"]
    wave4_written = pd.read_csv(tmp_path / "wave4_response.csv")
    assert list(wave4_written.columns) == ["pid", "QID158", "QID9_1"]
    catalog_out = json.loads((tmp_path / "question_catalog.json").read_text())
    assert [entry["QuestionID"] for entry in catalog_out] == ["QID229", "QID158", "QID9_1"]


def test_cached_benchmark_matches_the_source_tables(tmp_path: Path):
    data_dir = ROOT / "data"
    if not cache_is_complete(data_dir):
        pytest.skip("Twin-2K cache is not on disk")
    data = load_twin2k(data_dir, download=False)
    summary = write_economy_benchmark(data, tmp_path)

    assert summary["n_respondents"] == TWIN2K_N
    assert summary["n_columns"] == 74
    assert summary["families"] == {"economic": 23, "experiments": 11, "pricing": 40}
    assert summary["wave4_columns"] == 51
    assert [row["column"] for row in summary["columns"]] == list(ECONOMY_BENCHMARK_COLUMNS)

    wave1 = pd.read_csv(tmp_path / "wave1_3_response.csv")
    labels = pd.read_csv(tmp_path / "wave1_3_response_label.csv")
    assert list(wave1.columns) == ["pid", *ECONOMY_BENCHMARK_COLUMNS]
    assert list(labels.columns) == list(wave1.columns)
    assert len(wave1) == TWIN2K_N
    pd.testing.assert_series_equal(
        wave1["QID251_14"],
        data.wave1_3["QID251_14"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        wave1["pid"],
        data.wave1_3["pid"].reset_index(drop=True),
        check_names=False,
    )

    wave4_columns = [row["column"] for row in summary["columns"] if row["family"] != "economic"]
    wave4 = pd.read_csv(tmp_path / "wave4_response.csv")
    assert list(wave4.columns) == ["pid", *wave4_columns]
    pd.testing.assert_series_equal(
        wave4["QID9_1"],
        data.wave4["QID9_1"].reset_index(drop=True),
        check_names=False,
    )

    catalog = json.loads((tmp_path / "question_catalog.json").read_text())
    by_id = {entry["QuestionID"]: entry for entry in catalog}
    assert by_id["QID287"]["RowsID"] == ["1", "3", "5", "10", "12"]
    assert by_id["QID287"]["csv_columns"] == [
        "QID287_1",
        "QID287_3",
        "QID287_5",
        "QID287_10",
        "QID287_12",
    ]
    assert by_id["QID159"]["csv_columns"] == ["QID159_1"]
    assert len(by_id["QID251"]["Rows"]) == 14
    assert "QID8" not in by_id
    assert economy_benchmark_dir(data_dir).name == DATASET_NAME


def test_saved_benchmark_expands_to_the_fixed_columns():
    data_dir = ROOT / "data"
    if not (economy_benchmark_dir(data_dir) / "question_catalog.json").is_file():
        pytest.skip("economy benchmark dataset is not on disk")
    catalog, _wave = load_economy_benchmark(data_dir)
    items = economy_benchmark_items(catalog)
    assert [item["item_id"] for item in items] == list(ECONOMY_BENCHMARK_COLUMNS)
    assert all(item["kind"] == "choice" for item in items)


def test_panels_restart_turns_inside_each_family(tmp_path: Path):
    data_dir = ROOT / "data"
    if not cache_is_complete(data_dir):
        pytest.skip("Twin-2K cache is not on disk")
    if not (economy_benchmark_dir(data_dir) / "wave1_3_response.csv").is_file():
        pytest.skip("economy benchmark dataset is not on disk")
    built = build_economy_benchmark_panels(data_dir, n=2, seed=0, out_dir=tmp_path)
    assert [panel["family"] for panel in built["panels"]] == [
        "economic",
        "experiments",
        "pricing",
    ]
    assert [panel["items"][0]["turn"] for panel in built["panels"]] == [1, 1, 1]
    ids = [item["item_id"] for panel in built["panels"] for item in panel["items"]]
    assert ids == list(ECONOMY_BENCHMARK_COLUMNS)
    assert len(built["personas"]) == 2
    economic = built["panels"][0]
    assert economic["baseline"]["items"][0]["item_id"] == "QID229"
    assert len(economic["baseline"]["items"][0]["shares"]) == len(economic["items"][0]["options"])


def test_responses_cover_requires_every_persona_and_item(tmp_path: Path):
    records = [
        {"id": "a", "turns": [{"item_id": "QID229"}, {"item_id": "QID149"}]},
        {"id": "b", "turns": [{"item_id": "QID229"}, {"item_id": "QID149"}]},
    ]
    path = tmp_path / "responses.jsonl"
    write_jsonl(
        path,
        [
            {"id": "a", "answers": [{"item_id": "QID229"}, {"item_id": "QID149"}]},
            {"id": "b", "answers": [{"item_id": "QID229"}, {"item_id": "QID149"}]},
        ],
    )
    assert responses_cover(path, records)
    write_jsonl(path, [{"id": "a", "answers": [{"item_id": "QID229"}, {"item_id": "QID149"}]}])
    assert not responses_cover(path, records)


def _choice(question_id: str, block: str) -> dict:
    return {
        "QuestionID": question_id,
        "BlockName": block,
        "QuestionType": "MC",
        "QuestionText": question_id,
        "Options": ["Yes", "No"],
        "csv_columns": [question_id],
    }


def _matrix(question_id: str, block: str, row_ids: list[str]) -> dict:
    return {
        "QuestionID": question_id,
        "BlockName": block,
        "QuestionType": "Matrix",
        "QuestionText": "Would you support or oppose...",
        "Rows": [f"statement {row_id}" for row_id in row_ids],
        "RowsID": list(row_ids),
        "Columns": ["No", "Yes"],
        "csv_columns": [f"{question_id}_{row_id}" for row_id in row_ids],
    }


def _dataset(
    catalog: list[dict],
    wave1: pd.DataFrame,
    labels: pd.DataFrame,
    wave4: pd.DataFrame,
    wave4_labels: pd.DataFrame,
) -> Twin2KData:
    return Twin2KData(
        root=Path("."),
        revision="test",
        catalog=catalog,
        wave1_3=wave1,
        wave1_3_labels=labels,
        wave4=wave4,
        wave4_labels=wave4_labels,
    )
