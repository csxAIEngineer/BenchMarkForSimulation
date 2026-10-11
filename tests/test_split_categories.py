"""Split Twin-2K response tables into one folder per paper category."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from popbench.dao.interview import InterviewError
from popbench.dao.survey import PAPER_FAMILIES, write_category_splits
from popbench.dao.twin2k import Twin2KData, cache_is_complete, load_twin2k

ROOT = Path(__file__).resolve().parents[1]

# Response columns in the cached tables, excluding pid.
WAVE1_COLUMNS = {
    "demographics": 14,
    "personality": 338,
    "cognitive": 88,
    "economic": 194,
    "heuristics": 20,
    "experiments": 66,
    "pricing": 40,
}
WAVE4_FAMILIES = {"heuristics", "experiments", "pricing"}


def _dataset(catalog: list[dict], wave1: pd.DataFrame, wave4: pd.DataFrame) -> Twin2KData:
    return Twin2KData(
        root=Path("."),
        revision="test",
        catalog=catalog,
        wave1_3=wave1,
        wave1_3_labels=wave1.copy(),
        wave4=wave4,
        wave4_labels=wave4.copy(),
    )


def test_split_writes_one_folder_per_present_category(tmp_path: Path):
    catalog = [
        {"QuestionID": "Q1", "BlockName": "Demographics", "csv_columns": ["Q1"]},
        {"QuestionID": "Q2", "BlockName": "Personality", "csv_columns": []},
        {
            "QuestionID": "Q3",
            "BlockName": "Product Preferences - Pricing",
            "csv_columns": ["Q3", "Q3_2"],
        },
    ]
    wave1 = pd.DataFrame({"pid": ["a", "b"], "Q3": [1, 0], "Q1": [2, 3], "Q3_2": [4, 5]})
    wave4 = pd.DataFrame({"pid": ["a", "b"], "Q3_2": [4, 9], "Q3": [1, 1]})
    summary = write_category_splits(_dataset(catalog, wave1, wave4), tmp_path)

    assert [row["family"] for row in summary["categories"]] == [
        "demographics",
        "personality",
        "pricing",
    ]
    demo = pd.read_csv(tmp_path / "demographics" / "wave1_3_response.csv")
    assert list(demo.columns) == ["pid", "Q1"]
    assert list(demo["pid"]) == ["a", "b"]
    personality = json.loads((tmp_path / "personality" / "question_catalog.json").read_text())
    assert personality[0]["QuestionID"] == "Q2"
    assert list(pd.read_csv(tmp_path / "personality" / "wave1_3_response.csv").columns) == ["pid"]
    pricing = pd.read_csv(tmp_path / "pricing" / "wave4_response.csv")
    assert list(pricing.columns) == ["pid", "Q3_2", "Q3"]
    assert not (tmp_path / "demographics" / "wave4_response.csv").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["categories"][2]["wave4_columns"] == 2
    assert manifest["categories"][0]["label"] == "人口学"


def test_split_rejects_a_column_owned_twice(tmp_path: Path):
    catalog = [
        {"QuestionID": "Q1", "BlockName": "Demographics", "csv_columns": ["Q1"]},
        {"QuestionID": "Q1", "BlockName": "Personality", "csv_columns": ["Q1"]},
    ]
    wave1 = pd.DataFrame({"pid": ["a"], "Q1": [1]})
    wave4 = pd.DataFrame({"pid": ["a"]})
    with pytest.raises(InterviewError, match="belongs to more than one"):
        write_category_splits(_dataset(catalog, wave1, wave4), tmp_path)


def test_cached_tables_partition_into_seven_categories(tmp_path: Path):
    data_dir = ROOT / "data"
    if not cache_is_complete(data_dir):
        pytest.skip("Twin-2K cache is not on disk")
    summary = write_category_splits(load_twin2k(data_dir, download=False), tmp_path)
    assert [row["family"] for row in summary["categories"]] == list(PAPER_FAMILIES)
    by_family = {row["family"]: row for row in summary["categories"]}
    for family, n_columns in WAVE1_COLUMNS.items():
        row = by_family[family]
        assert row["wave1_3_columns"] == n_columns
        folder = tmp_path / family
        wave1 = pd.read_csv(folder / "wave1_3_response.csv", nrows=0)
        labels = pd.read_csv(folder / "wave1_3_response_label.csv", nrows=0)
        assert list(wave1.columns) == list(labels.columns)
        assert len(wave1.columns) == n_columns + 1
        wave4_path = folder / "wave4_response.csv"
        if family in WAVE4_FAMILIES:
            assert row["wave4_columns"] == n_columns
            assert wave4_path.is_file()
        else:
            assert row["wave4_columns"] == 0
            assert not wave4_path.exists()
    assert sum(row["wave1_3_columns"] for row in summary["categories"]) == 760
    assert sum(row["wave4_columns"] for row in summary["categories"]) == 126
