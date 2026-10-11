"""ATUS adults split into population groups with a replicate-weight test."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from popbench.dao.atus_populations import (
    ACTIVITY_LABELS,
    POPULATION_FACTORS,
    build_adult_frame,
    chi2_sf,
    evaluate_populations,
    replicate_matrix_from_frame,
    split_atus_populations,
    split_population_tables,
    survey_wald,
)
from popbench.dao.visitors import ATUS_MAJOR, VisitorError

ROOT = Path(__file__).resolve().parents[1]


def test_chi2_survival_matches_known_tail_probabilities():
    assert chi2_sf(0, 1) == 1.0
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-5)
    assert chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf(25, 1) == pytest.approx(math.erfc(5 / math.sqrt(2)))
    assert chi2_sf(10_000, 11) == 0.0


def test_survey_wald_uses_the_atus_replicate_formula():
    values = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    groups = np.array([0, 0, 1, 1])
    final = np.ones(4)
    replicates = np.array(
        [
            [1.1, 0.9],
            [0.9, 1.1],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    full = _means(values, final, groups)
    reps = _means_replicates(values, replicates, groups)
    wald, df, p_value = survey_wald(full, reps)
    assert wald == pytest.approx(25)
    assert df == 1
    assert p_value == pytest.approx(chi2_sf(25, 1))

    wide = np.array(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    wide_wald, _, wide_p = survey_wald(full, _means_replicates(values, wide, groups))
    assert wide_wald == pytest.approx(1)
    assert wide_p == pytest.approx(chi2_sf(1, 1))


def test_identical_group_means_are_not_significant():
    values = np.array([[0.0], [100.0], [0.0], [100.0]])
    groups = np.array([0, 0, 1, 1])
    weights = np.ones((4, 2))
    wald, _, p_value = survey_wald(_means(values, np.ones(4), groups), _means_replicates(values, weights, groups))
    assert wald == 0
    assert p_value == 1


def test_build_adult_frame_labels_and_drops_minors():
    frame = pd.DataFrame(
        [
            _row(TUCASEID=1, TEAGE=16, TESEX=2),
            _row(TUCASEID=3, TEAGE=29, TESEX=2, PEEDUCA=38, TRDPFTPT=2, TRSPPRES=2),
            _row(TUCASEID=2, TEAGE=70, PEEDUCA=43, TRCHILDNUM=1, TRYHHCHILD=10, GTMETSTA=3),
            _row(TUCASEID=4, TEAGE=40, PEHSPNON=1, PTDTRACE=1, TESCHENR=1, TRCHILDNUM=1, TRYHHCHILD=3),
            _row(TUCASEID=5, TEAGE=55, PTDTRACE=4, TRDPFTPT=-1),
            _row(TUCASEID=6, TEAGE=64, PTDTRACE=3, TESCHENR=-1),
        ]
    )
    frame["t010101"] = 60
    frame["t010102"] = -5
    adults = build_adult_frame(frame)
    assert adults["TUCASEID"].tolist() == [2, 3, 4, 5, 6]
    assert adults["personal_care"].tolist() == [60, 60, 60, 60, 60]
    by_id = adults.set_index("TUCASEID")
    assert by_id.loc[3, "sex"] == "Female"
    assert by_id.loc[3, "age_band"] == "18-29"
    assert by_id.loc[3, "education_level"] == "less_than_high_school"
    assert by_id.loc[3, "employment"] == "part_time"
    assert by_id.loc[3, "partner"] == "unmarried_partner"
    assert by_id.loc[2, "age_band"] == "65+"
    assert by_id.loc[2, "education_level"] == "bachelor_or_higher"
    assert by_id.loc[2, "children"] == "youngest_6_to_17"
    assert pd.isna(by_id.loc[2, "metro"])
    assert by_id.loc[4, "children"] == "youngest_under_6"
    assert by_id.loc[4, "race_ethnicity"] == "hispanic"
    assert by_id.loc[4, "school"] == "enrolled"
    assert by_id.loc[5, "race_ethnicity"] == "asian_nonhispanic"
    assert by_id.loc[5, "employment"] == "not_employed"
    assert by_id.loc[6, "age_band"] == "50-64"
    assert by_id.loc[6, "race_ethnicity"] == "other"
    assert pd.isna(by_id.loc[6, "school"])


def test_replicate_matrix_follows_case_order_and_rejects_gaps():
    replicates = _replicate_frame([1, 2])
    replicates.loc[replicates["TUCASEID"] == 2, "FINLWGT001"] = 7
    aligned = replicate_matrix_from_frame(pd.Series([2, 1]), replicates)
    assert aligned.shape == (2, 160)
    assert aligned[0, 0] == 7
    assert aligned[1, 0] == 1
    with pytest.raises(VisitorError, match="missing for"):
        replicate_matrix_from_frame(pd.Series([1, 3]), replicates)


def test_small_level_is_omitted_and_a_real_gap_is_kept(tmp_path: Path):
    adults = _population_frame(
        [
            (1, "full_time", 0),
            (2, "full_time", 100),
            (3, "not_employed", 0),
            (4, "not_employed", 0),
            (5, "part_time", 0),
        ]
    )
    replicates = np.array(
        [
            [1.1, 0.9],
            [0.9, 1.1],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    written = split_population_tables(adults, replicates, tmp_path, min_n=2)
    employment = next(row for row in written["factors"] if row["factor"] == "employment")
    assert employment["kept"] is True
    assert employment["joint_wald"] == pytest.approx(25, abs=1e-4)
    assert employment["p_value"] == pytest.approx(chi2_sf(25, 1))
    assert employment["levels_omitted"] == [
        {"level": "part_time", "title": "兼职", "n": 1}
    ]
    assert [group["level"] for group in employment["groups"]] == ["full_time", "not_employed"]
    assert employment["groups"][0]["mean_minutes"]["work"] == pytest.approx(50)
    assert employment["groups"][1]["mean_minutes"]["work"] == pytest.approx(0)
    work = next(row for row in employment["activities"] if row["activity"] == "work")
    assert work["significant"] is True
    assert (tmp_path / "employment" / "full_time.csv").is_file()
    assert (tmp_path / "employment" / "not_employed.csv").is_file()
    assert not (tmp_path / "employment" / "part_time.csv").exists()
    assert not (tmp_path / "sex").exists()
    full_time = pd.read_csv(tmp_path / "employment" / "full_time.csv")
    assert full_time["TUCASEID"].tolist() == [1, 2]
    assert full_time["work"].tolist() == [0, 100]
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["min_group_n"] == 2
    assert saved["n_adults"] == 5


def test_a_noisy_gap_is_not_written(tmp_path: Path):
    adults = _population_frame(
        [
            (1, "full_time", 0),
            (2, "full_time", 100),
            (3, "not_employed", 0),
            (4, "not_employed", 0),
        ]
    )
    replicates = np.array(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    results = evaluate_populations(adults, replicates, min_n=2)
    employment = next(row for row in results if row["factor"] == "employment")
    assert employment["kept"] is False
    assert employment["joint_wald"] == pytest.approx(1)
    assert employment["reason"] == "joint test is not below alpha"
    written = split_population_tables(adults, replicates, tmp_path, min_n=2)
    assert [row["factor"] for row in written["factors"] if row["kept"]] == []
    assert {path.name for path in tmp_path.iterdir()} == {"adults.csv", "manifest.json"}


def test_cached_atus_populations_differ_on_time_use(tmp_path: Path):
    summary = ROOT / "data" / "visitors" / "atus" / "sum" / "atussum_2023.dat"
    weights = ROOT / "data" / "visitors" / "atus" / "wgts" / "atuswgts_2023.dat"
    if not summary.is_file() or not weights.is_file():
        pytest.skip("ATUS summary or replicate weights are not on disk")
    written = split_atus_populations(ROOT / "data", dest=tmp_path)
    assert written["n_adults"] == 8382
    assert written["n_under_18"] == 166
    assert written["n_replicates"] == 160
    by_factor = {row["factor"]: row for row in written["factors"]}
    assert set(by_factor) == {factor.key for factor in POPULATION_FACTORS}
    _assert_counts(
        by_factor["sex"],
        {"Male": 3913, "Female": 4469},
    )
    _assert_counts(
        by_factor["age_band"],
        {"18-29": 881, "30-49": 2428, "50-64": 2083, "65+": 2990},
    )
    _assert_counts(
        by_factor["education_level"],
        {
            "less_than_high_school": 487,
            "high_school": 1913,
            "some_college": 2215,
            "bachelor_or_higher": 3767,
        },
    )
    _assert_counts(
        by_factor["employment"],
        {"full_time": 3978, "part_time": 951, "not_employed": 3453},
    )
    _assert_counts(
        by_factor["children"],
        {"no_household_child": 6219, "youngest_under_6": 915, "youngest_6_to_17": 1248},
    )
    _assert_counts(
        by_factor["partner"],
        {"spouse": 4558, "unmarried_partner": 357, "none": 3467},
    )
    assert by_factor["race_ethnicity"]["levels_omitted"][0]["level"] == "other"
    assert by_factor["race_ethnicity"]["levels_omitted"][0]["n"] == 140
    _assert_counts(by_factor["metro"], {"metropolitan": 7037, "nonmetropolitan": 1244})
    assert by_factor["metro"]["n_unlabeled"] == 101
    _assert_counts(by_factor["school"], {"enrolled": 301, "not_enrolled": 3008})
    assert by_factor["school"]["n_unlabeled"] == 5073
    assert "156" in by_factor["employment"]["note"]
    for key in (
        "sex",
        "age_band",
        "education_level",
        "employment",
        "children",
        "partner",
        "metro",
        "school",
        "race_ethnicity",
    ):
        assert by_factor[key]["kept"] is True
        assert by_factor[key]["p_value"] < 0.05
        folder = tmp_path / key
        seen: list[int] = []
        for group in by_factor[key]["groups"]:
            part = pd.read_csv(folder / f"{group['level']}.csv")
            assert len(part) == group["n"]
            assert part["TUCASEID"].is_unique
            seen.extend(part["TUCASEID"].tolist())
        assert len(seen) == len(set(seen))
    sex = _means_by_level(by_factor["sex"])
    assert sex["Male"]["work"] > sex["Female"]["work"]
    assert sex["Female"]["household"] > sex["Male"]["household"]
    assert sex["Female"]["care_household"] > sex["Male"]["care_household"]
    employment = _means_by_level(by_factor["employment"])
    assert employment["full_time"]["work"] > employment["part_time"]["work"]
    assert employment["part_time"]["work"] > employment["not_employed"]["work"]
    children = _means_by_level(by_factor["children"])
    assert children["youngest_under_6"]["care_household"] > children["no_household_child"]["care_household"]
    age = _means_by_level(by_factor["age_band"])
    assert age["30-49"]["work"] > age["65+"]["work"]


def _assert_counts(factor: dict, expected: dict[str, int]) -> None:
    assert {group["level"]: group["n"] for group in factor["groups"]} == expected


def _means_by_level(factor: dict) -> dict[str, dict[str, float]]:
    return {group["level"]: group["mean_minutes"] for group in factor["groups"]}


def _means(values: np.ndarray, weights: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from popbench.dao.atus_populations import _grouped_means

    return _grouped_means(values, weights, groups, 2, "test")


def _means_replicates(values: np.ndarray, replicates: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from popbench.dao.atus_populations import _replicate_means

    return _replicate_means(values, replicates, groups, 2, "test")


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "TUCASEID": 1,
        "TUFINLWGT": 10.0,
        "TEAGE": 40,
        "TESEX": 1,
        "PEEDUCA": 39,
        "PTDTRACE": 1,
        "PEHSPNON": 2,
        "GTMETSTA": 1,
        "TRDPFTPT": 1,
        "TESCHENR": 2,
        "TRSPPRES": 3,
        "TRCHILDNUM": 0,
        "TRYHHCHILD": -1,
    }
    for code in ATUS_MAJOR:
        row[f"t{code}0101"] = 0
    row.update(overrides)
    return row


def _population_frame(rows: list[tuple[int, str, int]]) -> pd.DataFrame:
    records = []
    for case_id, employment, work in rows:
        record = _row(TUCASEID=case_id, TRDPFTPT={"full_time": 1, "part_time": 2, "not_employed": -1}[employment])
        record["t050101"] = work
        records.append(record)
    adults = build_adult_frame(pd.DataFrame(records))
    assert adults["employment"].tolist() == [employment for _, employment, _ in rows]
    return adults


def _replicate_frame(case_ids: list[int]) -> pd.DataFrame:
    data: dict[str, object] = {"TUCASEID": case_ids}
    for index in range(1, 161):
        data[f"FINLWGT{index:03d}"] = 1.0
    return pd.DataFrame(data)


def test_activity_labels_follow_the_persona_major_groups():
    assert ACTIVITY_LABELS[0] == "personal_care"
    assert "work" in ACTIVITY_LABELS
    assert "care_household" in ACTIVITY_LABELS
