"""Split ATUS 2023 adults into population groups that differ in time use.

Each factor is a partition of adults (age 18+). A factor is written only when
a replicate-weight Wald test says the groups do not share the same mean
minutes. The variance is the ATUS User's Guide chapter 7 estimator: ``(4/R)``
times the sum of squared deviations of the replicate estimates from the
full-sample estimate. Activities are the same major groups the persona card
uses. Within a factor, activity tests are Holm-adjusted. A level with fewer
than ``MIN_GROUP_N`` adults is left out of that test.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from popbench.dao.atus_twin2k import PEEDUCA_BANDS
from popbench.dao.io import write_jsonl
from popbench.dao.personas import age_band
from popbench.dao.schema import PersonaCard
from popbench.dao.visitors import (
    ATUS_MAJOR,
    SEX_LABEL,
    VisitorError,
    atus_dir,
    cache_atus_replicates,
    population_persona_text,
    visitors_dir,
)

MIN_GROUP_N = 200
ALPHA = 0.05
REPLICATE_COUNT = 160
# The 4 is the ATUS/CPS replication factor (replicate factors 0.3, 1.0, and 1.7).
VARIANCE_NUMERATOR = 4
ACTIVITY_LABELS = tuple(ATUS_MAJOR.values())
REPLICATE_COLUMNS = tuple(f"FINLWGT{index:03d}" for index in range(1, REPLICATE_COUNT + 1))
SUMMARY_COLUMNS = (
    "TUCASEID",
    "TUFINLWGT",
    "TEAGE",
    "TESEX",
    "PEEDUCA",
    "PTDTRACE",
    "PEHSPNON",
    "GTMETSTA",
    "TRDPFTPT",
    "TESCHENR",
    "TRSPPRES",
    "TRCHILDNUM",
    "TRYHHCHILD",
)


@dataclass(frozen=True)
class PopulationFactor:
    key: str
    label: str
    note: str
    levels: tuple[str, ...]


POPULATION_FACTORS = (
    PopulationFactor("sex", "性别", "TESEX。男性与女性。", ("Male", "Female")),
    PopulationFactor(
        "age_band",
        "年龄",
        "TEAGE。与抽样用的四档相同。",
        ("18-29", "30-49", "50-64", "65+"),
    ),
    PopulationFactor(
        "education_level",
        "教育",
        "PEEDUCA。低于高中、高中、部分大学、学士及以上。",
        ("less_than_high_school", "high_school", "some_college", "bachelor_or_higher"),
    ),
    PopulationFactor(
        "employment",
        "就业",
        "TRDPFTPT。全职、兼职、未就业。",
        ("full_time", "part_time", "not_employed"),
    ),
    PopulationFactor(
        "children",
        "家中儿童",
        "TRCHILDNUM 与 TRYHHCHILD。没有 18 岁以下儿童、最小儿童不满 6 岁、最小儿童 6-17 岁。",
        ("no_household_child", "youngest_under_6", "youngest_6_to_17"),
    ),
    PopulationFactor(
        "partner",
        "配偶或伴侣",
        "TRSPPRES。配偶、未婚伴侣、没有。",
        ("spouse", "unmarried_partner", "none"),
    ),
    PopulationFactor(
        "metro",
        "都市地位",
        "GTMETSTA。未识别的不单独成组。",
        ("metropolitan", "nonmetropolitan"),
    ),
    PopulationFactor(
        "school",
        "在学",
        "TESCHENR。不在该题范围内的受访者不进入这一划分。",
        ("enrolled", "not_enrolled"),
    ),
    PopulationFactor(
        "race_ethnicity",
        "种族与西裔",
        "PEHSPNON 与 PTDTRACE。西裔为任意种族。其余非西裔人数不足则不进入检验。",
        ("hispanic", "white_nonhispanic", "black_nonhispanic", "asian_nonhispanic", "other"),
    ),
)
LEVEL_TITLES = {
    "Male": "男性",
    "Female": "女性",
    "18-29": "18-29岁",
    "30-49": "30-49岁",
    "50-64": "50-64岁",
    "65+": "65岁及以上",
    "less_than_high_school": "低于高中",
    "high_school": "高中",
    "some_college": "部分大学",
    "bachelor_or_higher": "学士及以上",
    "full_time": "全职",
    "part_time": "兼职",
    "not_employed": "未就业",
    "no_household_child": "家中无18岁以下儿童",
    "youngest_under_6": "最小儿童不满6岁",
    "youngest_6_to_17": "最小儿童6至17岁",
    "spouse": "有配偶",
    "unmarried_partner": "有未婚伴侣",
    "none": "无配偶或伴侣",
    "metropolitan": "都市",
    "nonmetropolitan": "非都市",
    "enrolled": "在学",
    "not_enrolled": "未在学",
    "hispanic": "西裔",
    "white_nonhispanic": "非西裔白人",
    "black_nonhispanic": "非西裔黑人",
    "asian_nonhispanic": "非西裔亚裔",
    "other": "其他非西裔",
}


def population_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "atus" / "by_population"


def population_visitors_dir(data_dir: Path) -> Path:
    return visitors_dir(data_dir) / "by_population"


def prepared_columns() -> list[str]:
    factor_keys = [factor.key for factor in POPULATION_FACTORS]
    return ["TUCASEID", "TUFINLWGT", "TEAGE", *factor_keys, *ACTIVITY_LABELS]


def chi2_sf(x: float, df: int) -> float:
    """Survival function of a chi-square random variable with ``df`` degrees."""
    if df < 1:
        raise ValueError("df must be positive")
    if x <= 0:
        return 1.0
    return _upper_gamma(df / 2.0, x / 2.0)


def survey_wald(full_means: np.ndarray, replicate_means: np.ndarray) -> tuple[float, int, float]:
    """Wald test that every group shares the same mean on each activity.

    ``full_means`` is shaped (groups, activities). ``replicate_means`` is
    shaped (replicates, groups, activities). The covariance uses ``(4/R)``.
    """
    if full_means.ndim != 2 or replicate_means.ndim != 3:
        raise ValueError("means must be (groups, activities) and (replicates, groups, activities)")
    n_groups, n_activities = full_means.shape
    if replicate_means.shape[1:] != full_means.shape:
        raise ValueError("replicate means do not match the full-sample means")
    if n_groups < 2 or n_activities < 1:
        raise ValueError("the Wald test needs at least two groups and one activity")
    n_replicates = replicate_means.shape[0]
    point = full_means.reshape(n_groups * n_activities)
    replicates = replicate_means.reshape(n_replicates, n_groups * n_activities)
    scale = VARIANCE_NUMERATOR / n_replicates
    deviation = replicates - point
    covariance = scale * (deviation.T @ deviation)
    contrast = _equality_contrast(n_groups, n_activities)
    delta = contrast @ point
    variance = contrast @ covariance @ contrast.T
    return _wald_statistic(delta, variance)


def build_adult_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Adults with population labels and major-activity minutes, one row each."""
    missing = [column for column in SUMMARY_COLUMNS if column not in summary.columns]
    if missing:
        raise VisitorError("ATUS summary is missing columns: " + ", ".join(missing))
    summary = summary.reset_index(drop=True)
    age = pd.to_numeric(summary["TEAGE"], errors="coerce")
    weight = pd.to_numeric(summary["TUFINLWGT"], errors="coerce")
    adults = summary.loc[age.ge(18) & weight.gt(0)].copy()
    if adults.empty:
        raise VisitorError("ATUS summary has no adults")
    adults["TUCASEID"] = pd.to_numeric(adults["TUCASEID"], errors="coerce")
    if adults["TUCASEID"].isna().any() or adults["TUCASEID"].duplicated().any():
        raise VisitorError("ATUS summary has missing or duplicate TUCASEID values")
    adults["TUCASEID"] = adults["TUCASEID"].astype("int64")
    adults["TUFINLWGT"] = weight.loc[adults.index].astype(float)
    adults["TEAGE"] = age.loc[adults.index].astype(int)
    labeled = _assign_labels(adults)
    minutes = _activity_minutes(adults)
    table = pd.concat([labeled, minutes], axis=1)
    table = table.sort_values("TUCASEID").reset_index(drop=True)
    return table.loc[:, prepared_columns()]


def replicate_matrix_from_frame(case_ids: pd.Series, replicates: pd.DataFrame) -> np.ndarray:
    """Replicate weights in the same order as ``case_ids``. Shape (people, 160)."""
    missing = [column for column in ("TUCASEID", *REPLICATE_COLUMNS) if column not in replicates.columns]
    if missing:
        raise VisitorError("ATUS replicate weights are missing columns: " + ", ".join(missing[:8]))
    ids = pd.to_numeric(replicates["TUCASEID"], errors="coerce")
    if ids.isna().any() or ids.duplicated().any():
        raise VisitorError("ATUS replicate weights have missing or duplicate TUCASEID values")
    indexed = replicates.copy()
    indexed["TUCASEID"] = ids.astype("int64")
    indexed = indexed.set_index("TUCASEID")
    wanted = pd.to_numeric(case_ids, errors="coerce").astype("int64")
    absent = [int(case) for case in wanted if int(case) not in indexed.index]
    if absent:
        raise VisitorError(f"ATUS replicate weights are missing for {len(absent)} adults")
    aligned = indexed.loc[wanted.to_numpy(), list(REPLICATE_COLUMNS)]
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise VisitorError("ATUS replicate weights must be finite and nonnegative")
    return values


def evaluate_populations(
    adults: pd.DataFrame,
    replicate_weights: np.ndarray,
    *,
    alpha: float = ALPHA,
    min_n: int = MIN_GROUP_N,
    unemployed_n: int | None = None,
) -> list[dict[str, Any]]:
    """Test each population factor. Rows of ``replicate_weights`` follow ``adults``."""
    _check_test_inputs(adults, replicate_weights, alpha=alpha, min_n=min_n)
    results = []
    for factor in POPULATION_FACTORS:
        note = factor.note
        if factor.key == "employment":
            note = _employment_note(factor.note, unemployed_n, min_n)
        results.append(
            _evaluate_factor(
                adults,
                replicate_weights,
                factor,
                note=note,
                alpha=alpha,
                min_n=min_n,
            )
        )
    return results


def split_population_tables(
    adults: pd.DataFrame,
    replicate_weights: np.ndarray,
    dest: Path,
    *,
    alpha: float = ALPHA,
    min_n: int = MIN_GROUP_N,
    sample: dict[str, int] | None = None,
    unemployed_n: int | None = None,
) -> dict[str, Any]:
    """Write ``adults.csv``, one folder per kept factor, and ``manifest.json``."""
    factors = evaluate_populations(
        adults,
        replicate_weights,
        alpha=alpha,
        min_n=min_n,
        unemployed_n=unemployed_n,
    )
    dest = Path(dest)
    _reset_output(dest)
    adults.to_csv(dest / "adults.csv", index=False)
    for factor in factors:
        if not factor["kept"]:
            continue
        folder = dest / factor["factor"]
        folder.mkdir(parents=True, exist_ok=True)
        for group in factor["groups"]:
            part = adults.loc[adults[factor["factor"]] == group["level"]]
            part.to_csv(folder / f"{group['level']}.csv", index=False)
    payload = _manifest(
        adults,
        replicate_weights,
        factors,
        alpha=alpha,
        min_n=min_n,
        sample=sample,
    )
    (dest / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return {**payload, "dir": dest}


def build_population_visitors(data_dir: Path) -> dict[str, Any]:
    """Write one persona card for every adult in each kept population group.

    Reads ``data/atus/by_population/<factor>/<level>.csv`` and writes
    ``data/visitors/by_population/<factor>/<level>/visitors.jsonl``. Every row
    becomes a card. Rows are ordered by ``TUCASEID``.
    """
    source = population_dir(data_dir)
    if not source.is_dir():
        raise VisitorError(
            f"ATUS population split is missing: {source}. Run `popbench split-populations`."
        )
    loaded = _load_population_groups(source)
    if not loaded:
        raise VisitorError(f"ATUS population split has no group tables: {source}")
    dest = population_visitors_dir(data_dir)
    _reset_visitor_output(dest)
    groups: list[dict[str, Any]] = []
    for factor, level, frame in loaded:
        folder = dest / factor.key / level
        folder.mkdir(parents=True, exist_ok=True)
        records = [
            _population_visitor_record(row, factor.key, level)
            for row in frame.to_dict(orient="records")
        ]
        write_jsonl(folder / "visitors.jsonl", records)
        (folder / "example.md").write_text(_population_example(factor, level, records))
        group = {
            "factor": factor.key,
            "label": factor.label,
            "level": level,
            "level_title": LEVEL_TITLES.get(level, level),
            "n_people": len(records),
            "panel": f"{factor.key}/{level}/visitors.jsonl",
        }
        (folder / "manifest.json").write_text(
            json.dumps({**group, "source": "atus-population"}, ensure_ascii=False, indent=2) + "\n"
        )
        groups.append(group)
    payload = {
        "source": "atus-population",
        "n_groups": len(groups),
        "n_cards": sum(group["n_people"] for group in groups),
        "groups": groups,
    }
    (dest / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {**payload, "dir": dest}


def split_atus_populations(data_dir: Path, dest: Path | None = None) -> dict[str, Any]:
    """Read the cached ATUS summary and replicate weights, then write the split."""
    data_dir = Path(data_dir)
    summary_path = atus_dir(data_dir) / "sum" / "atussum_2023.dat"
    if not summary_path.is_file():
        raise VisitorError(
            f"ATUS activity summary is missing: {summary_path}. "
            "Run `popbench fetch --visitors-only`."
        )
    weight_path = cache_atus_replicates(data_dir)
    summary = pd.read_csv(summary_path)
    adults = build_adult_frame(summary)
    replicate_weights = replicate_matrix_from_frame(
        adults["TUCASEID"],
        pd.read_csv(weight_path, usecols=["TUCASEID", *REPLICATE_COLUMNS]),
    )
    age = pd.to_numeric(summary["TEAGE"], errors="coerce")
    weight = pd.to_numeric(summary["TUFINLWGT"], errors="coerce")
    labor = pd.to_numeric(summary["TELFS"], errors="coerce") if "TELFS" in summary.columns else None
    unemployed_n = None
    if labor is not None:
        unemployed_n = int((age.ge(18) & labor.isin([3, 4])).sum())
    sample = {
        "n_respondents": int(len(summary)),
        "n_under_18": int(age.lt(18).sum()),
        "n_missing_age": int(age.isna().sum()),
        "n_adults_nonpositive_weight": int((age.ge(18) & ~weight.gt(0)).sum()),
    }
    return split_population_tables(
        adults,
        replicate_weights,
        dest if dest is not None else population_dir(data_dir),
        sample=sample,
        unemployed_n=unemployed_n,
    )


def _assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = pd.DataFrame(
        {
            "TUCASEID": frame["TUCASEID"].to_numpy(),
            "TUFINLWGT": frame["TUFINLWGT"].to_numpy(dtype=float),
            "TEAGE": frame["TEAGE"].to_numpy(dtype=int),
        },
        index=frame.index,
    )
    labeled["sex"] = _sex_labels(frame["TESEX"])
    labeled["age_band"] = frame["TEAGE"].map(lambda age: age_band(int(age)))
    labeled["education_level"] = _education_labels(frame["PEEDUCA"])
    labeled["employment"] = _employment_labels(frame["TRDPFTPT"])
    labeled["children"] = _children_labels(frame["TRCHILDNUM"], frame["TRYHHCHILD"])
    labeled["partner"] = _partner_labels(frame["TRSPPRES"])
    labeled["metro"] = _metro_labels(frame["GTMETSTA"])
    labeled["school"] = _school_labels(frame["TESCHENR"])
    labeled["race_ethnicity"] = _race_labels(frame["PTDTRACE"], frame["PEHSPNON"])
    return labeled


def _sex_labels(codes: pd.Series) -> pd.Series:
    sex = pd.to_numeric(codes, errors="coerce")
    labels = _blank(codes)
    labels.loc[sex == 1] = SEX_LABEL[1]
    labels.loc[sex == 2] = SEX_LABEL[2]
    return labels


def _education_labels(codes: pd.Series) -> pd.Series:
    number = pd.to_numeric(codes, errors="coerce")
    labels = _blank(codes)
    for low, high, name in PEEDUCA_BANDS:
        labels.loc[number.ge(low) & number.le(high)] = name
    return labels


def _employment_labels(codes: pd.Series) -> pd.Series:
    status = pd.to_numeric(codes, errors="coerce")
    labels = _blank(codes)
    labels.loc[status == 1] = "full_time"
    labels.loc[status == 2] = "part_time"
    labels.loc[status == -1] = "not_employed"
    return labels


def _children_labels(counts: pd.Series, youngest: pd.Series) -> pd.Series:
    count = pd.to_numeric(counts, errors="coerce")
    age = pd.to_numeric(youngest, errors="coerce")
    labels = _blank(counts)
    labels.loc[count == 0] = "no_household_child"
    labels.loc[count.gt(0) & age.ge(0) & age.le(5)] = "youngest_under_6"
    labels.loc[count.gt(0) & age.ge(6) & age.le(17)] = "youngest_6_to_17"
    return labels


def _partner_labels(codes: pd.Series) -> pd.Series:
    status = pd.to_numeric(codes, errors="coerce")
    labels = _blank(codes)
    labels.loc[status == 1] = "spouse"
    labels.loc[status == 2] = "unmarried_partner"
    labels.loc[status == 3] = "none"
    return labels


def _metro_labels(codes: pd.Series) -> pd.Series:
    status = pd.to_numeric(codes, errors="coerce")
    labels = _blank(codes)
    labels.loc[status == 1] = "metropolitan"
    labels.loc[status == 2] = "nonmetropolitan"
    return labels


def _school_labels(codes: pd.Series) -> pd.Series:
    status = pd.to_numeric(codes, errors="coerce")
    labels = _blank(codes)
    labels.loc[status == 1] = "enrolled"
    labels.loc[status == 2] = "not_enrolled"
    return labels


def _race_labels(race_codes: pd.Series, hispanic_codes: pd.Series) -> pd.Series:
    race = pd.to_numeric(race_codes, errors="coerce")
    hispanic = pd.to_numeric(hispanic_codes, errors="coerce")
    labels = _blank(race_codes)
    labels.loc[hispanic == 1] = "hispanic"
    nonhispanic = hispanic == 2
    labels.loc[nonhispanic & (race == 1)] = "white_nonhispanic"
    labels.loc[nonhispanic & (race == 2)] = "black_nonhispanic"
    labels.loc[nonhispanic & (race == 4)] = "asian_nonhispanic"
    labels.loc[nonhispanic & race.notna() & ~race.isin([1, 2, 4])] = "other"
    return labels


def _blank(index_source: pd.Series) -> pd.Series:
    return pd.Series(None, index=index_source.index, dtype=object)


def _activity_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    minutes: dict[str, pd.Series] = {}
    for code, label in ATUS_MAJOR.items():
        columns = [
            column
            for column in frame.columns
            if isinstance(column, str)
            and column.startswith("t")
            and len(column) >= 3
            and column[1:3] == code
            and column[1:].isdigit()
        ]
        if not columns:
            raise VisitorError(f"ATUS summary has no minutes for major activity {code} ({label})")
        block = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
        minutes[label] = block.sum(axis=1).round().astype(int)
    return pd.DataFrame(minutes, index=frame.index)


def _evaluate_factor(
    adults: pd.DataFrame,
    replicate_weights: np.ndarray,
    factor: PopulationFactor,
    *,
    note: str,
    alpha: float,
    min_n: int,
) -> dict[str, Any]:
    raw = adults[factor.key]
    counts = {level: int((raw == level).sum()) for level in factor.levels}
    omitted = [
        {"level": level, "title": LEVEL_TITLES[level], "n": count}
        for level, count in counts.items()
        if 0 < count < min_n
    ]
    kept_levels = [level for level in factor.levels if counts[level] >= min_n]
    base = {
        "factor": factor.key,
        "label": factor.label,
        "note": note,
        "n_unlabeled": int(raw.isna().sum()),
        "levels_omitted": omitted,
        "groups": [],
        "activities": [],
        "joint_wald": None,
        "df": None,
        "p_value": None,
        "kept": False,
    }
    if len(kept_levels) < 2:
        base["reason"] = "fewer than 2 groups at or above the minimum count"
        return base
    mask = raw.isin(kept_levels).to_numpy()
    level_index = {level: index for index, level in enumerate(kept_levels)}
    groups = pd.Series(raw.to_numpy()[mask]).map(level_index).to_numpy(dtype=int)
    values = adults.loc[:, list(ACTIVITY_LABELS)].to_numpy(dtype=float)[mask]
    weights = adults["TUFINLWGT"].to_numpy(dtype=float)[mask]
    full = _grouped_means(values, weights, groups, len(kept_levels), factor.key)
    replicates = _replicate_means(
        values,
        replicate_weights[mask],
        groups,
        len(kept_levels),
        factor.key,
    )
    joint_wald, joint_df, joint_p = survey_wald(full, replicates)
    activity_rows = []
    for index, activity in enumerate(ACTIVITY_LABELS):
        wald, df, p_value = survey_wald(full[:, index : index + 1], replicates[:, :, index : index + 1])
        activity_rows.append(
            {
                "activity": activity,
                "wald": _finite(wald),
                "df": int(df),
                "p_value": float(p_value),
                "minute_range": round(float(full[:, index].max() - full[:, index].min()), 2),
            }
        )
    adjusted = _holm([row["p_value"] for row in activity_rows])
    for row, p_holm in zip(activity_rows, adjusted):
        row["p_holm"] = float(p_holm)
        row["significant"] = bool(p_holm < alpha)
    kept = joint_p < alpha and any(row["significant"] for row in activity_rows)
    if joint_p >= alpha:
        reason = "joint test is not below alpha"
    elif not kept:
        reason = "no major activity stays significant after Holm correction"
    else:
        reason = None
    total_weight = float(weights.sum())
    groups_out = []
    for index, level in enumerate(kept_levels):
        group_weight = float(weights[groups == index].sum())
        groups_out.append(
            {
                "level": level,
                "title": LEVEL_TITLES[level],
                "n": counts[level],
                "weighted_share": round(group_weight / total_weight, 6),
                "mean_minutes": {
                    activity: round(float(full[index, column]), 2)
                    for column, activity in enumerate(ACTIVITY_LABELS)
                },
            }
        )
    return {
        **base,
        "groups": groups_out,
        "activities": activity_rows,
        "joint_wald": _finite(joint_wald),
        "df": int(joint_df),
        "p_value": float(joint_p),
        "kept": kept,
        "reason": reason,
    }


def _grouped_means(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    factor_key: str,
) -> np.ndarray:
    means = np.empty((n_groups, values.shape[1]), dtype=float)
    for group in range(n_groups):
        mask = groups == group
        group_weight = weights[mask]
        denom = float(group_weight.sum())
        if denom <= 0:
            raise VisitorError(f"{factor_key}: a group has no positive final weight")
        means[group] = (group_weight[:, None] * values[mask]).sum(axis=0) / denom
    return means


def _replicate_means(
    values: np.ndarray,
    replicate_weights: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    factor_key: str,
) -> np.ndarray:
    n_replicates = replicate_weights.shape[1]
    means = np.empty((n_replicates, n_groups, values.shape[1]), dtype=float)
    for group in range(n_groups):
        mask = groups == group
        group_weights = replicate_weights[mask]
        denom = group_weights.sum(axis=0)
        if np.any(denom <= 0):
            bad = np.flatnonzero(denom <= 0)[:5].tolist()
            raise VisitorError(
                f"{factor_key}: replicate weights sum to zero in a group at replicates {bad}"
            )
        numer = values[mask].T @ group_weights
        means[:, group, :] = (numer / denom).T
    return means


def _equality_contrast(n_groups: int, n_activities: int) -> np.ndarray:
    """Each later group's activity mean minus the first group's."""
    width = n_groups * n_activities
    contrast = np.zeros((n_activities * (n_groups - 1), width), dtype=float)
    row = 0
    for activity in range(n_activities):
        for group in range(1, n_groups):
            contrast[row, group * n_activities + activity] = 1.0
            contrast[row, activity] = -1.0
            row += 1
    return contrast


def _wald_statistic(delta: np.ndarray, variance: np.ndarray) -> tuple[float, int, float]:
    rank = int(np.linalg.matrix_rank(variance))
    if np.allclose(delta, 0, rtol=0, atol=1e-8):
        return 0.0, rank, 1.0
    if rank == 0:
        return math.inf, 0, 0.0
    solved = np.linalg.pinv(variance, rcond=1e-10) @ delta
    wald = float(delta @ solved)
    if wald < 0:
        wald = 0.0
    return wald, rank, chi2_sf(wald, rank)


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _check_test_inputs(
    adults: pd.DataFrame,
    replicate_weights: np.ndarray,
    *,
    alpha: float,
    min_n: int,
) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if min_n < 1:
        raise ValueError("min_n must be positive")
    missing = [column for column in prepared_columns() if column not in adults.columns]
    if missing:
        raise VisitorError("population table is missing columns: " + ", ".join(missing))
    if replicate_weights.ndim != 2 or replicate_weights.shape[0] != len(adults):
        raise VisitorError("replicate weights do not align with the adult table")
    if replicate_weights.shape[1] < 1:
        raise VisitorError("replicate weights need at least one replicate")
    if not np.isfinite(replicate_weights).all() or np.any(replicate_weights < 0):
        raise VisitorError("replicate weights must be finite and nonnegative")


def _manifest(
    adults: pd.DataFrame,
    replicate_weights: np.ndarray,
    factors: list[dict[str, Any]],
    *,
    alpha: float,
    min_n: int,
    sample: dict[str, int] | None,
) -> dict[str, Any]:
    n_replicates = int(replicate_weights.shape[1])
    counts = {
        "n_adults": int(len(adults)),
        "n_respondents": None if sample is None else sample.get("n_respondents"),
        "n_under_18": None if sample is None else sample.get("n_under_18"),
        "n_missing_age": None if sample is None else sample.get("n_missing_age"),
        "n_adults_nonpositive_weight": None
        if sample is None
        else sample.get("n_adults_nonpositive_weight"),
    }
    return {
        "source": "ATUS 2023 activity summary, adults with TEAGE >= 18",
        "weight": "TUFINLWGT",
        "variance": (
            f"({VARIANCE_NUMERATOR}/{n_replicates}) * sum((Y_r - Y)^2), "
            "ATUS User's Guide chapter 7"
        ),
        "n_replicates": n_replicates,
        "alpha": alpha,
        "min_group_n": min_n,
        "outcomes": list(ACTIVITY_LABELS),
        "rule": (
            "每个划分各自做复权 Wald 检验。联合检验 p 低于 "
            f"{alpha}，并且 11 项主要活动里至少一项在 Holm 校正后仍然显著，才写入文件夹。"
            f"未加权人数少于 {min_n} 的水平不进入检验。"
            "划分之间不再做多重校正。"
        ),
        **counts,
        "factors": factors,
    }


def _employment_note(base: str, unemployed_n: int | None, min_n: int) -> str:
    if unemployed_n is None:
        return base
    if unemployed_n < min_n:
        return (
            f"{base} 成年失业（TELFS 为 3 或 4）有 {unemployed_n} 人，"
            f"少于每组最少 {min_n} 人，所以留在未就业里。"
        )
    return f"{base} 成年失业（TELFS 为 3 或 4）有 {unemployed_n} 人，这一划分仍按全职、兼职和未就业切开。"


def _load_population_groups(source: Path) -> list[tuple[PopulationFactor, str, pd.DataFrame]]:
    loaded: list[tuple[PopulationFactor, str, pd.DataFrame]] = []
    columns = prepared_columns()
    for factor in POPULATION_FACTORS:
        for level in factor.levels:
            path = source / factor.key / f"{level}.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise VisitorError(f"{path} is missing columns: " + ", ".join(missing))
            if frame.empty:
                raise VisitorError(f"{path} has no adults")
            frame = frame.sort_values("TUCASEID").reset_index(drop=True)
            if frame["TUCASEID"].duplicated().any():
                raise VisitorError(f"{path} has duplicate TUCASEID values")
            loaded.append((factor, level, frame))
    return loaded


def _population_visitor_record(row: dict[str, Any], factor: str, level: str) -> dict[str, Any]:
    case_id = int(row["TUCASEID"])
    age = int(row["TEAGE"])
    labels = {key: _text_label(row.get(key)) for key in (item.key for item in POPULATION_FACTORS)}
    routine = {name: float(row[name]) for name in ACTIVITY_LABELS}
    visitor_id = f"visitor-{factor}-{level}-{case_id}"
    text = population_persona_text(
        age=age,
        sex=labels["sex"],
        education=labels["education_level"],
        labels=labels,
        routine=routine,
    )
    persona = PersonaCard(
        id=visitor_id,
        condition="atus_population",
        text=text,
        age=age,
        age_band=labels["age_band"],
        sex=labels["sex"],
        state=None,
        education_level=labels["education_level"],
        occupation=labels["employment"],
        marital_status=labels["partner"],
    )
    return {
        "id": visitor_id,
        "source": "atus-population",
        "persona": persona.to_dict(),
        "routine": routine,
        "demographics": {"TUCASEID": case_id, "age": age, **labels},
    }


def _population_example(
    factor: PopulationFactor,
    level: str,
    records: list[dict[str, Any]],
) -> str:
    person = records[0]
    return "\n".join(
        [
            f"# {factor.label} / {LEVEL_TITLES.get(level, level)}",
            "",
            f"- Factor: `{factor.key}`",
            f"- Level: `{level}`",
            f"- Visitor id: `{person['id']}`",
            f"- People in this group: {len(records)}",
            "",
            "## Persona card",
            "",
            "```",
            person["persona"]["text"],
            "```",
            "",
        ]
    )


def _text_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _reset_visitor_output(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)


def _reset_output(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for child in dest.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name in {"manifest.json", "adults.csv"}:
            child.unlink()


def _finite(value: float) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return round(number, 6)
    return None


def _upper_gamma(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("gamma arguments must be positive")
    if x == 0:
        return 1.0
    if x < a + 1:
        return 1.0 - _lower_gamma_series(a, x)
    return _upper_gamma_fraction(a, x)


def _lower_gamma_series(a: float, x: float) -> float:
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(400):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _upper_gamma_fraction(a: float, x: float) -> float:
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 400):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    log_prefix = -x + a * math.log(x) - gln
    if log_prefix < -700:
        return 0.0
    return math.exp(log_prefix) * h
