"""Clustered logistic dose-response summaries for the full-sample need arms.

``pct`` is ``tier * 20``. Human slope ``-0.01486`` is a fixed reference.
Bootstrap resamples people, not rows.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

HUMAN_BETA = -0.01486
DOSE_LEVEL = {"N1": 1, "N2": 2, "N3": 3, "N4": 4}
DOSE_LENGTH = {"N1": 50, "N2": 49, "N3": 48, "N4": 51, "BUD": 52}


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def fit_logit(y: np.ndarray, x_matrix: np.ndarray) -> np.ndarray | None:
    """Bernoulli IRLS. None when the design is singular."""
    beta = np.zeros(x_matrix.shape[1], dtype=float)
    for _ in range(40):
        eta = x_matrix @ beta
        probability = _sigmoid(eta)
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        working = eta + (y - probability) / weight
        weighted = x_matrix * weight[:, None]
        try:
            updated = np.linalg.solve(x_matrix.T @ weighted, weighted.T @ working)
        except np.linalg.LinAlgError:
            return None
        if float(np.max(np.abs(updated - beta))) < 1e-10:
            return updated
        beta = updated
    return beta


def _design(pct: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([np.ones(len(pct)), *columns])


def predict(alpha: float, beta: float, pct: float) -> float:
    eta = alpha + beta * pct
    if eta >= 0:
        return 1.0 / (1.0 + math.exp(-eta))
    exp_eta = math.exp(eta)
    return exp_eta / (1.0 + exp_eta)


def _records(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    kept = []
    for row in rows:
        buy = row.get("buy")
        tier = row.get("tier")
        if isinstance(buy, int) and isinstance(tier, int):
            kept.append(row)
    return kept


def _by_person(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in _records(rows):
        grouped[str(row.get("pid") or row.get("subject_id"))].append(row)
    return grouped


def _xy(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([int(row["buy"]) for row in rows], dtype=float)
    pct = np.array([int(row["tier"]) * 20 for row in rows], dtype=float)
    return y, pct


def arm_fit(rows: Sequence[Mapping[str, object]]) -> dict[str, Any] | None:
    """Fit logit(buy) = alpha + beta * pct on mapped rows."""
    usable = _records(rows)
    if len(usable) < 2:
        return None
    y_values, pct = _xy(usable)
    beta = fit_logit(y_values, _design(pct, [pct]))
    if beta is None:
        return None
    alpha = float(beta[0])
    slope = float(beta[1])
    return {
        "alpha": alpha,
        "beta": slope,
        "beta_over_human": abs(slope) / abs(HUMAN_BETA),
        "n": len(usable),
        "p0": predict(alpha, slope, 0),
        "p100": predict(alpha, slope, 100),
        "p200": predict(alpha, slope, 200),
    }


def _resample(grouped: Mapping[str, Sequence[Mapping[str, object]]], rng: random.Random) -> list[Mapping[str, object]]:
    people = sorted(grouped)
    drawn: list[Mapping[str, object]] = []
    for _ in people:
        drawn.extend(grouped[people[rng.randrange(len(people))]])
    return drawn


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def bootstrap_arm(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 42,
    replicates: int = 1000,
) -> dict[str, Any]:
    point = arm_fit(rows)
    grouped = _by_person(rows)
    if point is None or len(grouped) < 2:
        return {"fit": point, "se_alpha": None, "se_beta": None, "replicates": 0}
    rng = random.Random(seed)
    alphas: list[float] = []
    slopes: list[float] = []
    for _ in range(replicates):
        fit = arm_fit(_resample(grouped, rng))
        if fit is None:
            continue
        alphas.append(float(fit["alpha"]))
        slopes.append(float(fit["beta"]))
    return {
        "fit": point,
        "replicates": len(slopes),
        "se_alpha": _sample_sd(alphas),
        "se_beta": _sample_sd(slopes),
    }


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def two_sided_p(estimate: float | None, standard_error: float | None) -> float | None:
    if estimate is None or standard_error is None or standard_error == 0:
        return None
    z_value = abs(estimate) / standard_error
    return math.erfc(z_value / math.sqrt(2.0))


def _stack(parts: Sequence[tuple[str, Sequence[Mapping[str, object]]]]) -> list[dict[str, object]]:
    stacked = []
    for arm, rows in parts:
        for row in _records(rows):
            item = dict(row)
            item["dose"] = DOSE_LEVEL[arm]
            item["length"] = DOSE_LENGTH[arm]
            stacked.append(item)
    return stacked


def stack_dose_rows(rows_by_arm: Mapping[str, Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    return _stack([(arm, rows_by_arm[arm]) for arm in ("N1", "N2", "N3", "N4") if arm in rows_by_arm])


def pooled_fit(rows_by_arm: Mapping[str, Sequence[Mapping[str, object]]], columns: Sequence[str]) -> dict[str, float] | None:
    stacked = _stack([(arm, rows_by_arm[arm]) for arm in ("N1", "N2", "N3", "N4") if arm in rows_by_arm])
    if len(stacked) < 8:
        return None
    y_values, matrix = _pooled_matrix(stacked, columns)
    beta = fit_logit(y_values, matrix)
    if beta is None:
        return None
    names = ["alpha", *columns]
    # matrix columns are intercept, pct, then the named extras. pct is always included.
    names = ["alpha", "pct", *columns]
    return {name: float(value) for name, value in zip(names, beta)}


def _pooled_matrix(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    y_values = np.array([int(row["buy"]) for row in rows], dtype=float)
    pct = np.array([int(row["tier"]) * 20 for row in rows], dtype=float)
    pieces = [np.ones(len(rows)), pct]
    dose = np.array([float(row["dose"]) for row in rows])
    for name in columns:
        if name == "dose":
            pieces.append(dose)
        elif name == "dose2":
            pieces.append(dose ** 2)
        elif name == "len":
            pieces.append(np.array([float(row["length"]) for row in rows]))
        elif name == "pct_dose":
            pieces.append(pct * dose)
        else:
            raise ValueError(name)
    return y_values, np.column_stack(pieces)


def _named_matrix(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    y_values = np.array([int(row["buy"]) for row in rows], dtype=float)
    pct = np.array([int(row["tier"]) * 20 for row in rows], dtype=float)
    dose = np.array([float(row["dose"]) for row in rows]) if any(name.startswith("dose") or name == "pct_dose" for name in columns) else None
    pieces = [np.ones(len(rows))]
    for name in columns:
        if name == "pct":
            pieces.append(pct)
        elif name == "dose":
            pieces.append(dose)
        elif name == "dose2":
            pieces.append(dose ** 2)
        elif name == "len":
            pieces.append(np.array([float(row["length"]) for row in rows]))
        elif name == "pct_dose":
            pieces.append(pct * dose)
        else:
            raise ValueError(name)
    return y_values, np.column_stack(pieces)


def bernoulli_aic(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> float | None:
    """Observation-level AIC. Descriptive only; rows are not independent people."""
    if len(rows) < 4:
        return None
    y_values, matrix = _named_matrix(rows, columns)
    beta = fit_logit(y_values, matrix)
    if beta is None:
        return None
    probability = np.clip(_sigmoid(matrix @ beta), 1e-12, 1 - 1e-12)
    loglik = float(np.sum(y_values * np.log(probability) + (1 - y_values) * np.log(1 - probability)))
    return 2 * matrix.shape[1] - 2 * loglik


def bootstrap_coefficients(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, object]]],
    columns: Sequence[str],
    *,
    seed: int = 42,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Cluster bootstrap of a pooled dose model. People are drawn once and applied to every arm."""
    grouped = {arm: _by_person(rows) for arm, rows in rows_by_arm.items()}
    people = sorted(set().union(*[set(group) for group in grouped.values()]))
    point = pooled_fit(rows_by_arm, columns)
    if point is None or len(people) < 2:
        return {"coefficients": point, "se": {}, "p": {}, "replicates": 0}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(replicates):
        chosen = [people[rng.randrange(len(people))] for _ in people]
        sample = {}
        for arm, group in grouped.items():
            sample[arm] = [row for person in chosen for row in group.get(person, [])]
        fit = pooled_fit(sample, columns)
        if fit is None:
            continue
        for name, value in fit.items():
            draws[name].append(value)
    standard_errors = {name: _sample_sd(values) for name, values in draws.items()}
    p_values = {name: two_sided_p(point[name], standard_errors[name]) for name in point}
    return {
        "coefficients": point,
        "p": p_values,
        "replicates": min(len(values) for values in draws.values()) if draws else 0,
        "se": standard_errors,
    }


def bootstrap_curve_contrasts(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, object]]],
    pairs: Sequence[tuple[str, str]],
    *,
    seed: int = 42,
    replicates: int = 1000,
) -> list[dict[str, Any]]:
    """Percentile intervals for later minus earlier p(100), resampling people jointly."""
    grouped = {arm: _by_person(rows) for arm, rows in rows_by_arm.items()}
    people = sorted(set().union(*[set(group) for group in grouped.values()]))
    points = {}
    for left, right in pairs:
        left_fit = arm_fit(rows_by_arm[left])
        right_fit = arm_fit(rows_by_arm[right])
        if left_fit is None or right_fit is None:
            points[(left, right)] = None
        else:
            points[(left, right)] = float(right_fit["p100"]) - float(left_fit["p100"])
    rng = random.Random(seed)
    clouds: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    for _ in range(replicates):
        chosen = [people[rng.randrange(len(people))] for _ in people]
        fits = {}
        for arm, group in grouped.items():
            fits[arm] = arm_fit([row for person in chosen for row in group.get(person, [])])
        for pair in pairs:
            left_fit = fits.get(pair[0])
            right_fit = fits.get(pair[1])
            if left_fit is None or right_fit is None:
                continue
            clouds[pair].append(float(right_fit["p100"]) - float(left_fit["p100"]))
    results = []
    for pair in pairs:
        cloud = clouds[pair]
        estimate = points[pair]
        results.append(
            {
                "ci_high": None if not cloud or estimate is None else _percentile(cloud, 0.975),
                "ci_low": None if not cloud or estimate is None else _percentile(cloud, 0.025),
                "contrast": f"{pair[0]}->{pair[1]}",
                "difference": estimate,
            }
        )
    return results


def bootstrap_budget_gap(
    n4_rows: Sequence[Mapping[str, object]],
    bud_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 42,
    replicates: int = 1000,
) -> dict[str, Any]:
    """BUD minus N4 for p(0), p(100), p(200), and beta."""
    left = arm_fit(n4_rows)
    right = arm_fit(bud_rows)
    names = ("p0", "p100", "p200", "beta")
    if left is None or right is None:
        return {f"delta_{name}": None for name in names}
    point = {name: float(right[name]) - float(left[name]) for name in names}
    grouped_left = _by_person(n4_rows)
    grouped_right = _by_person(bud_rows)
    people = sorted(set(grouped_left) | set(grouped_right))
    rng = random.Random(seed)
    clouds: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(replicates):
        chosen = [people[rng.randrange(len(people))] for _ in people]
        fit_left = arm_fit([row for person in chosen for row in grouped_left.get(person, [])])
        fit_right = arm_fit([row for person in chosen for row in grouped_right.get(person, [])])
        if fit_left is None or fit_right is None:
            continue
        for name in names:
            clouds[name].append(float(fit_right[name]) - float(fit_left[name]))
    result: dict[str, Any] = {}
    for name in names:
        result[f"delta_{name}"] = point[name]
        result[f"delta_{name}_ci_low"] = None if not clouds[name] else _percentile(clouds[name], 0.025)
        result[f"delta_{name}_ci_high"] = None if not clouds[name] else _percentile(clouds[name], 0.975)
    return result
