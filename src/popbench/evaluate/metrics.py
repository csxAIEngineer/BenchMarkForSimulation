"""Scores against human shares and the wave-4 test-retest ceiling.

Each function reads saved answers. None of them call a model. A uniform random
chooser is the floor on closed-ended items.
"""

from __future__ import annotations

import math


def accuracy(predicted: list[str], human: list[str]) -> float:
    """Closed-ended exact-match rate."""
    if len(predicted) != len(human) or not predicted:
        raise ValueError("predicted and human answers must be the same non-empty length")
    hits = sum(a == b for a, b in zip(predicted, human))
    return hits / len(predicted)


def _as_shares(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("share vector is empty")
    if any(value < 0 for value in values):
        raise ValueError("shares cannot be negative")
    total = float(sum(values))
    if total <= 0:
        raise ValueError("shares must sum to a positive number")
    return [value / total for value in values]


def total_variation_distance(simulated_shares: list[float], human_shares: list[float]) -> float:
    """Total variation distance between two answer distributions."""
    if len(simulated_shares) != len(human_shares):
        raise ValueError("share vectors must have the same length")
    left = _as_shares(simulated_shares)
    right = _as_shares(human_shares)
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def jensen_shannon(simulated_shares: list[float], human_shares: list[float]) -> float:
    """Jensen-Shannon divergence, in bits, between two answer distributions."""
    if len(simulated_shares) != len(human_shares):
        raise ValueError("share vectors must have the same length")
    left = _as_shares(simulated_shares)
    right = _as_shares(human_shares)
    midpoint = [0.5 * (a + b) for a, b in zip(left, right)]

    def _kl(source: list[float], target: list[float]) -> float:
        total = 0.0
        for p_value, q_value in zip(source, target):
            if p_value > 0:
                total += p_value * math.log(p_value / q_value, 2)
        return total

    return 0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint)


def ordinal_mae(predicted: list[float], human: list[float]) -> float:
    """Mean absolute error on ordinal or numeric answers."""
    if len(predicted) != len(human) or not predicted:
        raise ValueError("predicted and human values must be the same non-empty length")
    return sum(abs(a - b) for a, b in zip(predicted, human)) / len(predicted)


def subgroup_tvd(
    simulated_shares: dict[str, list[float]],
    human_shares: dict[str, list[float]],
) -> dict[str, float]:
    """Total variation distance inside demographic subgroups."""
    raise NotImplementedError


def closeness_score(observed: float, *, perfect: float, random_floor: float) -> float:
    """Map one metric onto 0–100.

    100 means the value matches real humans. 0 means it matches a uniform
    random answer on the same scale. Values past the random floor stay at 0.
    """
    gap = abs(perfect - random_floor)
    if gap == 0:
        return 100.0 if observed == perfect else 0.0
    raw = 1 - abs(observed - perfect) / gap
    return max(0.0, min(100.0, 100.0 * raw))


def treatment_effect_error(simulated_ate: float, human_ate: float) -> float:
    """Absolute error of a between-subject average treatment effect."""
    return abs(simulated_ate - human_ate)


def wasserstein_1d(simulated_shares: list[float], human_shares: list[float]) -> float:
    """1-Wasserstein distance on consecutive integer options.

    OpinionQA maps ordinal choices to 1..N and uses this distance because a
    one-step Likert miss is closer than a miss at the other end of the scale.
    """
    if len(simulated_shares) != len(human_shares):
        raise ValueError("share vectors must have the same length")
    left = _as_shares(simulated_shares)
    right = _as_shares(human_shares)
    cdf_left = 0.0
    cdf_right = 0.0
    distance = 0.0
    for simulated, human in zip(left, right):
        cdf_left += simulated
        cdf_right += human
        distance += abs(cdf_left - cdf_right)
    return distance


def kolmogorov_smirnov(simulated_shares: list[float], human_shares: list[float]) -> float:
    """Kolmogorov-Smirnov distance between two ordinal answer distributions."""
    if len(simulated_shares) != len(human_shares):
        raise ValueError("share vectors must have the same length")
    left = _as_shares(simulated_shares)
    right = _as_shares(human_shares)
    cdf_left = 0.0
    cdf_right = 0.0
    gap = 0.0
    for simulated, human in zip(left, right):
        cdf_left += simulated
        cdf_right += human
        gap = max(gap, abs(cdf_left - cdf_right))
    return gap


def opinion_alignment(simulated_shares: list[float], human_shares: list[float]) -> float:
    """OpinionQA alignment: 1 minus Wasserstein divided by the scale span."""
    span = len(simulated_shares) - 1
    if span < 1:
        raise ValueError("alignment needs at least two options")
    return 1 - wasserstein_1d(simulated_shares, human_shares) / span


def ordinal_accuracy(predicted: list[float], human: list[float], *, span: float) -> float:
    """Twin-2K accuracy: 1 minus absolute error divided by the answer range.

    A perfect match is 1. The worst miss on the scale is 0.
    """
    if len(predicted) != len(human) or not predicted:
        raise ValueError("predicted and human values must be the same non-empty length")
    if span <= 0:
        raise ValueError("span must be positive")
    return sum(1 - abs(left - right) / span for left, right in zip(predicted, human)) / len(predicted)


def pearson(left: list[float], right: list[float]) -> float | None:
    """Pearson correlation. None when either side has no variance."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation needs two equal-length series of at least two points")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    scale_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    scale_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if scale_left == 0 or scale_right == 0:
        return None
    return covariance / (scale_left * scale_right)


def mean_squared_error(left: list[float], right: list[float]) -> float:
    """Mean squared error between two equal-length series."""
    if len(left) != len(right) or not left:
        raise ValueError("MSE needs two equal-length non-empty series")
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


def distribution_correlation(
    simulated_shares: list[float],
    human_shares: list[float],
) -> float | None:
    """Pearson correlation between two answer-share (or score) vectors.

    Used for behavioral fit: how closely the simulated distribution tracks the
    human / ATUS baseline shape.
    """
    if len(simulated_shares) != len(human_shares):
        raise ValueError("share vectors must have the same length")
    left = _as_shares(simulated_shares)
    right = _as_shares(human_shares)
    return pearson(left, right)


def response_amplitude_coverage(
    simulated_codes: list[float],
    human_codes: list[float],
) -> float:
    """Fraction of distinct human response codes also produced by simulation.

    On an ordinal scale this is support coverage of the human response
    amplitude: ``|sim ∩ human| / |human|`` over the observed code sets.
    """
    if not simulated_codes or not human_codes:
        raise ValueError("amplitude coverage needs non-empty code lists")
    human_set = {int(round(value)) for value in human_codes}
    sim_set = {int(round(value)) for value in simulated_codes}
    if not human_set:
        return 1.0 if not sim_set else 0.0
    return len(human_set & sim_set) / len(human_set)


def response_amplitude_ratio(
    simulated_codes: list[float],
    human_codes: list[float],
) -> float:
    """How much of the human ordinal span the simulation spans (capped at 1)."""
    if not simulated_codes or not human_codes:
        raise ValueError("amplitude ratio needs non-empty code lists")
    human_span = max(human_codes) - min(human_codes)
    sim_span = max(simulated_codes) - min(simulated_codes)
    if human_span <= 0:
        return 1.0 if sim_span <= 0 else 0.0
    return min(1.0, sim_span / human_span)


def cross_model_variance(series_by_model: list[list[float]]) -> dict[str, float | None]:
    """Robustness audit across models on one aligned metric series.

    Each inner list is one model's values (e.g. per-item mean codes or flattened
    shares). Returns mean per-position variance, mean std, and the max std.
    """
    if len(series_by_model) < 2:
        raise ValueError("cross-model variance needs at least two model series")
    width = len(series_by_model[0])
    if width == 0 or any(len(series) != width for series in series_by_model):
        raise ValueError("model series must be rectangular and non-empty")
    variances: list[float] = []
    stds: list[float] = []
    for column in range(width):
        values = [series[column] for series in series_by_model]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        variances.append(var)
        stds.append(math.sqrt(var))
    return {
        "n_models": float(len(series_by_model)),
        "n_positions": float(width),
        "mean_variance": sum(variances) / len(variances),
        "mean_std": sum(stds) / len(stds),
        "max_std": max(stds),
    }


def population_std(values: list[float]) -> float | None:
    """Standard deviation with the same divisor the human baseline uses."""
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def cronbach_alpha(rows: list[list[float]]) -> float | None:
    """Internal consistency of one personality scale. Rows are people, columns are items."""
    if len(rows) < 2 or not rows[0] or len(rows[0]) < 2:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Cronbach rows must be rectangular")
    item_variance = 0.0
    for column in range(width):
        item_variance += _sample_variance([row[column] for row in rows])
    totals = [sum(row) for row in rows]
    total_variance = _sample_variance(totals)
    if total_variance == 0:
        return None
    return (width / (width - 1)) * (1 - item_variance / total_variance)


def quadratic_weighted_kappa(predicted: list[int], human: list[int], *, categories: int) -> float | None:
    """Ordinal agreement. 1 is perfect, 0 matches chance, negative is worse than chance."""
    if len(predicted) != len(human) or not predicted:
        raise ValueError("predicted and human codes must be the same non-empty length")
    if categories < 2:
        raise ValueError("kappa needs at least two categories")
    observed = [[0.0 for _ in range(categories)] for _ in range(categories)]
    count = 0
    for left, right in zip(predicted, human):
        if not 1 <= int(left) <= categories or not 1 <= int(right) <= categories:
            continue
        observed[int(left) - 1][int(right) - 1] += 1
        count += 1
    if count == 0:
        return None
    row_mass = [sum(row) for row in observed]
    column_mass = [sum(observed[row][column] for row in range(categories)) for column in range(categories)]
    span = (categories - 1) ** 2
    weighted_observed = 0.0
    weighted_expected = 0.0
    for row in range(categories):
        for column in range(categories):
            weight = ((row - column) ** 2) / span
            weighted_observed += weight * observed[row][column] / count
            weighted_expected += weight * (row_mass[row] / count) * (column_mass[column] / count)
    if weighted_expected == 0:
        return None
    return 1 - weighted_observed / weighted_expected


def _sample_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)
