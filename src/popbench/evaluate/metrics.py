"""Scores of saved answers against held-out human answers. No model calls."""

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


def _mode_indices(shares: list[float]) -> set[int]:
    peak = max(shares)
    return {index for index, value in enumerate(shares) if value == peak}


def distribution_top1_accuracy(simulated_shares: list[float], human_shares: list[float]) -> float:
    """1.0 when the modes overlap, else 0.0. Ties match on any shared mode."""
    if len(simulated_shares) != len(human_shares):
        raise ValueError("share vectors must have the same length")
    left = _as_shares(simulated_shares)
    right = _as_shares(human_shares)
    if _mode_indices(left) & _mode_indices(right):
        return 1.0
    return 0.0


def ordinal_mae(predicted: list[float], human: list[float]) -> float:
    """Mean absolute error on numeric answers."""
    if len(predicted) != len(human) or not predicted:
        raise ValueError("predicted and human values must be the same non-empty length")
    return sum(abs(a - b) for a, b in zip(predicted, human)) / len(predicted)


def ordinal_accuracy(predicted: list[float], human: list[float], *, span: float) -> float:
    """1 minus absolute error divided by the answer range."""
    if len(predicted) != len(human) or not predicted:
        raise ValueError("predicted and human values must be the same non-empty length")
    if span <= 0:
        raise ValueError("span must be positive")
    return sum(1 - abs(left - right) / span for left, right in zip(predicted, human)) / len(predicted)


def individual_accuracy(predicted: list[float], human: list[float], *, span: float) -> float:
    """Paired closeness: 1 - MAD/range."""
    return ordinal_accuracy(predicted, human, span=span)


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


def population_std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def sample_std(values: list[float]) -> float | None:
    """Sample standard deviation with ``ddof=1``. None when fewer than two points."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def average_ranks(values: list[float]) -> list[float]:
    """1-based ranks. Ties share the average of the ranks they cover."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    """Spearman rank correlation. None when either ranking has no variance."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman needs two equal-length series of at least two points")
    return pearson(average_ranks(left), average_ranks(right))


def fisher_z(correlation: float) -> float:
    if correlation >= 1:
        correlation = 1 - 1e-12
    elif correlation <= -1:
        correlation = -1 + 1e-12
    return 0.5 * math.log((1 + correlation) / (1 - correlation))


def inverse_fisher_z(z_value: float) -> float:
    return math.tanh(z_value)


def fisher_z_mean_correlation(correlations: list[float]) -> float | None:
    """Average Pearson correlations on the Fisher-z scale, then transform back."""
    zs: list[float] = []
    for value in correlations:
        if value is None:
            continue
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            continue
        zs.append(fisher_z(number))
    if not zs:
        return None
    return inverse_fisher_z(sum(zs) / len(zs))


def glass_delta(simulated: list[float], human: list[float]) -> float | None:
    """(mean_sim - mean_human) / sd_human. None when the human sd is zero."""
    if len(simulated) != len(human) or not human:
        raise ValueError("Glass delta needs two equal-length non-empty series")
    scale = population_std(human)
    if scale is None or scale == 0:
        return None
    return (sum(simulated) / len(simulated) - sum(human) / len(human)) / scale


def sd_ratio(simulated: list[float], human: list[float]) -> float | None:
    """sd_sim / sd_human. None when the human sd is zero."""
    if not simulated or not human:
        raise ValueError("sd ratio needs non-empty series")
    human_sd = population_std(human)
    sim_sd = population_std(simulated)
    if human_sd is None or sim_sd is None or human_sd == 0:
        return None
    return sim_sd / human_sd
