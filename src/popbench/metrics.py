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


def treatment_effect_error(simulated_ate: float, human_ate: float) -> float:
    """Absolute error of a between-subject average treatment effect."""
    raise NotImplementedError
