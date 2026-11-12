"""Arm B summaries: observation-equal d_buy and product-equal conditional P50."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from popbench.evaluate.arm_a import interpolate_p50
from popbench.evaluate.metrics import pearson, sample_std


def _mapped(rows: Sequence[Mapping[str, object]], state: str | None = None, tier: int | None = None) -> list[int]:
    codes: list[int] = []
    for row in rows:
        if state is not None and row.get("need_state") != state:
            continue
        if tier is not None and int(row.get("tier")) != tier:
            continue
        code = row.get("buy")
        if isinstance(code, int):
            codes.append(code)
    return codes


def observation_p_buy(rows: Sequence[Mapping[str, object]], state: str, tier: int | None = None) -> float | None:
    """Mean buy indicator among mapped offers. Unmapped rows are not coded as zero."""
    codes = _mapped(rows, state, tier)
    if not codes:
        return None
    return sum(codes) / len(codes)


def d_buy(rows: Sequence[Mapping[str, object]], tier: int | None = None) -> float | None:
    """mean(P_buy | high) - mean(P_buy | low)."""
    high = observation_p_buy(rows, "high", tier)
    low = observation_p_buy(rows, "low", tier)
    if high is None or low is None:
        return None
    return high - low


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    low = int(position)
    high = int(position) if position == int(position) else int(position) + 1
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def bootstrap_d_buy(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 42,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Subject-cluster percentile interval for d_buy. Subjects are resampled with replacement."""
    by_subject: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row.get("pid") or row.get("subject_id") or "")].append(row)
    subjects = sorted(by_subject)
    point = d_buy(rows)
    if len(subjects) < 2 or point is None:
        return {"ci_high": None, "ci_low": None, "d_buy": point, "replicates": 0}
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        pool: list[Mapping[str, object]] = []
        for _subject in range(len(subjects)):
            pool.extend(by_subject[subjects[rng.randrange(len(subjects))]])
        value = d_buy(pool)
        if value is not None:
            draws.append(value)
    draws.sort()
    if not draws:
        return {"ci_high": None, "ci_low": None, "d_buy": point, "replicates": 0}
    return {
        "ci_high": _percentile(draws, 0.975),
        "ci_low": _percentile(draws, 0.025),
        "d_buy": point,
        "replicates": len(draws),
    }


def e7_pass(point: float | None, ci_low: float | None, ci_high: float | None) -> bool:
    """True when the high-minus-low purchase gap is positive and the interval excludes 0."""
    if point is None or ci_low is None or ci_high is None:
        return False
    return point > 0 and (ci_high < 0 or ci_low > 0)


def product_equal_p_buy(rows: Sequence[Mapping[str, object]], state: str) -> dict[int, float]:
    """Unweighted mean of within-product purchase rates at each tier, inside one need state."""
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        if row.get("need_state") != state:
            continue
        code = row.get("buy")
        product = row.get("product_id")
        tier = row.get("tier")
        if not isinstance(code, int) or not isinstance(tier, int) or not product:
            continue
        buckets[(str(product), tier)].append(code)
    by_tier: dict[int, list[float]] = defaultdict(list)
    for (_product, tier), codes in buckets.items():
        if codes:
            by_tier[tier].append(sum(codes) / len(codes))
    return {tier: sum(rates) / len(rates) for tier, rates in by_tier.items() if rates}


def condition_p50(rows: Sequence[Mapping[str, object]], state: str) -> dict[str, Any]:
    return interpolate_p50(product_equal_p_buy(rows, state))


def person_rates(
    rows: Sequence[Mapping[str, object]],
    human_codes: Mapping[tuple[str, str], int],
    state: str,
) -> tuple[list[float], list[float]]:
    """Mean purchase rate per person on this condition's triplets. Incomplete people are left out."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("need_state") == state:
            grouped[str(row.get("pid") or row.get("subject_id"))].append(row)
    model_rates: list[float] = []
    human_rates: list[float] = []
    for subject_id, subject_rows in sorted(grouped.items()):
        model_values: list[int] = []
        human_values: list[int] = []
        complete = True
        for row in subject_rows:
            key = (subject_id, str(row.get("question_id")))
            human = human_codes.get(key)
            model = row.get("buy")
            if human is None or not isinstance(model, int):
                complete = False
                break
            human_values.append(human)
            model_values.append(model)
        if not complete or not model_values:
            continue
        model_rates.append(sum(model_values) / len(model_values))
        human_rates.append(sum(human_values) / len(human_values))
    return model_rates, human_rates


def appendix_association(model_rates: Sequence[float], human_rates: Sequence[float]) -> dict[str, float | None]:
    """Pearson and sample-SD ratio. None when the series cannot support the statistic."""
    if len(model_rates) < 2 or len(model_rates) != len(human_rates):
        return {"cross_person_pearson": None, "sd_ratio": None}
    try:
        correlation = pearson(list(model_rates), list(human_rates))
    except ValueError:
        correlation = None
    human_sd = sample_std(list(human_rates))
    model_sd = sample_std(list(model_rates))
    ratio = None if human_sd in (None, 0) or model_sd is None else model_sd / human_sd
    return {"cross_person_pearson": correlation, "sd_ratio": ratio}
