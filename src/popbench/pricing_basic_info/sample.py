"""Deterministic sample-size rule and stratified draw.

The size rule is Cochran's planning formula for a proportion, with the
finite-population correction. It is not a hypothesis test. Within a stratum,
membership is the lowest SHA-256 key, so the same inputs produce the same ids
on any machine that implements SHA-256.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING

from popbench.pricing_basic_info.cards import normalize_cell
from popbench.pricing_basic_info.spec import ID_COLUMN


class SampleError(ValueError):
    """The sample request cannot be applied to this population."""


def cochran_n0(z: str, proportion: str, margin: str) -> int:
    """Smallest integer n0 such that n0 >= z^2 p (1-p) / e^2."""
    z_value = Decimal(z)
    p_value = Decimal(proportion)
    e_value = Decimal(margin)
    if z_value <= 0 or e_value <= 0 or not (Decimal("0") < p_value < Decimal("1")):
        raise SampleError("sample-size rule needs z > 0, 0 < p < 1, and e > 0")
    n0 = (z_value ** 2) * p_value * (Decimal("1") - p_value) / (e_value ** 2)
    return int(n0.to_integral_value(rounding=ROUND_CEILING))


def finite_population_n(n0: int, population: int) -> int:
    """ceil(n0 * N / (N + n0 - 1)), capped at N."""
    if n0 < 1 or population < 1:
        raise SampleError("sample size and population must be positive")
    corrected = _ceil_div(n0 * population, population + n0 - 1)
    return min(corrected, population)


def planned_sample_size(
    population: int,
    *,
    z: str = "1.96",
    proportion: str = "0.5",
    margin: str = "0.10",
    override: int | None = None,
) -> dict[str, object]:
    """Return the default n and the inputs that produced it."""
    n0 = cochran_n0(z, proportion, margin)
    computed = finite_population_n(n0, population)
    chosen = computed if override is None else override
    if chosen < 1:
        raise SampleError("sample size must be a positive integer")
    if chosen > population:
        raise SampleError(f"sample size {chosen} exceeds population {population}")
    return {
        "margin_of_error": margin,
        "method": "cochran_proportion_with_finite_population_correction",
        "n0": n0,
        "override": override,
        "population": population,
        "proportion": proportion,
        "sample_size": chosen,
        "z": z,
    }


def stratified_sample(
    rows: Sequence[Mapping[str, object]],
    *,
    stratify_column: str,
    sample_size: int,
    seed: int,
    dataset: str,
) -> dict[str, object]:
    """Proportional allocation, largest remainder, hash order inside a stratum."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SampleError("seed must be an integer")
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in rows:
        subject_id = str(row.get(ID_COLUMN) or "").strip()
        if not subject_id:
            raise SampleError("row is missing pid")
        if subject_id in seen:
            raise SampleError(f"duplicate pid: {subject_id}")
        seen.add(subject_id)
        stratum, _status = normalize_cell(row.get(stratify_column))
        groups.setdefault(stratum, []).append(subject_id)
    population = len(seen)
    if sample_size > population:
        raise SampleError(f"sample size {sample_size} exceeds population {population}")
    counts = {name: len(ids) for name, ids in groups.items()}
    allocation = _largest_remainder(counts, sample_size)
    picked: list[str] = []
    for stratum in sorted(groups):
        ranked = sorted(
            groups[stratum],
            key=lambda subject_id: (_rank(seed, dataset, stratum, subject_id), subject_id),
        )
        picked.extend(ranked[: allocation[stratum]])
    ordered = sorted(picked, key=lambda subject_id: (_rank(seed, dataset, "selected", subject_id), subject_id))
    return {
        "allocation": {name: allocation[name] for name in sorted(allocation)},
        "dataset": dataset,
        "population_size": population,
        "population_stratum_counts": {name: counts[name] for name in sorted(counts)},
        "sample_size": len(ordered),
        "seed": seed,
        "stratify_column": stratify_column,
        "subject_ids": ordered,
    }


def _largest_remainder(counts: Mapping[str, int], sample_size: int) -> dict[str, int]:
    names = sorted(counts)
    total = sum(counts[name] for name in names)
    if total < 1 or sample_size < 1:
        raise SampleError("allocation needs a positive population and sample")
    base: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for name in names:
        numer = counts[name] * sample_size
        base[name] = numer // total
        remainders.append((numer % total, name))
    left = sample_size - sum(base.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _remainder, name in remainders:
        if left == 0:
            break
        if base[name] < counts[name]:
            base[name] += 1
            left -= 1
    if left != 0:
        raise SampleError("largest-remainder allocation could not place every seat")
    return base


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _rank(seed: int, dataset: str, stratum: str, subject_id: str) -> str:
    payload = f"{seed}\n{dataset}\n{stratum}\n{subject_id}".encode()
    return hashlib.sha256(payload).hexdigest()
