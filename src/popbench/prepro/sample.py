"""Deterministic sample size and multi-margin Hamilton allocation.

One stratifier uses largest-remainder (Hamilton) targets. Several stratifiers
each keep those same one-way targets. A deterministic flow fills the joint
cells without exceeding the population cell counts, then unit swaps move the
table toward the proportional cell quotas while leaving the margins unchanged.

A level whose Hamilton target is 0 gets no seats. That is a property of
proportional allocation, not evidence that every rare level is covered.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING

from popbench.prepro.cards import normalize_cell
from popbench.prepro.spec import ID_COLUMN

_JOIN = "\x1f"
RARE_LEVEL_POLICY = (
    "Hamilton largest-remainder gives a level zero seats when its quota is "
    "below 1 and it loses the remainder contest. Proportional allocation does "
    "not guarantee that every rare level appears in the sample."
)


class SampleError(ValueError):
    """The sample request cannot be applied to this population."""


def require_int(name: str, value: object, *, allow_none: bool = False) -> int | None:
    """Reject bool. ``True`` is an ``int`` subclass and must not become 1."""
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SampleError(f"{name} must be an integer")
    return value


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
    n0 = require_int("n0", n0)
    population = require_int("population", population)
    assert n0 is not None and population is not None
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
    population = require_int("population", population)
    override = require_int("sample_size", override, allow_none=True)
    assert population is not None
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
    stratify_columns: Sequence[str],
    sample_size: int,
    seed: int,
    dataset: str,
) -> dict[str, object]:
    """Draw ``sample_size`` ids with every stratifier margin on its Hamilton target."""
    sample_size = require_int("sample_size", sample_size)
    seed = require_int("seed", seed)
    assert sample_size is not None and seed is not None
    if not stratify_columns:
        raise SampleError("at least one stratifier is required")
    groups, marginal_counts = _group_rows(rows, stratify_columns)
    population = sum(len(ids) for ids in groups.values())
    if sample_size > population:
        raise SampleError(f"sample size {sample_size} exceeds population {population}")
    marginal_targets = {
        column: _largest_remainder(marginal_counts[column], sample_size) for column in stratify_columns
    }
    allocation, joint_stats = _allocate_joint(groups, marginal_targets, sample_size, stratify_columns)
    _assert_hamilton_margins(allocation, groups, marginal_targets, sample_size, stratify_columns)
    picked: list[str] = []
    for cell in sorted(groups):
        ranked = sorted(
            groups[cell],
            key=lambda subject_id: (_rank(seed, dataset, cell, subject_id), subject_id),
        )
        picked.extend(ranked[: allocation[cell]])
    ordered = sorted(picked, key=lambda subject_id: (_rank(seed, dataset, "selected", subject_id), subject_id))
    if len(ordered) != sample_size:
        raise SampleError(f"selected {len(ordered)} ids, expected {sample_size}")
    return {
        "allocation": {cell: allocation[cell] for cell in sorted(allocation)},
        "dataset": dataset,
        "joint_quota_adjustment": joint_stats,
        "marginal_allocation": {column: marginal_targets[column] for column in stratify_columns},
        "marginal_population_counts": {column: marginal_counts[column] for column in stratify_columns},
        "margin_balance": _margin_balance(marginal_targets, marginal_counts, sample_size, population),
        "population_size": population,
        "population_stratum_counts": {cell: len(groups[cell]) for cell in sorted(groups)},
        "rare_level_policy": RARE_LEVEL_POLICY,
        "sample_size": len(ordered),
        "seed": seed,
        "stratify_columns": list(stratify_columns),
        "subject_ids": ordered,
    }


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    marginal: dict[str, dict[str, int]] = {column: defaultdict(int) for column in columns}
    seen: set[str] = set()
    for row in rows:
        subject_id = str(row.get(ID_COLUMN) or "").strip()
        if not subject_id:
            raise SampleError("row is missing pid")
        if subject_id in seen:
            raise SampleError(f"duplicate pid: {subject_id}")
        seen.add(subject_id)
        values = []
        for column in columns:
            if column not in row:
                raise SampleError(f"{subject_id} is missing stratifier column {column}")
            value, _status = normalize_cell(row.get(column))
            values.append(value)
            marginal[column][value] += 1
        groups[_cell_key(values)].append(subject_id)
    return groups, {column: dict(counts) for column, counts in marginal.items()}


def _allocate_joint(
    groups: Mapping[str, Sequence[str]],
    marginal_targets: Mapping[str, Mapping[str, int]],
    sample_size: int,
    columns: Sequence[str],
) -> tuple[dict[str, int], dict[str, object]]:
    if len(columns) == 1:
        targets = marginal_targets[columns[0]]
        allocation = {cell: targets.get(cell, 0) for cell in groups}
        if sum(allocation.values()) != sample_size:
            raise SampleError("single-margin allocation does not sum to the sample size")
        return allocation, {
            "applied": False,
            "exact_cell_quota": False,
            "note": "One stratifier has no joint table. Hamilton margins are the whole allocation.",
        }
    if len(columns) != 2:
        raise SampleError("this builder balances one or two stratifiers")
    row_name, col_name = columns
    caps: dict[tuple[str, str], int] = {}
    for cell, ids in groups.items():
        row_value, col_value = cell.split(_JOIN, 1)
        caps[(row_value, col_value)] = len(ids)
    table = _feasible_table(dict(marginal_targets[row_name]), dict(marginal_targets[col_name]), caps, sample_size)
    table, joint_stats = _repair_toward_quota(
        table, caps, sample_size, sum(len(ids) for ids in groups.values())
    )
    allocation = {_cell_key([row_value, col_value]): count for (row_value, col_value), count in table.items()}
    for cell in groups:
        allocation.setdefault(cell, 0)
    return allocation, joint_stats


def _feasible_table(
    row_targets: dict[str, int],
    col_targets: dict[str, int],
    caps: Mapping[tuple[str, str], int],
    sample_size: int,
) -> dict[tuple[str, str], int]:
    """Integer table with the Hamilton margins and cell caps. Edges are sorted."""
    rows = sorted(row_targets)
    cols = sorted(col_targets)
    residual: dict[tuple[str, str], int] = {}
    for row_value in rows:
        residual[("S", f"R:{row_value}")] = row_targets[row_value]
        for col_value in cols:
            residual[(f"R:{row_value}", f"C:{col_value}")] = caps.get((row_value, col_value), 0)
    for col_value in cols:
        residual[(f"C:{col_value}", "T")] = col_targets[col_value]
    sent = 0
    while sent < sample_size:
        path = _bfs(residual, rows, cols)
        if path is None:
            raise SampleError(
                "Hamilton margins are not feasible under the joint cell counts. "
                "Refusing to drop a margin target."
            )
        _augment(residual, path)
        sent += 1
    table = {(row_value, col_value): 0 for row_value in rows for col_value in cols}
    for row_value in rows:
        for col_value in cols:
            cap = caps.get((row_value, col_value), 0)
            table[(row_value, col_value)] = cap - residual[(f"R:{row_value}", f"C:{col_value}")]
    return table


def _bfs(
    residual: Mapping[tuple[str, str], int],
    rows: Sequence[str],
    cols: Sequence[str],
) -> list[tuple[str, str]] | None:
    del rows, cols
    parents: dict[str, str | None] = {"S": None}
    queue = ["S"]
    while queue:
        node = queue.pop(0)
        for nxt in _successors(residual, node):
            if nxt in parents:
                continue
            parents[nxt] = node
            if nxt == "T":
                path: list[tuple[str, str]] = []
                cursor: str | None = "T"
                while cursor is not None and parents[cursor] is not None:
                    previous = parents[cursor]
                    assert previous is not None
                    path.append((previous, cursor))
                    cursor = previous
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _successors(residual: Mapping[tuple[str, str], int], node: str) -> list[str]:
    return sorted(right for (left, right), capacity in residual.items() if left == node and capacity > 0)


def _augment(residual: dict[tuple[str, str], int], path: Sequence[tuple[str, str]]) -> None:
    for left, right in path:
        residual[(left, right)] = residual.get((left, right), 0) - 1
        residual[(right, left)] = residual.get((right, left), 0) + 1


def _repair_toward_quota(
    table: dict[tuple[str, str], int],
    caps: Mapping[tuple[str, str], int],
    sample_size: int,
    population: int,
) -> tuple[dict[tuple[str, str], int], dict[str, object]]:
    """One 2x2 swap per step, in either direction, until L1 cannot fall.

    Each step changes four distinct cells by exactly one person. The next
    step is chosen only after that swap, so two cycles are not applied on
    the same counts. Row and column sums stay where they started.
    """
    quota = {cell: Decimal(sample_size) * Decimal(caps.get(cell, 0)) / Decimal(population) for cell in table}
    before = _quota_distance(table, quota)
    before_gaps = _cell_gap_summary(table, quota)
    current = before
    rows = sorted({row_value for row_value, _col in table})
    cols = sorted({col_value for _row, col_value in table})
    row_totals = _axis_totals(table, axis=0)
    col_totals = _axis_totals(table, axis=1)
    swaps: list[dict[str, object]] = []
    while len(swaps) <= len(table) * max(sample_size, 1):
        best: tuple[Decimal, int, str, str, str, str] | None = None
        for index, row_value in enumerate(rows):
            for other_row in rows[index + 1 :]:
                for col_index, col_value in enumerate(cols):
                    for other_col in cols[col_index + 1 :]:
                        for direction in (1, -1):
                            if not _cycle_fits(table, caps, row_value, other_row, col_value, other_col, direction):
                                continue
                            distance = _quota_distance(
                                _moved(table, row_value, other_row, col_value, other_col, direction),
                                quota,
                            )
                            if distance < current:
                                candidate = (distance, direction, row_value, other_row, col_value, other_col)
                                if best is None or candidate < best:
                                    best = candidate
        if best is None:
            break
        distance, direction, row_value, other_row, col_value, other_col = best
        cells = (
            (row_value, col_value),
            (row_value, other_col),
            (other_row, col_value),
            (other_row, other_col),
        )
        if len(set(cells)) != 4:
            raise SampleError("a 2x2 swap repeated a cell")
        _apply_cycle(table, row_value, other_row, col_value, other_col, direction)
        if _axis_totals(table, axis=0) != row_totals or _axis_totals(table, axis=1) != col_totals:
            raise SampleError("a 2x2 swap changed a Hamilton margin")
        current = distance
        swaps.append(
            {
                "cells": [f"{left}{_JOIN}{right}" for left, right in cells],
                "direction": "forward" if direction == 1 else "reverse",
                "l1_after": f"{current:.8f}",
            }
        )
    else:
        raise SampleError("2x2 repair did not stop")
    after_gaps = _cell_gap_summary(table, quota)
    return table, {
        "applied": True,
        "cells_abs_gap_at_least_one_after": after_gaps["count_at_least_one"],
        "cells_abs_gap_at_least_one_before": before_gaps["count_at_least_one"],
        "exact_cell_quota": False,
        "l1_after": f"{current:.8f}",
        "l1_before": f"{before:.8f}",
        "max_abs_cell_gap_after": after_gaps["max_abs"],
        "max_abs_cell_gap_before": before_gaps["max_abs"],
        "note": (
            "Each step applies one feasible 2x2 swap, forward or reverse, and "
            "only when it lowers L1 distance to n * N_cell / N. Hamilton margins "
            "stay fixed. This is not an exact cell-quota match and does not "
            "adjust unstratified fields."
        ),
        "swaps": swaps,
        "swap_count": len(swaps),
    }


def _moved(
    table: Mapping[tuple[str, str], int],
    row_value: str,
    other_row: str,
    col_value: str,
    other_col: str,
    direction: int,
) -> dict[tuple[str, str], int]:
    moved = dict(table)
    _apply_cycle(moved, row_value, other_row, col_value, other_col, direction)
    return moved


def _apply_cycle(
    table: dict[tuple[str, str], int],
    row_value: str,
    other_row: str,
    col_value: str,
    other_col: str,
    direction: int,
) -> None:
    table[(row_value, col_value)] += direction
    table[(other_row, other_col)] += direction
    table[(row_value, other_col)] -= direction
    table[(other_row, col_value)] -= direction


def _cycle_fits(
    table: Mapping[tuple[str, str], int],
    caps: Mapping[tuple[str, str], int],
    row_value: str,
    other_row: str,
    col_value: str,
    other_col: str,
    direction: int,
) -> bool:
    signs = {
        (row_value, col_value): direction,
        (other_row, other_col): direction,
        (row_value, other_col): -direction,
        (other_row, col_value): -direction,
    }
    for cell, sign in signs.items():
        updated = table[cell] + sign
        if updated < 0 or updated > caps.get(cell, 0):
            return False
    return True


def _axis_totals(table: Mapping[tuple[str, str], int], *, axis: int) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for cell, count in table.items():
        totals[cell[axis]] += count
    return dict(totals)


def _cell_gap_summary(table: Mapping[tuple[str, str], int], quota: Mapping[tuple[str, str], Decimal]) -> dict[str, object]:
    gaps = [abs(Decimal(table[cell]) - quota[cell]) for cell in table]
    at_least_one = sum(1 for gap in gaps if gap >= 1)
    maximum = max(gaps) if gaps else Decimal("0")
    return {"count_at_least_one": at_least_one, "max_abs": f"{maximum:.8f}"}


def _quota_distance(table: Mapping[tuple[str, str], int], quota: Mapping[tuple[str, str], Decimal]) -> Decimal:
    return sum((abs(Decimal(table[cell]) - quota[cell]) for cell in table), Decimal("0"))


def _assert_hamilton_margins(
    allocation: Mapping[str, int],
    groups: Mapping[str, Sequence[str]],
    marginal_targets: Mapping[str, Mapping[str, int]],
    sample_size: int,
    columns: Sequence[str],
) -> None:
    for column in columns:
        got: dict[str, int] = {level: 0 for level in marginal_targets[column]}
        for cell, count in allocation.items():
            values = cell.split(_JOIN) if len(columns) > 1 else [cell]
            got[values[columns.index(column)]] = got.get(values[columns.index(column)], 0) + count
        for level, target in marginal_targets[column].items():
            if got.get(level, 0) != target:
                raise SampleError(f"{column}={level!r} sample {got.get(level, 0)} != Hamilton target {target}")
        for cell, count in allocation.items():
            if count > len(groups[cell]):
                raise SampleError(f"cell {cell!r} allocation exceeds its population")
    if sum(allocation.values()) != sample_size:
        raise SampleError("joint allocation does not sum to the sample size")


def _margin_balance(
    marginal_targets: Mapping[str, Mapping[str, int]],
    marginal_counts: Mapping[str, Mapping[str, int]],
    sample_size: int,
    population: int,
) -> dict[str, dict[str, dict[str, object]]]:
    balance: dict[str, dict[str, dict[str, object]]] = {}
    for column, targets in marginal_targets.items():
        levels: dict[str, dict[str, object]] = {}
        for level in sorted(targets):
            pop_count = marginal_counts[column][level]
            quota = Decimal(sample_size) * Decimal(pop_count) / Decimal(population)
            gap = abs(Decimal(targets[level]) - quota)
            if gap >= 1:
                raise SampleError(f"{column}={level!r} Hamilton gap {gap} is not below 1")
            levels[level] = {
                "absolute_gap": f"{gap:.8f}",
                "population": pop_count,
                "quota": f"{quota:.8f}",
                "sample": targets[level],
                "within_hamilton_unit": True,
            }
        balance[column] = levels
    return balance


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
    return {name: base[name] for name in names}


def _cell_key(values: Sequence[str]) -> str:
    if len(values) == 1:
        return values[0]
    return _JOIN.join(values)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _rank(seed: int, dataset: str, stratum: str, subject_id: str) -> str:
    payload = f"{seed}\n{dataset}\n{stratum}\n{subject_id}".encode()
    return hashlib.sha256(payload).hexdigest()
