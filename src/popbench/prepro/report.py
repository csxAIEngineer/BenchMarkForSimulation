"""Counts, proportions, and absolute deviations for one draw.

Decimals are a fixed-width display of the integer counts. The report does
not contain a significance test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_EVEN

from popbench.prepro.cards import normalize_cell
from popbench.prepro.spec import ID_COLUMN

_DISPLAY = Decimal("0.00000001")
_NOTE = (
    "Counts, proportions, and absolute deviations describe this draw against "
    "the full eligible population. This report does not include a significance test."
)


def distribution_report(
    rows: Sequence[Mapping[str, object]],
    sample_ids: Sequence[str],
    columns: Sequence[str],
    *,
    groupings: Mapping[str, str] | None = None,
) -> dict[str, object]:
    by_id = {}
    for row in rows:
        subject_id = str(row.get(ID_COLUMN) or "").strip()
        if subject_id in by_id:
            raise ValueError(f"duplicate pid: {subject_id}")
        by_id[subject_id] = row
    missing_ids = [subject_id for subject_id in sample_ids if subject_id not in by_id]
    if missing_ids:
        raise ValueError(f"sample id is not in the population: {missing_ids[:3]}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample ids are not unique")
    sample_set = set(sample_ids)
    population_n = len(by_id)
    sample_n = len(sample_ids)
    fields = {
        column: _field_report(by_id, sample_set, column, population_n, sample_n)
        for column in columns
    }
    reported = {
        f"{column}_{kind}": _grouped_report(by_id, sample_set, column, kind, population_n, sample_n)
        for column, kind in (groupings or {}).items()
    }
    return {
        "fields": fields,
        "note": _NOTE,
        "population_size": population_n,
        "reported_groupings": reported,
        "reported_groupings_role": "descriptive bins, not design strata",
        "sample_size": sample_n,
    }


def _field_report(
    by_id: Mapping[str, Mapping[str, object]],
    sample_ids: set[str],
    column: str,
    population_n: int,
    sample_n: int,
    *,
    values_are_final: bool = False,
) -> dict[str, object]:
    population_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    for subject_id, row in by_id.items():
        if values_are_final:
            value = str(row.get(column) or "")
            if value == "":
                value = "MISSING"
        else:
            value, _status = normalize_cell(row.get(column))
        population_counts[value] += 1
        if subject_id in sample_ids:
            sample_counts[value] += 1
    levels = []
    abs_diffs: list[Decimal] = []
    l1 = Decimal("0")
    for value in sorted(population_counts):
        pop_count = population_counts[value]
        sample_count = sample_counts.get(value, 0)
        difference = _proportion(sample_count, sample_n) - _proportion(pop_count, population_n)
        abs_diffs.append(abs(difference))
        l1 += abs(difference)
        levels.append(
            {
                "population_count": pop_count,
                "population_proportion": _display(_proportion(pop_count, population_n)),
                "proportion_difference": _display(difference),
                "sample_count": sample_count,
                "sample_proportion": _display(_proportion(sample_count, sample_n)),
                "value": value,
            }
        )
    return {
        "levels": levels,
        "max_abs_proportion_difference": _display(max(abs_diffs) if abs_diffs else Decimal("0")),
        "population_denominator": population_n,
        "sample_denominator": sample_n,
        "total_variation_distance": _display(l1 / Decimal("2")),
    }


def _grouped_report(
    by_id: Mapping[str, Mapping[str, object]],
    sample_ids: set[str],
    column: str,
    kind: str,
    population_n: int,
    sample_n: int,
) -> dict[str, object]:
    if kind != "age_decade":
        raise ValueError(f"unknown grouping {kind}")
    grouped_rows = []
    for subject_id, row in by_id.items():
        value, _status = normalize_cell(row.get(column))
        grouped_rows.append({ID_COLUMN: subject_id, column: _age_decade(value)})
    report = _field_report(
        {row[ID_COLUMN]: row for row in grouped_rows},
        sample_ids,
        column,
        population_n,
        sample_n,
        values_are_final=True,
    )
    report["bin"] = "floor(numeric age / 10) * 10 through that decade; blank stays MISSING"
    report["role"] = "descriptive_bin_not_a_design_stratum"
    report["source_column"] = column
    return report


def _age_decade(value: str) -> str:
    if value == "MISSING":
        return value
    if not value.isdigit():
        raise ValueError(f"age decade grouping does not rewrite non-numeric age {value!r}")
    decade = int(value) // 10 * 10
    return f"{decade}-{decade + 9}"


def _proportion(count: int, total: int) -> Decimal:
    if total < 1:
        raise ValueError("proportion denominator must be positive")
    return Decimal(count) / Decimal(total)


def _display(value: Decimal) -> str:
    quantized = value.quantize(_DISPLAY, rounding=ROUND_HALF_EVEN)
    return f"{quantized:.8f}"
