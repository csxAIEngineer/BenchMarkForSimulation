"""Every source column is classified. An unknown or unmapped column fails."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class ColumnClassError(ValueError):
    """A source column is missing, unknown, or not mapped onto the card."""


def classify_header(
    fieldnames: Sequence[str],
    *,
    id_column: str,
    basic: Sequence[str],
    outcome: Sequence[str],
    design: Sequence[str] = (),
    admin: Sequence[str] = (),
    known_other: Sequence[str] = (),
    unmapped_basic: Sequence[str] = (),
) -> dict[str, str]:
    """Return column -> role. Fail closed on anything outside the lists.

    A blank cell in a present basic column is handled later as MISSING.
    A basic column that is absent from the header is not turned into MISSING.
    """
    roles: dict[str, str] = {}
    lists = (
        (id_column, "id"),
        *((column, "basic") for column in basic),
        *((column, "outcome") for column in outcome),
        *((column, "design") for column in design),
        *((column, "admin") for column in admin),
        *((column, "known_not_on_card") for column in known_other),
    )
    for column, role in lists:
        if column in roles and roles[column] != role:
            raise ColumnClassError(f"{column} is listed in more than one role")
        roles[column] = role
    if unmapped_basic:
        raise ColumnClassError(
            "unmapped basic-info question "
            + ", ".join(sorted(unmapped_basic))
            + ". Add it to the card mapping or it stays unknown."
        )
    missing = [column for column in (id_column, *basic, *outcome) if column not in fieldnames]
    if missing:
        raise ColumnClassError("required column is absent: " + ", ".join(missing))
    unknown = [column for column in fieldnames if column not in roles]
    if unknown:
        raise ColumnClassError("unknown column: " + ", ".join(unknown))
    return {column: roles[column] for column in fieldnames}


def assert_row_columns(row: Mapping[str, object], allowed: Sequence[str]) -> None:
    allowed_set = set(allowed)
    unknown = [str(key) for key in row if str(key) not in allowed_set]
    if unknown:
        raise ColumnClassError("unknown column: " + ", ".join(sorted(unknown)))
