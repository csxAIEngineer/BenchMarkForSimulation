"""Copy one source row onto a basic-info card.

The card has two sections, Identity and Basics. Every Basics line is one
source column. Empty cells stay empty in the form of the MISSING token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from popbench.prepro.spec import (
    CARD_KEYS,
    CARD_VERSION,
    FIELD_KEYS,
    ID_COLUMN,
    MISSING_TOKEN,
    QUESTION_TEXT_COPIED,
    QUESTION_TEXT_UNAVAILABLE,
    STATUS_MISSING,
    STATUS_PRESENT,
)


class CardBuildError(ValueError):
    """A row cannot be copied onto a card without changing its meaning."""


def normalize_cell(value: object) -> tuple[str, str]:
    """Return the cell text and present/missing.

    Only leading and trailing whitespace is removed. A blank cell is missing.
    Any other text, including "Prefer not to answer" or "Unknown", is kept.
    The sentinel MISSING is reserved, so a source cell with that exact text fails.
    """
    if value is None:
        return MISSING_TOKEN, STATUS_MISSING
    text = str(value).strip()
    if text == "":
        return MISSING_TOKEN, STATUS_MISSING
    if text == MISSING_TOKEN:
        raise CardBuildError(
            f"source cell is the reserved token {MISSING_TOKEN!r}; it is not rewritten and not accepted"
        )
    if any(mark in text for mark in ("\n", "\r")):
        raise CardBuildError("source cell contains a newline; it was not rewritten")
    return text, STATUS_PRESENT


def build_card(
    row: Mapping[str, object],
    *,
    dataset: str,
    source_file: str,
    columns: tuple[str, ...] | list[str],
    question_text: Mapping[str, str] | None,
    question_options: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Build one card. A column that is not in ``row`` is an absent column, not MISSING."""
    subject_id = str(row.get(ID_COLUMN) or "").strip()
    if not subject_id:
        raise CardBuildError("row is missing pid")
    fields: list[dict[str, object]] = []
    for column in columns:
        if column not in row:
            raise CardBuildError(f"required column {column} is absent")
        value, status = normalize_cell(row.get(column))
        if question_text is None:
            stem: str | None = None
            stem_status = QUESTION_TEXT_UNAVAILABLE
            options: list[str] | None = None
        else:
            if column not in question_text:
                raise CardBuildError(f"question catalog has no stem for {column}")
            raw_stem = question_text[column]
            if raw_stem is None or str(raw_stem).strip() == "":
                raise CardBuildError(f"question catalog stem for {column} is empty")
            stem = str(raw_stem)
            stem_status = QUESTION_TEXT_COPIED
            raw_options = () if question_options is None else question_options.get(column, ())
            options = [str(option) for option in raw_options]
            if status == STATUS_PRESENT and options and value not in options:
                raise CardBuildError(f"{column} value {value!r} is not in the catalog options")
        fields.append(
            {
                "source_column": column,
                "source_options": options,
                "source_question_id": column,
                "source_question_text": stem,
                "source_question_text_status": stem_status,
                "status": status,
                "value": value,
            }
        )
    card: dict[str, object] = {
        "basic_info": fields,
        "card_version": CARD_VERSION,
        "dataset": dataset,
        "source_file": source_file,
        "source_id_column": ID_COLUMN,
        "subject_id": subject_id,
        "text": render_text(
            dataset=dataset,
            subject_id=subject_id,
            source_file=source_file,
            fields=fields,
        ),
    }
    if set(card) != set(CARD_KEYS):
        raise CardBuildError(f"card keys drifted: {sorted(set(card))}")
    for field in fields:
        if set(field) != set(FIELD_KEYS):
            raise CardBuildError("basic_info field keys drifted")
    return card


def render_text(
    *,
    dataset: str,
    subject_id: str,
    source_file: str,
    fields: list[dict[str, object]],
) -> str:
    lines = [
        "## Identity",
        f"dataset: {dataset}",
        f"subject_id: {subject_id}",
        f"source_file: {source_file}",
        f"source_id_column: {ID_COLUMN}",
        "",
        "## Basics",
    ]
    for field in fields:
        lines.append(f"{field['source_column']}: {field['value']}")
    return "\n".join(lines) + "\n"
