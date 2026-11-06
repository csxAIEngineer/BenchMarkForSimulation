"""Assemble cards, the sample, and the distribution report from in-memory rows.

No network and no model. Outcome cells are written to a separate document
and are checked so they do not appear as basic-info values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from popbench.pricing_basic_info.cards import build_card, normalize_cell
from popbench.pricing_basic_info.report import distribution_report
from popbench.pricing_basic_info.sample import planned_sample_size, stratified_sample
from popbench.pricing_basic_info.serialize import dump_jsonl, dumps, sha256_text
from popbench.pricing_basic_info.spec import ID_COLUMN, MISSING_TOKEN

_GENERATED_MARKERS = (
    "## Life history",
    "Childhood:",
    "Work path:",
    "Turning point:",
    "Personal income:",
    "Usual weekly work hours:",
    "tendency",
)


class AssembleError(ValueError):
    """Rows and outcomes cannot be assembled without changing a source value."""


def field_map(
    *,
    dataset: str,
    source_file: str,
    columns: Sequence[str],
    question_text: Mapping[str, str] | None,
    blocked_columns: Sequence[str],
) -> dict[str, object]:
    fields = []
    for column in columns:
        if question_text is None:
            stem = None
            stem_status = "unavailable_machine_readable_wording"
        else:
            stem = question_text[column]
            stem_status = "copied_from_question_catalog"
        fields.append(
            {
                "card_section": "Basics",
                "source_column": column,
                "source_file": source_file,
                "source_question_id": column,
                "source_question_text": stem,
                "source_question_text_status": stem_status,
                "transform": (
                    "Copy the cell after removing only leading and trailing whitespace. "
                    f"A blank cell is stored as {MISSING_TOKEN}. No other value is inferred."
                ),
            }
        )
    return {
        "blocked_columns": list(blocked_columns),
        "dataset": dataset,
        "fields": fields,
        "source_file": source_file,
    }


def assemble_dataset(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    source_file: str,
    columns: Sequence[str],
    stratify_column: str,
    question_text: Mapping[str, str] | None,
    outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    outcome_ids: Sequence[str],
    forbidden_columns: Sequence[str],
    seed: int,
    sample_rule: Mapping[str, str],
    sample_size_override: int | None = None,
    eligibility_note: str,
) -> dict[str, str]:
    """Return filename -> UTF-8 text for one dataset. Byte-stable for the same rows."""
    eligible, exclusions = _eligible(rows, outcomes, outcome_ids)
    plan = planned_sample_size(
        len(eligible),
        z=str(sample_rule["z"]),
        proportion=str(sample_rule["proportion"]),
        margin=str(sample_rule["margin_of_error"]),
        override=sample_size_override,
    )
    draw = stratified_sample(
        eligible,
        stratify_column=stratify_column,
        sample_size=int(plan["sample_size"]),
        seed=seed,
        dataset=dataset,
    )
    cards = [
        build_card(
            row,
            dataset=dataset,
            source_file=source_file,
            columns=columns,
            question_text=question_text,
        )
        for row in _sorted_rows(eligible)
    ]
    _reject_generated_text(cards)
    _assert_values_match_source(cards, eligible, columns)
    _reject_outcome_leak(cards, outcomes, forbidden_columns)
    report = distribution_report(eligible, draw["subject_ids"], columns)
    report = {
        "dataset": dataset,
        "sample_size_rule": plan,
        "seed": seed,
        "stratify_column": stratify_column,
        **report,
    }
    sample = {
        "allocation": draw["allocation"],
        "dataset": dataset,
        "population_size": draw["population_size"],
        "population_stratum_counts": draw["population_stratum_counts"],
        "sample_size": draw["sample_size"],
        "sample_size_rule": plan,
        "seed": seed,
        "stratify_column": stratify_column,
        "subject_ids": draw["subject_ids"],
    }
    exclusion_doc = {
        "dataset": dataset,
        "eligibility": eligibility_note,
        "excluded": exclusions,
        "included": len(eligible),
        "input_rows": len(rows),
    }
    outcome_rows = [_outcome_row(dataset, row, outcomes[str(row[ID_COLUMN]).strip()]) for row in _sorted_rows(eligible)]
    documents = {
        "cards.jsonl": dump_jsonl(cards),
        "distribution.json": dumps(report),
        "exclusions.json": dumps(exclusion_doc),
        "outcomes.jsonl": dump_jsonl(outcome_rows),
        "sample.json": dumps(sample),
    }
    for name, text in documents.items():
        if name.endswith(".jsonl") and text and not text.endswith("\n"):
            raise AssembleError(f"{name} is not newline-terminated")
    return documents


def _eligible(
    rows: Sequence[Mapping[str, object]],
    outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    outcome_ids: Sequence[str],
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    eligible: list[Mapping[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        subject_id = str(row.get(ID_COLUMN) or "").strip()
        if not subject_id:
            excluded.append({"reason": "missing pid", "subject_id": ""})
            continue
        if subject_id in seen:
            raise AssembleError(f"duplicate pid: {subject_id}")
        seen.add(subject_id)
        items = outcomes.get(subject_id)
        reason = _outcome_reason(items, outcome_ids)
        if reason is not None:
            excluded.append({"reason": reason, "subject_id": subject_id})
            continue
        eligible.append(row)
    excluded.sort(key=lambda item: (item["subject_id"], item["reason"]))
    return eligible, excluded


def _outcome_reason(
    items: Sequence[Mapping[str, object]] | None,
    outcome_ids: Sequence[str],
) -> str | None:
    if not items:
        return "missing pricing outcomes"
    ids = [str(item.get("question_id") or "") for item in items]
    expected = list(outcome_ids)
    if ids != expected:
        return "pricing outcome ids are not the expected set in order"
    for item in items:
        answer, status = normalize_cell(item.get("answer"))
        if status != "present":
            return f"pricing outcome {item.get('question_id')} is blank"
        if answer != str(item.get("answer")).strip():
            return f"pricing outcome {item.get('question_id')} was not stored as the source cell"
    return None


def _outcome_row(
    dataset: str,
    row: Mapping[str, object],
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "items": [dict(item) for item in items],
        "subject_id": str(row[ID_COLUMN]).strip(),
    }


def _sorted_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(rows, key=lambda row: _subject_sort(str(row[ID_COLUMN]).strip()))


def _subject_sort(subject_id: str) -> tuple[int, int | str]:
    if subject_id.isdigit():
        return (0, int(subject_id))
    return (1, subject_id)


def _reject_generated_text(cards: Sequence[Mapping[str, object]]) -> None:
    for card in cards:
        text = str(card["text"])
        for marker in _GENERATED_MARKERS:
            if marker in text:
                raise AssembleError(f"{card['subject_id']} card contains generated text {marker!r}")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        if headings != ["## Identity", "## Basics"]:
            raise AssembleError(f"{card['subject_id']} card has unexpected sections: {headings}")


def _assert_values_match_source(
    cards: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    by_id = {str(row[ID_COLUMN]).strip(): row for row in rows}
    for card in cards:
        row = by_id[str(card["subject_id"])]
        fields = card["basic_info"]
        if not isinstance(fields, list):
            raise AssembleError("basic_info is not a list")
        if [str(field["source_column"]) for field in fields] != list(columns):
            raise AssembleError(f"{card['subject_id']} basic-info columns do not match the mapping")
        for field in fields:
            expected, status = normalize_cell(row.get(str(field["source_column"])))
            if field["value"] != expected or field["status"] != status:
                raise AssembleError(f"{card['subject_id']} {field['source_column']} does not match the source cell")


def _reject_outcome_leak(
    cards: Sequence[Mapping[str, object]],
    outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    forbidden_columns: Sequence[str],
) -> None:
    blocked = set(forbidden_columns)
    for card in cards:
        fields = card["basic_info"]
        if not isinstance(fields, list):
            raise AssembleError("basic_info is not a list")
        columns = [str(field["source_column"]) for field in fields]
        if any(column in blocked for column in columns):
            raise AssembleError(f"{card['subject_id']} stored a blocked column on the card")
        text = str(card["text"])
        for item in outcomes[str(card["subject_id"])]:
            question = str(item.get("question_text") or "")
            if len(question) >= 40 and question in text:
                raise AssembleError(f"{card['subject_id']} copied a pricing stimulus onto the card")


def document_hashes(documents: Mapping[str, str]) -> dict[str, str]:
    return {name: sha256_text(text) for name, text in sorted(documents.items())}
