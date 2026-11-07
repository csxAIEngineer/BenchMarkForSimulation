"""Assemble cards, the sample, and the distribution report from in-memory rows.

No network and no model. Outcome cells are written to a separate document.
Every outcome string is checked against the card on token boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from popbench.prepro.cards import basics_label, build_card, normalize_cell
from popbench.prepro.columns import assert_row_columns
from popbench.prepro.report import distribution_report
from popbench.prepro.sample import planned_sample_size, stratified_sample
from popbench.prepro.serialize import dump_jsonl, dumps
from popbench.prepro.spec import ID_COLUMN, MISSING_TOKEN

_GENERATED_MARKERS = (
    "## Life history",
    "Childhood:",
    "Work path:",
    "Turning point:",
    "Personal income:",
    "Usual weekly work hours:",
    "tendency",
)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*|\$\d+(?:\.\d+)?")
_PRODUCT = re.compile(
    r"following product in that category: (.+?)\. The product is priced at: (\$\d+(?:\.\d+)?)"
)


class AssembleError(ValueError):
    """Rows and outcomes cannot be assembled without changing a source value."""


def field_map(
    *,
    dataset: str,
    source_file: str,
    columns: Sequence[str],
    question_text: Mapping[str, str] | None,
    question_options: Mapping[str, Sequence[str]] | None,
    blocked_columns: Sequence[str],
) -> dict[str, object]:
    fields = []
    for column in columns:
        if question_text is None:
            stem = None
            stem_status = "unavailable_machine_readable_wording"
            options = None
        else:
            stem = question_text[column]
            stem_status = "copied_from_question_catalog"
            options = list(question_options[column]) if question_options is not None else []
        fields.append(
            {
                "card_section": "Basics",
                "source_column": column,
                "source_file": source_file,
                "source_options": options,
                "source_question_id": column,
                "source_question_text": stem,
                "source_question_text_status": stem_status,
                "transform": (
                    "Copy the cell after removing only leading and trailing whitespace. "
                    f"A blank cell is stored as {MISSING_TOKEN}. "
                    "A cell whose text is that token fails. No other value is inferred."
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
    stratify_columns: Sequence[str],
    question_text: Mapping[str, str] | None,
    question_options: Mapping[str, Sequence[str]] | None = None,
    outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    outcome_ids: Sequence[str],
    forbidden_columns: Sequence[str],
    allowed_columns: Sequence[str],
    seed: int,
    sample_rule: Mapping[str, str],
    sample_size_override: int | None = None,
    eligibility_note: str,
    reported_groupings: Sequence[str] = (),
) -> dict[str, str]:
    """Return filename -> UTF-8 text for one dataset. Byte-stable for the same rows."""
    for row in rows:
        assert_row_columns(row, allowed_columns)
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
        stratify_columns=stratify_columns,
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
            question_options=question_options,
        )
        for row in _sorted_rows(eligible)
    ]
    _reject_generated_text(cards)
    _assert_values_match_source(cards, eligible, columns)
    assert_no_outcome_leak(cards, outcomes, forbidden_columns)
    report = distribution_report(
        eligible,
        draw["subject_ids"],
        columns,
        groupings={column: "age_decade" for column in reported_groupings},
    )
    report = {
        "dataset": dataset,
        "not_statistically_matched": (
            "Margins of the stratifiers are the Hamilton targets. "
            "Other fields are counts only. They are not a claim that the sample matches the population."
        ),
        "sample_size_rule": plan,
        "seed": seed,
        "stratify_columns": list(stratify_columns),
        **report,
    }
    sample = {
        "allocation": draw["allocation"],
        "dataset": dataset,
        "joint_quota_adjustment": draw["joint_quota_adjustment"],
        "margin_balance": draw["margin_balance"],
        "marginal_allocation": draw["marginal_allocation"],
        "population_size": draw["population_size"],
        "population_stratum_counts": draw["population_stratum_counts"],
        "rare_level_policy": draw["rare_level_policy"],
        "sample_size": draw["sample_size"],
        "sample_size_rule": plan,
        "seed": seed,
        "stratify_columns": list(stratify_columns),
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
    return documents


def assert_no_outcome_leak(
    cards: Sequence[Mapping[str, object]],
    outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    forbidden_columns: Sequence[str],
) -> None:
    """Fail if any outcome string lands on a card outside that person's basic cells."""
    blocked = set(forbidden_columns)
    for card in cards:
        fields = card["basic_info"]
        if not isinstance(fields, list):
            raise AssembleError("basic_info is not a list")
        columns = [str(field["source_column"]) for field in fields]
        if any(column in blocked for column in columns):
            raise AssembleError(f"{card['subject_id']} stored a blocked column on the card")
        source_values = {str(field["value"]) for field in fields}
        text = str(card["text"])
        basics = text.split("## Basics\n", 1)[-1]
        card_tokens = set(_TOKEN.findall(basics))
        line_values = _line_values(fields, basics)
        for item in outcomes[str(card["subject_id"])]:
            for needle in _outcome_strings(item):
                if needle in source_values:
                    continue
                if needle.isdigit():
                    if needle in line_values:
                        raise AssembleError(f"{card['subject_id']} copied outcome value {needle!r} onto the card")
                    continue
                if _distinctive(needle) and needle in basics:
                    raise AssembleError(f"{card['subject_id']} copied outcome text {needle!r} onto the card")
                for token in _leak_tokens(needle):
                    if token in source_values:
                        continue
                    if token in card_tokens:
                        raise AssembleError(f"{card['subject_id']} copied outcome token {token!r} onto the card")


def _line_values(fields: object, basics: str) -> set[str]:
    """Values on Basics lines. A catalog stem may itself contain ``: ``."""
    expected: dict[str, str] = {}
    if isinstance(fields, list):
        for field in fields:
            if isinstance(field, dict) and "value" in field:
                expected[f"{basics_label(field)}: {field['value']}"] = str(field["value"])
    values: set[str] = set()
    for line in basics.splitlines():
        if line in expected:
            values.add(expected[line])
        elif ": " in line:
            values.add(line.split(": ", 1)[1])
    return values


def _outcome_strings(item: Mapping[str, object]) -> list[str]:
    found: list[str] = []
    for key in ("answer", "question_id", "question_text", "source_column"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    stimulus = item.get("stimulus")
    if isinstance(stimulus, dict):
        for value in stimulus.values():
            if isinstance(value, str) and value.strip():
                found.append(value.strip())
    question = item.get("question_text")
    if isinstance(question, str):
        match = _PRODUCT.search(question)
        if match:
            found.append(match.group(1).strip())
            found.append(match.group(2).strip())
    return found


def _distinctive(needle: str) -> bool:
    return len(needle) >= 4 or any(mark in needle for mark in ("_", "$", "."))


def _leak_tokens(text: str) -> list[str]:
    """Product codes and prices. Bare integers are checked as whole line values."""
    tokens = []
    for token in _TOKEN.findall(text):
        if token.isdigit():
            continue
        if "_" in token or "." in token or token.startswith("$"):
            tokens.append(token)
    return tokens


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
    if ids != list(outcome_ids):
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
            column = str(field["source_column"])
            if column not in row:
                raise AssembleError(f"{card['subject_id']} required column {column} is absent")
            expected, status = normalize_cell(row.get(column))
            if field["value"] != expected or field["status"] != status:
                raise AssembleError(f"{card['subject_id']} {column} does not match the source cell")
