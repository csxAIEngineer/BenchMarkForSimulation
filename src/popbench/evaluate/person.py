"""Person-level scores for Twin2K binary pricing questions.

``categorical.mean_top1_acc`` asks whether the group's most common answer
matches the group's most common human answer, one question at a time. It is
not individual accuracy. Individual accuracy asks whether this subject's
reply matches this subject's held-out answer on the same question.

Cross-person correlation, Glass delta, and the SD ratio use each person's
purchase rate on one shared question set. A rate of 0.25 means the person
was coded Yes on one quarter of those pricing questions. These are not
willingness-to-pay estimates, and they are not item-level modal rates.

Units
-----
individual accuracy
    Fraction of scorable pairs, in ``[0, 1]``. Higher is better.
cross-person Pearson and Spearman
    Unitless correlation of person purchase rates, in ``[-1, 1]``.
    Higher is better. Spearman uses average ranks for ties.
Glass delta
    ``(mean(model rates) - mean(human rates)) / sample_sd(human rates)``
    with ``ddof=1``. Closer to 0 is better. Positive means the model
    gives people higher purchase rates than the humans did.
SD ratio
    ``sample_sd(model rates) / sample_sd(human rates)`` with ``ddof=1``.
    Closer to 1 is better. Below 1 means between-person variability is
    compressed. This is not the population SD used by the BDM WTP block.

Pairing
-------
Keys are ``(subject_id, question_id)``. Row order is ignored. The support
``Q`` is every question whose held-out answers form a Yes=1 / No=0 codebook.
A subject is complete when every question in ``Q`` has one human answer and
one model answer, both coded 0 or 1, with no parse error and no conflicting
duplicate. Purchase rate is the mean of those codes over that same ``Q``.

Scorable accuracy pairs are keys inside ``Q`` where both sides exist and
neither side conflicts. A parse error or an unmapped reply is incorrect and
stays in the denominator. Conflicting duplicates are excluded from the
accuracy denominator and from rates, and they are counted. Identical
duplicate rows collapse to one pair. Missing model or human rows are counted
and are not imputed. Incomplete subjects remain in individual accuracy but
are left out of the cross-person aggregates.

A metric is JSON ``null`` with a reason when it is undefined: fewer than
two complete subjects, or a constant rate vector (sample SD 0). A model SD
of 0 with a positive human SD is a real SD ratio of 0, not null.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from popbench.evaluate.metrics import pearson, sample_std, spearman
from popbench.evaluate.options import binary_codebook, code_purchase

DIRECTIONS = {
    "individual_accuracy_micro": "higher_better",
    "individual_accuracy_macro": "higher_better",
    "cross_person_pearson": "higher_better",
    "cross_person_spearman": "higher_better",
    "glass_delta": "closer_to_zero",
    "sd_ratio": "closer_to_one",
}


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _metric(value: float | None, reason: str | None) -> dict[str, Any]:
    number = _finite(value)
    if number is None:
        return {"reason": reason or "undefined", "value": None}
    return {"reason": None, "value": number}


def _rows_for(bucket: dict[tuple[str, str], list[dict[str, Any]]], key: tuple[str, str]) -> list[dict[str, Any]]:
    return bucket.get(key, [])


def _collapse_text(rows: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Collapse one side of a key.

    Returns ``(status, text)``. Status is ``missing``, ``parse_error``,
    ``ok``, or ``conflict``. Identical texts collapse. Parse errors do not
    become a purchase code.
    """
    if not rows:
        return "missing", None
    if any(row.get("parse_error") for row in rows):
        if len(rows) > 1 and any(not row.get("parse_error") for row in rows):
            return "conflict", None
        return "parse_error", None
    texts = [str(row.get("answer") or "").strip() for row in rows]
    if len(set(texts)) > 1:
        return "conflict", None
    return "ok", texts[0]


def score_pricing_persons(
    responses: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any] | None:
    """Score binary purchase questions. None when the outcomes are not that task."""
    human_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    answers_by_question: dict[str, list[str]] = defaultdict(list)
    for row in outcome_rows:
        subject_id = str(row.get("subject_id") or "")
        items = row.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id") or "")
            answer = str(item.get("answer") or "").strip()
            human_rows[(subject_id, question_id)].append({"answer": answer, "parse_error": None})
            answers_by_question[question_id].append(answer)

    codebooks: dict[str, dict[str, int]] = {}
    skipped_questions: list[str] = []
    for question_id in sorted(answers_by_question):
        options = sorted(set(answers_by_question[question_id]))
        codebook = binary_codebook(options)
        if codebook is None:
            skipped_questions.append(question_id)
            continue
        codebooks[question_id] = codebook
    if not codebooks:
        return None

    model_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    response_subjects: set[str] = set()
    n_response_rows = 0
    for row in responses:
        subject_id = str(row.get("subject_id") or "")
        question_id = str(row.get("question_id") or "")
        if question_id not in codebooks:
            continue
        response_subjects.add(subject_id)
        n_response_rows += 1
        model_rows[(subject_id, question_id)].append(
            {"answer": str(row.get("answer") or "").strip(), "parse_error": row.get("parse_error")}
        )

    support = tuple(sorted(codebooks))
    n_identical_duplicate_rows = 0
    n_duplicate_conflicts = 0
    n_missing_model = 0
    n_missing_human = 0
    n_parse_error_pairs = 0
    n_unmapped_model = 0
    n_unmapped_human = 0
    n_scorable_pairs = 0
    n_correct_pairs = 0
    per_person: list[dict[str, Any]] = []
    complete_human: list[float] = []
    complete_model: list[float] = []

    for subject_id in sorted(response_subjects):
        correct = 0
        scorable = 0
        codes_human: list[int] = []
        codes_model: list[int] = []
        complete = True
        for question_id in support:
            key = (subject_id, question_id)
            human_side = _rows_for(human_rows, key)
            model_side = _rows_for(model_rows, key)
            n_identical_duplicate_rows += max(0, len(human_side) - 1)
            n_identical_duplicate_rows += max(0, len(model_side) - 1)
            human_status, human_text = _collapse_text(human_side)
            model_status, model_text = _collapse_text(model_side)
            if human_status == "conflict" or model_status == "conflict":
                n_duplicate_conflicts += 1
                # Conflicting copies were counted as identical extras above.
                n_identical_duplicate_rows -= max(0, len(human_side) - 1)
                n_identical_duplicate_rows -= max(0, len(model_side) - 1)
                complete = False
                continue
            if human_status == "missing":
                n_missing_human += 1
                complete = False
                continue
            if model_status == "missing":
                n_missing_model += 1
                complete = False
                continue
            codebook = codebooks[question_id]
            human_code = None if human_text is None else code_purchase(human_text, codebook)
            if human_code is None:
                n_unmapped_human += 1
            if model_status == "parse_error":
                n_parse_error_pairs += 1
                model_code = None
            else:
                model_code = None if model_text is None else code_purchase(model_text, codebook)
                if model_code is None:
                    n_unmapped_model += 1
            scorable += 1
            matched = human_code is not None and model_code is not None and human_code == model_code
            if matched:
                correct += 1
            if human_code is None or model_code is None:
                complete = False
                continue
            codes_human.append(human_code)
            codes_model.append(model_code)
        if len(codes_human) != len(support):
            complete = False
        person: dict[str, Any] = {
            "accuracy": (correct / scorable) if scorable else None,
            "complete": complete,
            "n_correct": correct,
            "n_scorable": scorable,
            "purchase_rate_human": None,
            "purchase_rate_model": None,
            "subject_id": subject_id,
        }
        if complete:
            human_rate = sum(codes_human) / len(support)
            model_rate = sum(codes_model) / len(support)
            person["purchase_rate_human"] = human_rate
            person["purchase_rate_model"] = model_rate
            complete_human.append(human_rate)
            complete_model.append(model_rate)
        n_scorable_pairs += scorable
        n_correct_pairs += correct
        per_person.append(person)

    macro_values = [float(person["accuracy"]) for person in per_person if person["n_scorable"]]
    if n_scorable_pairs:
        micro = _metric(n_correct_pairs / n_scorable_pairs, None)
    else:
        micro = _metric(None, "no scorable pairs")
    if macro_values:
        macro = _metric(sum(macro_values) / len(macro_values), None)
    else:
        macro = _metric(None, "no subject has a scorable pair")

    n_complete = len(complete_human)
    pearson_reason = None
    spearman_reason = None
    glass_reason = None
    ratio_reason = None
    pearson_value = None
    spearman_value = None
    glass_value = None
    ratio_value = None
    if n_complete < 2:
        pearson_reason = "fewer than 2 complete subjects"
        spearman_reason = pearson_reason
        glass_reason = pearson_reason
        ratio_reason = pearson_reason
    else:
        human_sd = sample_std(complete_human)
        model_sd = sample_std(complete_model)
        try:
            pearson_value = pearson(complete_model, complete_human)
        except ValueError:
            pearson_value = None
        if pearson_value is None:
            pearson_reason = "a person-rate vector is constant"
        try:
            spearman_value = spearman(complete_model, complete_human)
        except ValueError:
            spearman_value = None
        if spearman_value is None:
            spearman_reason = "a person-rate ranking is constant"
        if human_sd is None or human_sd == 0:
            glass_reason = "human person-rate sample sd is 0"
            ratio_reason = "human person-rate sample sd is 0"
        else:
            human_mean = sum(complete_human) / n_complete
            model_mean = sum(complete_model) / n_complete
            glass_value = (model_mean - human_mean) / human_sd
            ratio_value = 0.0 if model_sd == 0 else (model_sd or 0.0) / human_sd

    return {
        "complete_subject_policy": (
            "A subject is complete when every binary-purchase question in the "
            "support has one coded human answer and one coded model answer. "
            "Rates use that same question set. Incomplete subjects stay in "
            "individual accuracy and are excluded from correlation, Glass delta, "
            "and the SD ratio."
        ),
        "coverage": {
            "n_complete_subjects": n_complete,
            "n_correct_pairs": n_correct_pairs,
            "n_duplicate_conflicts": n_duplicate_conflicts,
            "n_identical_duplicate_rows": n_identical_duplicate_rows,
            "n_incomplete_subjects": len(response_subjects) - n_complete,
            "n_missing_human": n_missing_human,
            "n_missing_model": n_missing_model,
            "n_parse_error_pairs": n_parse_error_pairs,
            "n_questions_in_support": len(support),
            "n_questions_skipped_not_binary": len(skipped_questions),
            "n_response_rows_on_support": n_response_rows,
            "n_scorable_pairs": n_scorable_pairs,
            "n_subjects_in_macro": len(macro_values),
            "n_subjects_with_responses": len(response_subjects),
            "n_unmapped_human": n_unmapped_human,
            "n_unmapped_model": n_unmapped_model,
        },
        "cross_person_pearson": _metric(pearson_value, pearson_reason),
        "cross_person_spearman": _metric(spearman_value, spearman_reason),
        "directions": DIRECTIONS,
        "glass_delta": _metric(glass_value, glass_reason),
        "individual_accuracy_macro": macro,
        "individual_accuracy_micro": micro,
        "not_group_top1": (
            "categorical.mean_top1_acc is 1 when the modal answer in the whole "
            "group matches the human mode on that question. "
            "individual_accuracy_micro is the fraction of (subject, question) "
            "pairs whose coded reply equals that same person's held-out answer."
        ),
        "per_person": per_person,
        "purchase_rate_unit": (
            "Fraction of support questions coded Yes=1. Not a willingness-to-pay."
        ),
        "question_ids": list(support),
        "sd_ratio": _metric(ratio_value, ratio_reason),
    }
