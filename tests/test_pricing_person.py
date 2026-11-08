"""Person-level Twin pricing scores. No model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from popbench.evaluate.options import code_purchase, match_option
from popbench.evaluate.person import score_pricing_persons
from popbench.evaluate.score import score_dataset

YES = "Yes, I would purchase the product"
NO = "No, I would not purchase the product"
OPTIONS = [NO, YES]


def _human(subject: str, answers: dict[str, str]) -> dict[str, object]:
    return {
        "items": [{"answer": answer, "question_id": question} for question, answer in answers.items()],
        "subject_id": subject,
    }


def _reply(subject: str, question: str, answer: str, *, parse_error: str | None = None) -> dict[str, object]:
    return {
        "answer": answer,
        "parse_error": parse_error,
        "question_id": question,
        "subject_id": subject,
    }


def _value(block: dict[str, object], name: str) -> float | None:
    metric = block[name]
    assert isinstance(metric, dict)
    value = metric["value"]
    assert value is None or isinstance(value, float)
    return value


def test_option_map_is_deterministic_and_rejects_ambiguous_text() -> None:
    assert match_option("Yes", OPTIONS) == YES
    assert match_option("No", OPTIONS) == NO
    assert match_option("Would purchase", OPTIONS) == YES
    assert match_option("Would not purchase", OPTIONS) == NO
    assert match_option("I would purchase this product", OPTIONS) == YES
    assert match_option("I would not purchase this product.", OPTIONS) == NO
    assert match_option(YES, OPTIONS) == YES
    assert match_option("Would", OPTIONS) is None
    assert match_option("Would purchase — extra commentary", OPTIONS) is None
    assert match_option("I tabloid would purchase this product", OPTIONS) is None
    codebook = {NO: 0, YES: 1}
    assert code_purchase("Yes, I would purchase this product.", codebook) == 1
    assert code_purchase("No, I would not purchase", codebook) == 0
    assert code_purchase("Would", codebook) is None


def test_perfect_match_rates_correlation_delta_and_sd_ratio() -> None:
    outcomes = [
        _human("s1", {"q1": YES, "q2": YES}),
        _human("s2", {"q1": YES, "q2": NO}),
        _human("s3", {"q1": NO, "q2": NO}),
    ]
    responses = [
        _reply("s1", "q1", "Yes"),
        _reply("s1", "q2", "Would purchase"),
        _reply("s2", "q1", YES),
        _reply("s2", "q2", "Would not purchase"),
        _reply("s3", "q1", "No"),
        _reply("s3", "q2", NO),
    ]
    scored = score_pricing_persons(responses, outcomes)
    assert scored is not None
    assert _value(scored, "individual_accuracy_micro") == 1.0
    assert _value(scored, "individual_accuracy_macro") == 1.0
    assert _value(scored, "cross_person_pearson") == pytest.approx(1.0)
    assert _value(scored, "cross_person_spearman") == pytest.approx(1.0)
    assert _value(scored, "glass_delta") == pytest.approx(0.0)
    assert _value(scored, "sd_ratio") == pytest.approx(1.0)
    assert scored["coverage"]["n_complete_subjects"] == 3
    assert scored["coverage"]["n_questions_in_support"] == 2


def test_alignment_ignores_row_order() -> None:
    outcomes = [
        _human("b", {"q2": NO, "q1": YES}),
        _human("a", {"q1": NO, "q2": YES}),
    ]
    ordered = [
        _reply("a", "q1", "No"),
        _reply("a", "q2", "Yes"),
        _reply("b", "q1", "Yes"),
        _reply("b", "q2", "No"),
    ]
    shuffled = [ordered[2], ordered[0], ordered[3], ordered[1]]
    assert score_pricing_persons(ordered, outcomes) == score_pricing_persons(shuffled, list(reversed(outcomes)))


def test_group_top1_is_not_individual_accuracy() -> None:
    outcomes = [
        _human("1", {"q": YES}),
        _human("2", {"q": YES}),
        _human("3", {"q": NO}),
    ]
    responses = [
        _reply("1", "q", YES),
        _reply("2", "q", NO),
        _reply("3", "q", YES),
    ]
    person = score_pricing_persons(responses, outcomes)
    assert person is not None
    assert _value(person, "individual_accuracy_micro") == pytest.approx(1 / 3)
    dataset = score_dataset("twin2k_500", responses, outcomes)
    assert dataset["categorical"]["mean_top1_acc"] == 1.0
    assert dataset["pricing_person"]["individual_accuracy_micro"]["value"] == pytest.approx(1 / 3)


def test_identical_duplicates_do_not_inflate_and_conflicts_are_excluded() -> None:
    outcomes = [_human("1", {"q": YES}), _human("2", {"q": NO})]
    duplicated = [
        _reply("1", "q", "Yes"),
        _reply("1", "q", "Yes"),
        _reply("2", "q", "No"),
    ]
    scored = score_pricing_persons(duplicated, outcomes)
    assert scored is not None
    assert scored["coverage"]["n_scorable_pairs"] == 2
    assert scored["coverage"]["n_identical_duplicate_rows"] == 1
    assert _value(scored, "individual_accuracy_micro") == 1.0

    conflict = [
        _reply("1", "q", "Yes"),
        _reply("1", "q", "No"),
        _reply("2", "q", "No"),
    ]
    conflicted = score_pricing_persons(conflict, outcomes)
    assert conflicted is not None
    assert conflicted["coverage"]["n_duplicate_conflicts"] == 1
    assert conflicted["coverage"]["n_scorable_pairs"] == 1
    assert conflicted["coverage"]["n_complete_subjects"] == 1


def test_unmapped_and_parse_errors_count_against_accuracy_and_block_rates() -> None:
    outcomes = [
        _human("1", {"q1": YES, "q2": NO}),
        _human("2", {"q1": NO, "q2": YES}),
    ]
    responses = [
        _reply("1", "q1", "Would"),
        _reply("1", "q2", "No"),
        _reply("2", "q1", "No", parse_error="bad json"),
        _reply("2", "q2", "Yes"),
    ]
    scored = score_pricing_persons(responses, outcomes)
    assert scored is not None
    coverage = scored["coverage"]
    assert coverage["n_unmapped_model"] == 1
    assert coverage["n_parse_error_pairs"] == 1
    assert coverage["n_scorable_pairs"] == 4
    assert coverage["n_correct_pairs"] == 2
    assert _value(scored, "individual_accuracy_micro") == pytest.approx(0.5)
    assert coverage["n_complete_subjects"] == 0
    assert _value(scored, "cross_person_pearson") is None
    assert "fewer than 2" in str(scored["cross_person_pearson"]["reason"])


def test_missing_question_keeps_partial_accuracy_out_of_the_rate() -> None:
    outcomes = [
        _human("1", {"q1": YES, "q2": NO}),
        _human("2", {"q1": NO, "q2": YES}),
    ]
    responses = [
        _reply("1", "q1", "Yes"),
        _reply("2", "q1", "No"),
        _reply("2", "q2", "Yes"),
    ]
    scored = score_pricing_persons(responses, outcomes)
    assert scored is not None
    assert scored["coverage"]["n_missing_model"] == 1
    assert scored["coverage"]["n_complete_subjects"] == 1
    person = next(row for row in scored["per_person"] if row["subject_id"] == "1")
    assert person["complete"] is False
    assert person["accuracy"] == 1.0
    assert person["purchase_rate_model"] is None
    assert _value(scored, "glass_delta") is None


def test_positive_glass_delta_and_compressed_sd_ratio() -> None:
    outcomes = [
        _human("1", {"q1": YES, "q2": YES, "q3": YES, "q4": YES}),
        _human("2", {"q1": YES, "q2": YES, "q3": NO, "q4": NO}),
        _human("3", {"q1": NO, "q2": NO, "q3": NO, "q4": NO}),
    ]
    responses = [
        _reply("1", "q1", "Yes"),
        _reply("1", "q2", "Yes"),
        _reply("1", "q3", "Yes"),
        _reply("1", "q4", "No"),
        _reply("2", "q1", "Yes"),
        _reply("2", "q2", "Yes"),
        _reply("2", "q3", "No"),
        _reply("2", "q4", "No"),
        _reply("3", "q1", "Yes"),
        _reply("3", "q2", "No"),
        _reply("3", "q3", "No"),
        _reply("3", "q4", "No"),
    ]
    scored = score_pricing_persons(responses, outcomes)
    assert scored is not None
    assert _value(scored, "glass_delta") == pytest.approx(0.0)
    assert _value(scored, "sd_ratio") == pytest.approx(0.5)
    assert _value(scored, "cross_person_pearson") == pytest.approx(1.0)

    over = [
        _reply(subject, question, "Yes")
        for subject in ("1", "2")
        for question in ("q1", "q2")
    ]
    over_scored = score_pricing_persons(
        over,
        [_human("1", {"q1": YES, "q2": YES}), _human("2", {"q1": NO, "q2": NO})],
    )
    assert over_scored is not None
    assert _value(over_scored, "glass_delta") == pytest.approx(0.5 / (0.5**0.5))
    assert _value(over_scored, "sd_ratio") == pytest.approx(0.0)
    assert over_scored["cross_person_pearson"]["value"] is None
    assert over_scored["cross_person_pearson"]["reason"]


def test_zero_human_variance_is_null_not_a_fake_number() -> None:
    outcomes = [_human("1", {"q": YES}), _human("2", {"q": YES}), _human("9", {"q": NO})]
    responses = [_reply("1", "q", "Yes"), _reply("2", "q", "No")]
    scored = score_pricing_persons(responses, outcomes)
    assert scored is not None
    for name in ("cross_person_pearson", "cross_person_spearman", "glass_delta", "sd_ratio"):
        assert scored[name]["value"] is None
        assert scored[name]["reason"]
    blob = json.dumps(scored)
    assert "NaN" not in blob
    assert "Infinity" not in blob


def test_bdm_report_does_not_gain_a_pricing_block() -> None:
    responses = [
        {"answer": "4", "parse_error": None, "question_id": "local_field", "subject_id": "1"},
        {"answer": "TRUE", "parse_error": None, "question_id": "conclusion_purchased_product", "subject_id": "1"},
    ]
    outcomes = [
        {
            "items": [
                {"answer": "5", "question_id": "local_field"},
                {"answer": "TRUE", "question_id": "conclusion_purchased_product"},
            ],
            "subject_id": "1",
        }
    ]
    scored = score_dataset("bdm_lettuce", responses, outcomes)
    assert "pricing_person" not in scored
    assert set(scored) == {
        "categorical",
        "n_parse_errors",
        "n_scored",
        "purchase",
        "purchase_accuracy",
        "wtp",
    }


def test_score_run_adds_pricing_person_for_short_yes_no(tmp_path: Path) -> None:
    from popbench.evaluate.score import score_run

    responses = tmp_path / "sim" / "demo" / "twin2k_500"
    responses.mkdir(parents=True)
    lines = [
        _reply("7", "QID9_1", "Yes"),
        _reply("8", "QID9_1", "No"),
    ]
    (responses / "responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in lines),
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes" / "twin2k_500"
    outcomes.mkdir(parents=True)
    people = [
        _human("7", {"QID9_1": YES}),
        _human("8", {"QID9_1": NO}),
    ]
    (outcomes / "outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in people),
        encoding="utf-8",
    )
    report = score_run(
        tmp_path,
        {
            "datasets": ["twin2k_500"],
            "model": "demo",
            "outcomes_dir": "outcomes",
            "output": str(tmp_path / "report.json"),
            "responses_dir": "sim",
        },
    )
    pricing = report["datasets"]["twin2k_500"]["pricing_person"]
    assert pricing["individual_accuracy_micro"]["value"] == 1.0
    assert pricing["directions"]["glass_delta"] == "closer_to_zero"
    assert pricing["directions"]["sd_ratio"] == "closer_to_one"
    assert pricing["directions"]["individual_accuracy_micro"] == "higher_better"
