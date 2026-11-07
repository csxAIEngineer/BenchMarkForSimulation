"""Held-out scoring does not call a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from popbench.evaluate.metrics import distribution_top1_accuracy, total_variation_distance
from popbench.evaluate.score import mean_tvd_by_item, score_dataset, score_run


def test_binary_top1_is_which_side_is_higher() -> None:
    assert distribution_top1_accuracy([0.2, 0.8], [0.45, 0.55]) == 1.0
    assert distribution_top1_accuracy([0.7, 0.3], [0.4, 0.6]) == 0.0


def test_tie_matches_only_on_a_shared_mode() -> None:
    assert distribution_top1_accuracy([1, 1], [1, 2]) == 1.0
    assert distribution_top1_accuracy([2, 1, 2], [1, 3, 1]) == 0.0


def test_counts_normalize_like_tvd() -> None:
    assert distribution_top1_accuracy([3, 1], [6, 2]) == 1.0
    assert total_variation_distance([3, 1], [6, 2]) == 0.0


def test_mean_top1_acc_averages_item_modes() -> None:
    pairs = {
        "price": [("buy", "buy"), ("buy", "not"), ("not", "buy")],
        "color": [("red", "blue"), ("red", "blue")],
    }
    result = mean_tvd_by_item(pairs)
    assert result["mean_top1_acc"] == 0.5
    assert result["n_items_top1"] == 2
    assert result["mean_tvd"] == pytest.approx(0.5)


def test_score_dataset_separates_wtp_from_purchase() -> None:
    responses = [
        {"answer": "4", "parse_error": None, "question_id": "local_field", "subject_id": "1"},
        {"answer": "6", "parse_error": None, "question_id": "local_field", "subject_id": "2"},
        {"answer": "TRUE", "parse_error": None, "question_id": "conclusion_purchased_product", "subject_id": "1"},
        {"answer": "FALSE", "parse_error": None, "question_id": "conclusion_purchased_product", "subject_id": "2"},
        {"answer": None, "parse_error": "missing", "question_id": "local_field", "subject_id": "3"},
    ]
    outcomes = [
        {
            "items": [
                {"answer": "5", "question_id": "local_field"},
                {"answer": "TRUE", "question_id": "conclusion_purchased_product"},
            ],
            "subject_id": "1",
        },
        {
            "items": [
                {"answer": "5", "question_id": "local_field"},
                {"answer": "TRUE", "question_id": "conclusion_purchased_product"},
            ],
            "subject_id": "2",
        },
    ]
    scored = score_dataset("bdm_lettuce", responses, outcomes)
    assert scored["n_parse_errors"] == 1
    assert scored["n_scored"] == 4
    assert scored["wtp"]["mae"] == pytest.approx(1.0)
    assert scored["wtp"]["indiv_acc"] == pytest.approx(0.9)
    assert scored["purchase_accuracy"] == 0.5


def test_score_run_writes_under_the_configured_path(tmp_path: Path) -> None:
    responses = tmp_path / "sim" / "deepseek-v4-pro" / "twin2k_500"
    responses.mkdir(parents=True)
    (responses / "responses.jsonl").write_text(
        json.dumps(
            {
                "answer": "Buy",
                "dataset": "twin2k_500",
                "model": "deepseek-v4-pro",
                "parse_error": None,
                "question_id": "QID9_1",
                "subject_id": "7",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes" / "twin2k_500"
    outcomes.mkdir(parents=True)
    (outcomes / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "items": [{"answer": "Buy", "question_id": "QID9_1"}],
                "subject_id": "7",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "outputs" / "evaluate" / "report.json"
    report = score_run(
        tmp_path,
        {
            "datasets": ["twin2k_500"],
            "model": "deepseek-v4-pro",
            "outcomes_dir": "outcomes",
            "output": str(report_path),
            "responses_dir": "sim",
        },
    )
    assert report_path.is_file()
    assert report["datasets"]["twin2k_500"]["categorical"]["mean_tvd"] == 0.0
    assert report["provider"] == "cr"
