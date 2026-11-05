"""Pricing-only CR benchmark wiring. No network and no full jsonl."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from popbench.comms.errors import CommsTransportError  # noqa: E402
from popbench.datasets.bdm_lettuce.loader import bundle_from_records as bdm_bundle  # noqa: E402
from popbench.datasets.twin2k_500.loader import bundle_from_records as twin_bundle  # noqa: E402
from pricing_models_cr import (  # noqa: E402
    BDM_PRICING_IDS,
    METHOD,
    MODEL_IDS,
    OMITTED_FROM_TABLE,
    TABLE_COLUMNS,
    assert_models_documented,
    assert_uniform_question_count,
    filter_questions,
    is_bdm_pricing,
    is_twin_pricing,
    job_is_complete,
    planned_batches,
    public_table_row,
)
from run_pricing_models_cr import BatchProgress  # noqa: E402
from run_skill_compare_cr import bdm_income, stratified_ids, twin_income  # noqa: E402


def _twin_question(question_id: str) -> dict:
    return {
        "id": question_id,
        "prompt": f"Consider {question_id}",
        "options": ["Yes, I would purchase the product", "No, I would not purchase the product"],
        "kind": "multiple_choice",
        "gold": "Yes, I would purchase the product",
    }


def _twin_record(subject_id: str, income: str, question_ids: list[str]) -> dict:
    return {
        "id": subject_id,
        "demographics": {"QID21": income},
        "demographic_text": income,
        "persona_text": "persona",
        "persona_summary": "summary",
        "questions": [_twin_question(question_id) for question_id in question_ids],
    }


def test_model_ids_are_documented_cr_slugs() -> None:
    assert MODEL_IDS == ("deepseek-v4-pro", "gpt-5.5", "glm-5.3")
    assert METHOD == "atus_expanded"
    assert_models_documented()


def test_pricing_filter_drops_non_pricing_items() -> None:
    twin = twin_bundle([_twin_record("7", "$100,000 or more", ["QID9_1", "QID1", "QID9_2"])])
    priced = filter_questions(twin, is_twin_pricing)
    assert [case.question.id for case in priced.cases] == ["QID9_1", "QID9_2"]
    assert [subject.id for subject in priced.subjects] == ["7"]

    bdm = bdm_bundle(
        [
            {
                "id": "3",
                "demographics": {"income": "$150,000 CAD and above", "age": "30"},
                "demographic_text": "Age: 30",
                "questions": [
                    {"id": "local_field", "prompt": "wtp", "options": [], "kind": "numeric", "gold": "2.5"},
                    {"id": "not_pricing", "prompt": "other", "options": ["A"], "kind": "text", "gold": "A"},
                    {
                        "id": "conclusion_purchased_product",
                        "prompt": "buy?",
                        "options": ["TRUE", "FALSE"],
                        "kind": "multiple_choice",
                        "gold": "FALSE",
                    },
                ],
            }
        ]
    )
    priced_bdm = filter_questions(bdm, is_bdm_pricing)
    assert [case.question.id for case in priced_bdm.cases] == ["local_field", "conclusion_purchased_product"]
    assert "not_pricing" not in BDM_PRICING_IDS


def test_uniform_question_count_rejects_a_short_subject() -> None:
    bundle = filter_questions(
        twin_bundle([_twin_record("7", "$50,000", ["QID9_1"])]),
        is_twin_pricing,
    )
    with pytest.raises(ValueError, match="expected 40"):
        assert_uniform_question_count(bundle, 40)


def test_stratified_sample_is_stable_for_seed() -> None:
    records = [
        _twin_record(str(index), "$10,000" if index % 2 == 0 else "$100,000 or more", ["QID9_1"])
        for index in range(20)
    ]
    bundle = twin_bundle(records)
    first, first_meta = stratified_ids(bundle.subjects, income_fn=twin_income, n=6, seed=42)
    second, second_meta = stratified_ids(bundle.subjects, income_fn=twin_income, n=6, seed=42)
    other, _other_meta = stratified_ids(bundle.subjects, income_fn=twin_income, n=6, seed=7)
    assert first == second
    assert first_meta["seed"] == 42
    assert first_meta["n_selected"] == 6
    assert other != first


def test_bdm_income_field_is_income() -> None:
    bundle = bdm_bundle(
        [
            {
                "id": "3",
                "demographics": {"income": "under 5"},
                "questions": [],
            }
        ]
    )
    assert bdm_income(bundle.subjects[0]) == "under 5"


def test_batch_plan_matches_pricing_shapes() -> None:
    assert planned_batches(50, 40, 15) == 150
    assert planned_batches(50, 7, 15) == 50
    assert planned_batches(50, 40, 15) * 3 + planned_batches(50, 7, 15) * 3 == 600


def test_public_table_omits_operational_counts() -> None:
    metrics = {
        "mean_tvd": 0.2,
        "mean_top1_acc": 0.5,
        "indiv_acc": 0.7,
        "fisher_z_corr": 0.1,
        "glass_delta": 0.0,
        "sd_ratio": 1.0,
        "accuracy": 0.4,
        "n_scored": 50,
        "parse_errors": 3,
        "items": [{"item_id": "QID9_1"}],
    }
    row = public_table_row("twin2k_500", "glm-5.3", metrics)
    assert list(row) == ["dataset", "model", *TABLE_COLUMNS]
    for key in OMITTED_FROM_TABLE:
        assert key not in row
    assert "items" not in row


def test_transport_failure_does_not_emit_rows() -> None:
    class Boom:
        def complete_batch(self, packages, *, sample_seed: int = 0):
            raise CommsTransportError("timed out")

    seen: list[str] = []
    progress = BatchProgress(
        Boom(),
        on_batch=lambda *_args: seen.append("batch"),
        on_error=lambda packages, exc: seen.append(str(exc)),
    )
    package = SimpleNamespace(subject_id="7", question_id="QID9_1")
    assert progress.complete_batch([package], sample_seed=42) == ()
    assert progress.failed == 1
    assert seen == ["timed out"]


def test_job_is_complete_requires_responses_and_manifest(tmp_path: Path) -> None:
    assert job_is_complete(tmp_path) is False
    (tmp_path / "responses.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
    assert job_is_complete(tmp_path) is False
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert job_is_complete(tmp_path) is True
    (tmp_path / "responses.jsonl").write_text("", encoding="utf-8")
    assert job_is_complete(tmp_path) is False
