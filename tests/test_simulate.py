"""CR simulation requests use local cards and do not send held-out answers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cr_api.client import TEXT_MODELS
from popbench.simulate.prompt import build_messages, question_prompt
from popbench.simulate.run import known_model_ids, run_simulation


def test_cr_text_models_remain_the_simulation_catalog():
    assert "deepseek-v4-pro" in known_model_ids()
    assert known_model_ids() == {model for family in TEXT_MODELS.values() for model in family}


def test_prompt_keeps_the_card_and_drops_the_human_answer():
    item = {
        "answer": "SECRET_HUMAN_BID",
        "answer_position": 2,
        "question_id": "QID9_1",
        "question_text": "Would you buy oat milk at $3.40?",
    }
    prompt = question_prompt(item)
    messages = build_messages("## Identity\npid: 7\n", [("QID9_1", prompt)])
    blob = "\n".join(message["content"] for message in messages)
    assert "SECRET_HUMAN_BID" not in blob
    assert "answer_position" not in blob
    assert "Would you buy oat milk at $3.40?" in blob
    assert "pid: 7" in blob
    assert messages[0]["role"] == "system"


def test_bdm_purchase_prompt_uses_the_draw_and_not_the_decision():
    item = {
        "answer": "PURCHASED_SECRET",
        "question_id": "conclusion_purchased_product",
        "question_text": "",
        "stimulus": {
            "conclusion_generated_price": "1.25",
            "conclusion_selected_product": "local_field",
        },
    }
    prompt = question_prompt(item)
    assert "local_field" in prompt
    assert "1.25" in prompt
    assert "PURCHASED_SECRET" not in prompt
    assert "TRUE or FALSE" in prompt


def test_run_calls_cr_and_writes_local_responses(tmp_path: Path):
    root = tmp_path
    _write_inputs(root)
    calls: list[dict] = []

    class FakeCR:
        def chat(self, messages, model="deepseek-v4-pro", temperature=None, timeout=300):
            calls.append({"messages": messages, "model": model, "temperature": temperature})
            return SimpleNamespace(
                content=json.dumps(
                    {"answers": [{"question_id": "QID9_1", "answer": "Buy"}]},
                    ensure_ascii=False,
                )
            )

    summary = run_simulation(
        root,
        {
            "attempts": 1,
            "cards_dir": "cards",
            "datasets": ["twin2k_500"],
            "limit_subjects": 1,
            "model": "deepseek-v4-pro",
            "outcomes_dir": "outcomes",
            "output_dir": "out",
            "temperature": 0,
            "timeout": 30,
            "use_sample": True,
        },
        client=FakeCR(),
    )
    assert summary["provider"] == "cr"
    assert calls and calls[0]["model"] == "deepseek-v4-pro"
    user = calls[0]["messages"][1]["content"]
    persona, _questions = user.split("## Questions", 1)
    assert "Sex: What is the sex that you were assigned at birth?: Female" in persona
    assert "## Identity" not in persona
    assert "## Basics" not in persona
    assert "QID" not in persona
    assert "evidence" not in user
    assert "ONLY_IN_AUDIT" not in user
    assert "SECRET_HUMAN_BID" not in user
    written = Path(summary["responses"][0])
    row = json.loads(written.read_text(encoding="utf-8").splitlines()[0])
    assert row["answer"] == "Buy"
    assert row["parse_error"] is None
    assert "human" not in row
    again = run_simulation(
        root,
        {
            "attempts": 1,
            "cards_dir": "cards",
            "datasets": ["twin2k_500"],
            "limit_subjects": 1,
            "model": "deepseek-v4-pro",
            "outcomes_dir": "outcomes",
            "output_dir": "out",
            "temperature": 0,
            "timeout": 30,
            "use_sample": True,
        },
        client=FakeCR(),
    )
    assert len(calls) == 1
    assert again["responses"]


def test_unknown_model_is_rejected_before_a_call(tmp_path: Path):
    with pytest.raises(ValueError, match="TEXT_MODELS"):
        run_simulation(
            tmp_path,
            {
                "cards_dir": "cards",
                "datasets": ["twin2k_500"],
                "model": "not-a-cr-model",
                "outcomes_dir": "outcomes",
                "output_dir": "out",
            },
            client=object(),
        )


def _write_inputs(root: Path) -> None:
    card = {
        "basic_info": [
            {
                "evidence": {"qid": "QID12", "source_column": "QID12"},
                "qid": "QID12",
                "source_column": "QID12",
                "source_options": ["ONLY_IN_AUDIT"],
                "source_question_text": "What is the sex that you were assigned at birth?",
                "value": "Female",
            }
        ],
        "subject_id": "7",
        "text": "Sex: What is the sex that you were assigned at birth?: Female\n",
    }
    cards = root / "cards" / "twin2k_500"
    cards.mkdir(parents=True)
    (cards / "cards.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")
    sample = {"subject_ids": ["7"]}
    (cards / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    outcome = {
        "dataset": "twin2k_500",
        "items": [
            {
                "answer": "SECRET_HUMAN_BID",
                "question_id": "QID9_1",
                "question_text": "Would you buy oat milk at $3.40?",
            }
        ],
        "subject_id": "7",
    }
    outcomes = root / "outcomes" / "twin2k_500"
    outcomes.mkdir(parents=True)
    (outcomes / "outcomes.jsonl").write_text(json.dumps(outcome) + "\n", encoding="utf-8")
