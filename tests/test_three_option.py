"""Three-option refusal arm: letters, permutation restore, and isolated batched calls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from popbench.evaluate.arm_a import interpolate_p50, permutation_balance
from popbench.evaluate.options import NO_OPTION, YES_OPTION
from popbench.simulate.run import run_simulation
from popbench.simulate.three_option import (
    BUY_TEXT,
    NO_NEED_TEXT,
    TOO_EXPENSIVE_TEXT,
    parse_choice,
    perm_for,
    three_option_prompt,
    to_semantic,
)

STEM = (
    "Please consider the following product category: Dairy Products. "
    "Suppose you are in a grocery store, and you see the following product in that category: "
    "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks. "
    "The product is priced at: $5.00. Would you or would you not purchase this product?"
)
FREE_STEM = STEM.replace(
    "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks. The product is priced at: $5.00.",
    "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks. The product is priced at: $0.00.",
)


def test_option_strings_are_the_brief_wording_without_padding():
    assert len(BUY_TEXT) == 33
    assert len(NO_NEED_TEXT) == 30
    assert len(TOO_EXPENSIVE_TEXT) == 30
    assert "consider need" not in three_option_prompt(STEM, ("A", "B", "C")).lower()


def test_prompt_keeps_the_stem_and_appends_three_lines():
    order = ("C", "A", "B")
    prompt = three_option_prompt(STEM, order)
    assert prompt.startswith(STEM)
    assert "Options:\n" in prompt
    assert prompt.index("A) ") < prompt.index("B) ") < prompt.index("C) ")
    assert f"A) {TOO_EXPENSIVE_TEXT}" in prompt
    assert f"B) {BUY_TEXT}" in prompt
    assert f"C) {NO_NEED_TEXT}" in prompt
    assert "Answer with a single letter" not in prompt


def test_displayed_letter_a_on_order_cab_is_too_expensive():
    assert to_semantic("A", ("C", "A", "B")) == "too_expensive"
    assert to_semantic("B", ("C", "A", "B")) == "buy"
    assert to_semantic("C", ("C", "A", "B")) == "no_need"


def test_parser_uses_only_the_letter_patterns():
    assert parse_choice("A") == "A"
    assert parse_choice("A)") == "A"
    assert parse_choice("(A)") == "A"
    assert parse_choice("a") == "A"
    assert parse_choice("A.") == "A"
    assert parse_choice("**A**") == "A"
    assert parse_choice("Option A") == "A"
    assert parse_choice("answer: b") == "B"
    assert parse_choice("I do not need this product") is None
    assert parse_choice("No, it is too expensive for me") is None
    assert parse_choice("About the price") is None
    assert parse_choice("") is None


def test_permutation_is_deterministic_and_one_of_six():
    first = perm_for("915", "Oats")
    assert first == perm_for("915", "Oats")
    assert first in {
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "A", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
        ("C", "B", "A"),
    }


def test_upward_tier_blocks_p50_interpolation():
    blocked = interpolate_p50({0: 0.4, 1: 0.8, 2: 0.3})
    assert blocked["p50"] is None
    assert blocked["upward_tiers"][0]["from_tier"] == 0
    crossed = interpolate_p50({0: 0.8, 1: 0.4})
    assert crossed["p50"] == pytest.approx(15.0)
    assert crossed["upward_tiers"] == []


def test_permutation_balance_flags_a_skewed_sample():
    balanced = permutation_balance([perm for perm in (
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "A", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
        ("C", "B", "A"),
    ) for _ in range(100)])
    assert all(row["pass"] for row in balanced)
    skewed = permutation_balance([("A", "B", "C")] * 100)
    assert skewed[0]["pass"] is False


def test_three_option_is_one_batched_call_in_its_own_directory(tmp_path: Path):
    root = tmp_path
    _write_pricing(root)
    calls: list[dict] = []

    class FakeCR:
        def chat(self, messages, model="deepseek-v4-pro", temperature=None, timeout=300, reasoning_effort=None):
            calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "temperature": temperature,
                }
            )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "answers": [
                            {"question_id": "QID9_1", "answer": "A"},
                            {"question_id": "QID9_2", "answer": "I do not need this product"},
                        ]
                    }
                ),
                reasoning_content="",
            )

    config = {
        "arm": "three-option",
        "attempts": 1,
        "cards_dir": "cards",
        "datasets": ["twin2k_500"],
        "model": "gemini-3.8-flash",
        "outcomes_dir": "outcomes",
        "output_dir": "out",
        "questions_per_request": "all",
        "temperature": 0,
        "timeout": 30,
        "use_sample": True,
    }
    summary = run_simulation(root, config, client=FakeCR())
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["temperature"] == 0
    user = calls[0]["messages"][1]["content"]
    assert user.count("question_id:") == 2
    assert STEM in user
    assert FREE_STEM in user
    assert "SECRET_HUMAN" not in user
    assert "Do not explain" in calls[0]["messages"][0]["content"]
    target = Path(summary["responses"][0])
    assert target == root / "out" / "gemini-3.8-flash__three-option" / "twin2k_500" / "responses.jsonl"
    assert not (root / "out" / "gemini-3.8-flash").exists()
    assert not (root / "out" / "gemini-3.8-flash__reasoning-none").exists()
    assert not (root / "out" / "gemini-3.8-flash__reasoning-none__single-q").exists()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    by_id = {row["question_id"]: row for row in rows}
    paid = by_id["QID9_1"]
    assert paid["parsed_letter"] == "A"
    assert paid["semantic"] == to_semantic("A", paid["option_order"])
    assert paid["answer"] == (YES_OPTION if paid["semantic"] == "buy" else NO_OPTION)
    assert paid["price"] in {"5.00", "0.00"}
    assert paid["tier"] in {0, 10}
    free = by_id["QID9_2"]
    assert free["parsed_letter"] is None
    assert free["semantic"] is None
    assert free["unmapped_reason"] == "no A/B/C letter match"
    assert free["raw_text"] == "I do not need this product"
    assert free["answer"] == "I do not need this product"
    calls.clear()
    run_simulation(root, config, client=FakeCR())
    assert calls == []


def test_three_option_refuses_reasoning_and_single_question(tmp_path: Path):
    with pytest.raises(ValueError, match="omit reasoning_effort"):
        run_simulation(
            tmp_path,
            {
                "arm": "three-option",
                "cards_dir": "cards",
                "datasets": ["twin2k_500"],
                "model": "gemini-3.8-flash",
                "outcomes_dir": "outcomes",
                "output_dir": "out",
                "reasoning_effort": "none",
            },
            client=object(),
        )
    with pytest.raises(ValueError, match="one request"):
        run_simulation(
            tmp_path,
            {
                "arm": "three-option",
                "cards_dir": "cards",
                "datasets": ["twin2k_500"],
                "model": "gemini-3.8-flash",
                "outcomes_dir": "outcomes",
                "output_dir": "out",
                "questions_per_request": 1,
            },
            client=object(),
        )


def _write_pricing(root: Path) -> None:
    card = {
        "subject_id": "7",
        "text": "Sex: What is the sex that you were assigned at birth?: Female\n",
    }
    cards = root / "cards" / "twin2k_500"
    cards.mkdir(parents=True)
    (cards / "cards.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")
    (cards / "sample.json").write_text(json.dumps({"subject_ids": ["7"]}) + "\n", encoding="utf-8")
    outcome = {
        "dataset": "twin2k_500",
        "subject_id": "7",
        "items": [
            {"answer": "SECRET_HUMAN_PAID", "question_id": "QID9_1", "question_text": STEM},
            {"answer": "SECRET_HUMAN_FREE", "question_id": "QID9_2", "question_text": FREE_STEM},
        ],
    }
    outcomes = root / "outcomes" / "twin2k_500"
    outcomes.mkdir(parents=True)
    (outcomes / "outcomes.jsonl").write_text(json.dumps(outcome) + "\n", encoding="utf-8")
