"""Need-state injection keeps the two-option stem and isolates its directory."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from popbench.evaluate.arm_a import write_triplets
from popbench.evaluate.arm_b import bootstrap_d_buy, d_buy, e7_pass
from popbench.simulate.need_inject import (
    NEED_SENTENCES,
    inject_need,
    need_state,
    parse_two_option,
    sentence_lengths_ok,
)
from popbench.simulate.run import run_simulation

STEM = (
    "Please consider the following product category: Dairy Products. "
    "Suppose you are in a grocery store, and you see the following product in that category: "
    "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks. "
    "The product is priced at: $5.00. Would you or would you not purchase this product?"
)
FREE_STEM = STEM.replace("$5.00.", "$0.00.")
PRODUCT = "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks"


def test_need_sentences_are_50_or_51_characters():
    assert sentence_lengths_ok()
    assert {len(text) for text in NEED_SENTENCES.values()} <= {50, 51}
    assert NEED_SENTENCES["fill"] == "This product is kept on a shelf by the store exit."


def test_need_state_is_stable_and_uses_the_need_salt():
    assert need_state("915", PRODUCT) == need_state("915", PRODUCT)
    assert need_state("915", PRODUCT) in {"high", "low", "fill"}


def test_insertion_is_inline_between_price_and_the_question():
    prompt = inject_need(STEM, "high")
    sentence = NEED_SENTENCES["high"]
    assert f"$5.00. {sentence} Would you or would you not purchase this product?" in prompt
    assert "\n" + sentence not in prompt.split("Options:")[0]
    assert "Options:\nA) Yes, I would purchase the product\nB) No, I would not purchase the product\n" in prompt
    assert "consider need" not in prompt.lower()


def test_parser_keeps_yes_no_and_letters_and_ignores_need_words():
    assert parse_two_option("Yes, I would purchase the product") == "A"
    assert parse_two_option("No, I would not purchase the product") == "B"
    assert parse_two_option("A") == "A"
    assert parse_two_option("B)") == "B"
    assert parse_two_option("You ran out of it last week, so you have none left.") is None
    assert parse_two_option("I need this") is None


def test_e7_requires_a_positive_gap_whose_interval_excludes_zero():
    rows = []
    for subject in ("1", "2", "3", "4"):
        rows.append({"pid": subject, "need_state": "high", "buy": 1, "tier": 0})
        rows.append({"pid": subject, "need_state": "low", "buy": 0, "tier": 0})
    assert d_buy(rows) == 1
    boot = bootstrap_d_buy(rows, seed=42, replicates=50)
    assert e7_pass(boot["d_buy"], boot["ci_low"], boot["ci_high"])
    assert e7_pass(-0.1, -0.2, -0.01) is False
    assert e7_pass(0.1, -0.01, 0.2) is False


def test_need_inject_is_one_batched_call_outside_existing_directories(tmp_path: Path):
    root = tmp_path
    _write_pricing(root)
    calls: list[dict] = []

    class FakeCR:
        def chat(self, messages, model="deepseek-v4-pro", temperature=None, timeout=300, reasoning_effort=None):
            calls.append({"messages": messages, "reasoning_effort": reasoning_effort, "temperature": temperature})
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "answers": [
                            {"question_id": "QID9_1", "answer": "A"},
                            {"question_id": "QID9_2", "answer": "I still have plenty"},
                        ]
                    }
                ),
                reasoning_content="",
            )

    config = {
        "arm": "need-inject",
        "attempts": 1,
        "cards_dir": "cards",
        "datasets": ["twin2k_500"],
        "model": "gemini-3.8-flash",
        "outcomes_dir": "outcomes",
        "output_dir": "out",
        "questions_per_request": "all",
        "temperature": 0,
        "timeout": 30,
        "triplets": "triplets.parquet",
        "use_sample": True,
    }
    summary = run_simulation(root, config, client=FakeCR())
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["temperature"] == 0
    user = calls[0]["messages"][1]["content"]
    system = calls[0]["messages"][0]["content"]
    assert "Put only the answer in answer." in system
    assert user.count("question_id:") == 2
    assert "Sex: What is the sex" in user
    high = NEED_SENTENCES["high"]
    assert high not in user.split("## Questions")[0]
    assert f"$5.00. {NEED_SENTENCES[need_state('7', PRODUCT)]} Would you" in user
    assert f"$0.00. {NEED_SENTENCES[need_state('7', PRODUCT)]} Would you" in user
    target = Path(summary["responses"][0])
    assert target == root / "out" / "gemini-3.8-flash__need-inject" / "twin2k_500" / "responses.jsonl"
    assert not (root / "out" / "gemini-3.8-flash").exists()
    assert not (root / "out" / "gemini-3.8-flash__three-option").exists()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    by_id = {row["question_id"]: row for row in rows}
    assert by_id["QID9_1"]["parsed_choice"] == "A"
    assert by_id["QID9_1"]["buy"] == 1
    assert by_id["QID9_1"]["need_state"] == need_state("7", PRODUCT)
    assert by_id["QID9_1"]["price"] == "5.00"
    assert by_id["QID9_2"]["parsed_choice"] is None
    assert by_id["QID9_2"]["raw_text"] == "I still have plenty"
    assert by_id["QID9_2"]["unmapped_reason"]
    calls.clear()
    run_simulation(root, config, client=FakeCR())
    assert calls == []


def _write_pricing(root: Path) -> None:
    cards = root / "cards" / "twin2k_500"
    cards.mkdir(parents=True)
    (cards / "cards.jsonl").write_text(
        json.dumps({"subject_id": "7", "text": "Sex: What is the sex that you were assigned at birth?: Female\n"}) + "\n",
        encoding="utf-8",
    )
    (cards / "sample.json").write_text(json.dumps({"subject_ids": ["7"]}) + "\n", encoding="utf-8")
    outcome = {
        "dataset": "twin2k_500",
        "subject_id": "7",
        "items": [
            {"answer": "SECRET_HUMAN", "question_id": "QID9_1", "question_text": STEM},
            {"answer": "SECRET_HUMAN", "question_id": "QID9_2", "question_text": FREE_STEM},
        ],
    }
    outcomes = root / "outcomes" / "twin2k_500"
    outcomes.mkdir(parents=True)
    (outcomes / "outcomes.jsonl").write_text(json.dumps(outcome) + "\n", encoding="utf-8")
    write_triplets(
        root / "triplets.parquet",
        [
            {"pid": "7", "product_id": PRODUCT, "question_id": "QID9_1", "price": "5.00", "tier": 10},
            {"pid": "7", "product_id": PRODUCT, "question_id": "QID9_2", "price": "0.00", "tier": 0},
        ],
    )
