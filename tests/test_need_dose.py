"""Full-sample need dose uses one sentence per arm and isolated directories."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from popbench.evaluate.arm_a import write_triplets
from popbench.evaluate.need_dose_stats import arm_fit, predict
from popbench.simulate.need_dose import DOSE_LENGTHS, DOSE_SENTENCES, assert_dose_lengths, inject_dose
from popbench.simulate.need_inject import NEED_SENTENCES
from popbench.simulate.run import run_simulation

STEM = (
    "Please consider the following product category: Dairy Products. "
    "Suppose you are in a grocery store, and you see the following product in that category: "
    "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks. "
    "The product is priced at: $5.00. Would you or would you not purchase this product?"
)
PRODUCT = "Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks"


def test_dose_lengths_are_exact_and_match_phase1_endpoints():
    assert_dose_lengths()
    assert [DOSE_LENGTHS[arm] for arm in ("N1", "N2", "N3", "N4", "BUD")] == [50, 49, 48, 51, 52]
    assert [len(DOSE_SENTENCES[arm]) for arm in DOSE_LENGTHS] == [50, 49, 48, 51, 52]
    assert DOSE_SENTENCES["N1"] == NEED_SENTENCES["low"]
    assert DOSE_SENTENCES["N4"] == NEED_SENTENCES["high"]
    assert "$20" in DOSE_SENTENCES["BUD"]


def test_dose_insertion_keeps_one_price_clause():
    prompt = inject_dose(STEM, "N2")
    sentence = DOSE_SENTENCES["N2"]
    assert f"$5.00. {sentence} Would you or would you not purchase this product?" in prompt
    assert prompt.split("Options:")[0].count("\n" + sentence) == 0
    assert prompt.count("The product is priced at: $5.00.") == 1
    assert "B) No, I would not purchase the product" in prompt


def test_arm_fit_recovers_a_negative_price_slope():
    rows = []
    for person in range(8):
        for tier, chance in ((0, 1), (5, 1), (10, 0)):
            rows.append({"pid": str(person), "tier": tier, "buy": chance})
    fit = arm_fit(rows)
    assert fit is not None
    assert fit["beta"] < 0
    assert fit["p0"] > fit["p200"]
    assert predict(0.0, -0.01486, 0) == 0.5


def test_need_dose_writes_only_its_arm_directory(tmp_path: Path):
    root = tmp_path
    _write_pricing(root)
    calls: list[dict] = []

    class FakeCR:
        def chat(self, messages, model="deepseek-v4-pro", temperature=None, timeout=300, reasoning_effort=None):
            calls.append({"reasoning_effort": reasoning_effort, "temperature": temperature, "user": messages[1]["content"]})
            return SimpleNamespace(
                content=json.dumps({"answers": [{"question_id": "QID9_1", "answer": "A"}]}),
                reasoning_content="",
            )

    config = {
        "arm": "need-dose",
        "attempts": 1,
        "cards_dir": "cards",
        "datasets": ["twin2k_500"],
        "model": "gemini-3.8-flash",
        "need_arm": "N3",
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
    assert DOSE_SENTENCES["N3"] in calls[0]["user"]
    assert "Sex: What is the sex" in calls[0]["user"].split("## Questions")[0]
    assert DOSE_SENTENCES["N3"] not in calls[0]["user"].split("## Questions")[0]
    target = Path(summary["responses"][0])
    assert target == root / "out" / "gemini-3.8-flash__need-dose" / "twin2k_500" / "N3" / "responses.jsonl"
    assert not (root / "out" / "gemini-3.8-flash").exists()
    assert not (root / "out" / "gemini-3.8-flash__need-inject").exists()
    row = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert row["need_arm"] == "N3"
    assert row["sentence"] == DOSE_SENTENCES["N3"]
    assert row["buy"] == 1
    assert row["price"] == "5.00"
    calls.clear()
    config["need_arm"] = "BUD"
    summary = run_simulation(root, config, client=FakeCR())
    bud = Path(summary["responses"][0])
    assert bud == root / "out" / "gemini-3.8-flash__budget" / "twin2k_500" / "responses.jsonl"
    assert json.loads(bud.read_text().splitlines()[0])["sentence"] == DOSE_SENTENCES["BUD"]
    assert len(list((root / "out" / "cache" / "gemini-3.8-flash__need-dose" / "N3").rglob("*.json"))) == 1
    assert len(list((root / "out" / "cache" / "gemini-3.8-flash__budget").rglob("*.json"))) == 1


def _write_pricing(root: Path) -> None:
    cards = root / "cards" / "twin2k_500"
    cards.mkdir(parents=True)
    (cards / "cards.jsonl").write_text(
        json.dumps({"subject_id": "7", "text": "Sex: What is the sex that you were assigned at birth?: Female\n"}) + "\n",
        encoding="utf-8",
    )
    (cards / "sample.json").write_text(json.dumps({"subject_ids": ["7"]}) + "\n", encoding="utf-8")
    outcomes = root / "outcomes" / "twin2k_500"
    outcomes.mkdir(parents=True)
    (outcomes / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "dataset": "twin2k_500",
                "subject_id": "7",
                "items": [{"answer": "SECRET", "question_id": "QID9_1", "question_text": STEM}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_triplets(
        root / "triplets.parquet",
        [{"pid": "7", "product_id": PRODUCT, "question_id": "QID9_1", "price": "5.00", "tier": 10}],
    )
