"""Paper-category and Table 1 instrument mapping for Twin-2K questions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from popbench.dao.interview import InterviewError
from popbench.dao.survey import (
    INSTRUCTION,
    INSTRUMENT_FAMILIES,
    PAPER_FAMILIES,
    build_closed_items,
    families_by_question,
    family_for,
    instrument_for,
)
from popbench.dao.twin2k import CATALOG_QUESTIONS, catalog_dir

ROOT = Path(__file__).resolve().parents[1]

# QuestionIDs in the cached catalog. The README's "34 (+2 intro)" economic
# line omits two instruction screens inside the Economic preferences block
# (QID243, QID249), so that category is 36 + 2 intro = 38 and the seven
# categories sum to 256.
PAPER_QUESTION_COUNTS = {
    "demographics": 14,
    "personality": 49,
    "cognitive": 70,
    "economic": 38,
    "heuristics": 5,
    "experiments": 39,
    "pricing": 41,
}


def test_family_for_uses_published_block_names():
    assert family_for("Demographics") == "demographics"
    assert family_for("Personality ") == "personality"
    assert family_for("Cognitive tests ") == "cognitive"
    assert family_for("Forward Flow") == "cognitive"
    assert family_for("Economic preferences ") == "economic"
    assert family_for("Economic preferences - intro ") == "economic"
    assert family_for("Non-experimental heuristics and biases ") == "heuristics"
    assert family_for("Product Preferences - Pricing") == "pricing"
    assert family_for("Anchoring - African countries high") == "experiments"
    assert family_for("False consensus ") == "experiments"
    assert set(PAPER_FAMILIES) == set(PAPER_QUESTION_COUNTS)


def test_unknown_block_is_an_error():
    with pytest.raises(InterviewError, match="unknown Twin-2K block"):
        family_for("Behavioral economics experiments")
    with pytest.raises(InterviewError, match="unknown Twin-2K block"):
        family_for("")


def test_closed_items_reject_unknown_blocks_before_skipping_them():
    catalog = [
        {
            "QuestionID": "Q1",
            "BlockName": "Behavioral economics experiments",
            "QuestionType": "DB",
            "is_descriptive": True,
        }
    ]
    with pytest.raises(InterviewError, match="unknown Twin-2K block"):
        build_closed_items(catalog)


def test_catalog_questions_match_paper_counts():
    path = catalog_dir(ROOT / "data") / "question_catalog.json"
    if not path.is_file():
        pytest.skip("Twin-2K question catalog is not on disk")
    catalog = json.loads(path.read_text())
    classified = families_by_question(catalog)
    assert len(catalog) == CATALOG_QUESTIONS == len(classified) == 256
    assert Counter(row["family"] for row in classified) == PAPER_QUESTION_COUNTS
    assert {row["family"] for row in classified} == set(PAPER_FAMILIES)

    by_block: dict[str, list[dict[str, str]]] = {}
    for row in classified:
        by_block.setdefault(row["block"], []).append(row)
    assert len(by_block["Cognitive tests"]) == 69
    assert [row["question_id"] for row in by_block["Forward Flow"]] == ["QID10"]
    assert by_block["Forward Flow"][0]["family"] == "cognitive"
    assert len(by_block["Economic preferences"]) == 36
    assert len(by_block["Economic preferences - intro"]) == 2
    assert {row["family"] for row in by_block["Economic preferences"]} == {"economic"}
    assert {row["family"] for row in by_block["Economic preferences - intro"]} == {"economic"}
    assert sum(row["family"] == "experiments" for row in classified) == 39
    # The same QuestionID is reused in different blocks, so each entry keeps its own category.
    qid278 = [row for row in classified if row["question_id"] == "QID278"]
    assert {row["family"] for row in qid278} == {"cognitive", "economic"}


def test_closed_items_use_paper_family():
    path = catalog_dir(ROOT / "data") / "question_catalog.json"
    if not path.is_file():
        pytest.skip("Twin-2K question catalog is not on disk")
    catalog = json.loads(path.read_text())
    items = build_closed_items(catalog)
    assert items
    assert all(item["family"] == family_for(item["block"]) for item in items)
    assert {item["family"] for item in items} <= set(PAPER_FAMILIES)
    cognitive = [item for item in items if item["block"] == "Cognitive tests"]
    assert cognitive
    assert {item["family"] for item in cognitive} == {"cognitive"}
    # Forward Flow is an open word-association task, so it is classified but not asked.
    assert not any(item["block"] == "Forward Flow" for item in items)
    assert Counter(item["family"] for item in items) == {
        "demographics": 14,
        "personality": 275,
        "cognitive": 63,
        "economic": 176,
        "heuristics": 20,
        "experiments": 66,
        "pricing": 40,
    }


def _entry(block: str, text: str, **extra: object) -> dict:
    entry = {"BlockName": block, "QuestionType": "MC", "QuestionText": text}
    entry.update(extra)
    return entry


def test_descriptive_screens_are_instructions():
    entry = _entry(
        "Personality",
        "In this section we will ask you a few questions about your personality.",
        QuestionType="DB",
        is_descriptive=True,
    )
    assert instrument_for(entry) == INSTRUCTION


def test_shared_personality_opening_stays_three_scales():
    opening = (
        "Here are a number of characteristics that may or may not apply to you. "
        "Please indicate next to each statement the extent to which you agree or disagree "
        "with that statement."
    )
    assert instrument_for(_entry("Personality", opening + " I see myself as someone who...")) == "Big Five"
    assert (
        instrument_for(
            _entry(
                "Personality",
                opening,
                Rows=["I would prefer complex to simple problems"],
            )
        )
        == "Need for cognition"
    )
    assert (
        instrument_for(
            _entry(
                "Personality",
                opening,
                Rows=["It is important to me that the products I use do not harm the environment."],
            )
        )
        == "Green values"
    )


def test_cognitive_tasks_follow_table_1():
    assert instrument_for(_entry("Cognitive tests", "Emily's father has three daughters. The first two are named April and May.")) == "CRT"
    assert (
        instrument_for(
            _entry(
                "Cognitive tests",
                "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball.",
            )
        )
        == "Numeracy"
    )
    assert instrument_for(_entry("Cognitive tests", "Synonym: Select the word that is most nearly the same in meaning as CONCUR")) == "Crystallized intelligence"
    assert (
        instrument_for(
            _entry(
                "Cognitive tests",
                "Premise 1: If the winter is harsh, then there will be a flu epidemic.",
            )
        )
        == "Deductive certainty"
    )
    assert instrument_for(_entry("Cognitive tests", "Premise 1: All animals love water.")) == "Syllogisms"
    assert instrument_for(_entry("Forward Flow", "On this page, starting with the word paper, write the next word.")) == "Forward Flow"


def test_economic_lists_with_the_same_instructions_stay_apart():
    shared = "Please choose between the following options. For each line, choose left or right."
    assert instrument_for(_entry("Economic preferences", shared, QuestionType="Matrix", Rows=["$6.00 in 6 weeks:$3.00 in 5 weeks"])) == "Discount"
    assert instrument_for(_entry("Economic preferences", shared, QuestionType="Matrix", Rows=["$6.00 in 1 week:$3.00 now"])) == "Present bias"
    assert (
        instrument_for(
            _entry(
                "Economic preferences",
                "In this question, the LOTTERY is a 50% chance of winning $6. Please choose between the following options.",
            )
        )
        == "Risk aversion"
    )
    assert (
        instrument_for(
            _entry(
                "Economic preferences",
                "In this question, the LOTTERY is a 50% chance of LOSING $6. Please choose between the following options.",
            )
        )
        == "Loss aversion"
    )


def test_experiment_conditions_share_a_table_1_name():
    assert instrument_for(_entry("Anchoring - African countries high", "How many African countries do you think are in the United Nations?")) == "Anchoring"
    assert instrument_for(_entry("Anchoring - redwood low", "Is the tallest redwood tree more or less than 85 feet tall?")) == "Anchoring"
    assert instrument_for(_entry("Disease-loss", "Imagine that the U.S. is preparing for the outbreak of an unusual disease.")) == "Framing"
    highway = "Imagine that highway safety experts have determined that a substantial number of people are at risk of dying in a type of automobile fire."
    one = instrument_for(_entry("Proportion dominance 1A", highway + " A requirement would save the 150 people."))
    two = instrument_for(_entry("Proportion dominance 1B", highway + " A requirement would save 98% of the 150 people."))
    assert one == two == highway
    seatbelt = instrument_for(
        _entry(
            "Proportion dominance 2A",
            "You have recently graduated from university, obtained a good job, and are buying a new car. A seatbelt would save 500 drivers.",
        )
    )
    assert seatbelt.startswith("You have recently graduated")
    assert seatbelt != highway


def test_other_blocks_are_not_given_a_table_1_task():
    assert instrument_for(_entry("Demographics", "What is your age?")) == ""
    assert instrument_for(_entry("Non-experimental heuristics and biases", "Please rate the following products.")) == ""
    with pytest.raises(InterviewError, match="unknown Twin-2K block"):
        instrument_for(_entry("Behavioral economics experiments", "Choose one."))


def test_catalog_instruments_match_table_1():
    path = catalog_dir(ROOT / "data") / "question_catalog.json"
    if not path.is_file():
        pytest.skip("Twin-2K question catalog is not on disk")
    catalog = json.loads(path.read_text())
    classified = families_by_question(catalog)
    focus = [row for row in classified if row["family"] in INSTRUMENT_FAMILIES]
    assert len(focus) == 196
    assert all(row["instrument"] for row in focus)

    expected = _expected_instruments()
    non_experiments = [row for row in focus if row["family"] != "experiments"]
    got = {(row["question_id"], row["block"]): row["instrument"] for row in non_experiments}
    assert got == expected
    _assert_experiment_instruments(focus)

    for entry, row in zip(catalog, classified, strict=True):
        if entry.get("is_descriptive") or entry.get("QuestionType") == "DB":
            assert row["instrument"] == INSTRUCTION
        elif row["family"] not in INSTRUMENT_FAMILIES:
            assert row["instrument"] == ""

    items = build_closed_items(catalog)
    by_block = {}
    for row in classified:
        by_block.setdefault(row["block"], set()).add(row["instrument"])
    for item in items:
        assert item["instrument"] in by_block[item["block"]]
        assert item["instrument"] != INSTRUCTION
    assert {item["instrument"] for item in items if item["column"].startswith("QID25_")} == {"Big Five"}
    assert {item["instrument"] for item in items if item["column"] == "QID36"} == {"Financial literacy"}
    assert {item["instrument"] for item in items if item["column"] == "QID49_TEXT"} == {"Numeracy"}
    assert {item["instrument"] for item in items if item["block"].startswith("Anchoring")} == {"Anchoring"}


def _ids(instrument: str, block: str, *question_ids: str) -> dict[tuple[str, str], str]:
    return {(question_id, block): instrument for question_id in question_ids}


def _expected_instruments() -> dict[tuple[str, str], str]:
    """QuestionID and block to the Table 1 name, for the cached catalog."""
    personality = "Personality"
    cognitive = "Cognitive tests"
    economic = "Economic preferences"
    expected: dict[tuple[str, str], str] = {}
    groups = [
        _ids(INSTRUCTION, personality, "QID260", "QID262", "QID285", "QID127", "QID240"),
        _ids("Big Five", personality, "QID25"),
        _ids("Need for cognition", personality, "QID26"),
        _ids("Agentic vs. communal values", personality, "QID29"),
        _ids("Consumer minimalism", personality, "QID35"),
        _ids("Empathy", personality, "QID232"),
        _ids("Green values", personality, "QID235"),
        _ids("Social desirability", personality, "QID27"),
        _ids("Conscientiousness", personality, "QID30"),
        _ids("Anxiety", personality, "QID125"),
        _ids("Individualism vs. collectivism", personality, "QID233"),
        _ids("Selves questionnaire", personality, "QID268", "QID269", "QID270"),
        _ids("Regulatory focus", personality, "QID28"),
        _ids("Tightwads vs. spendthrift", personality, "QID31", "QID32", "QID33", "QID34"),
        _ids(
            "Depression",
            personality,
            "QID126",
            "QID128",
            "QID129",
            "QID130",
            "QID131",
            "QID132",
            "QID133",
            "QID134",
            "QID136",
            "QID137",
            "QID138",
            "QID139",
            "QID140",
            "QID141",
            "QID142",
            "QID143",
            "QID144",
            "QID145",
            "QID146",
            "QID147",
            "QID148",
        ),
        _ids("Need for uniqueness", personality, "QID234"),
        _ids("Self-monitoring", personality, "QID236"),
        _ids("Self-concept clarity", personality, "QID237"),
        _ids("Need for closure", personality, "QID238"),
        _ids("Maximization", personality, "QID239"),
        _ids(INSTRUCTION, cognitive, "QID241", "QID62", "QID73", "QID265"),
        _ids("CRT", cognitive, "QID52", "QID53", "QID54", "QID55"),
        _ids("Fluid intelligence", cognitive, "QID56", "QID57", "QID58", "QID59", "QID60", "QID61"),
        _ids(
            "Crystallized intelligence",
            cognitive,
            "QID63",
            "QID64",
            "QID65",
            "QID66",
            "QID67",
            "QID68",
            "QID69",
            "QID70",
            "QID71",
            "QID72",
            "QID74",
            "QID75",
            "QID76",
            "QID77",
            "QID78",
            "QID79",
            "QID80",
            "QID81",
            "QID82",
            "QID83",
        ),
        _ids(
            "Syllogisms",
            cognitive,
            "QID278",
            "QID268",
            "QID269",
            "QID270",
            "QID279",
            "QID271",
            "QID272",
            "QID273",
            "QID274",
            "QID275",
            "QID276",
            "QID277",
        ),
        _ids("Overconfidence", cognitive, "QID123"),
        _ids("Overplacement", cognitive, "QID124"),
        _ids("Financial literacy", cognitive, "QID36", "QID37", "QID38", "QID39", "QID40", "QID41", "QID42", "QID43"),
        _ids("Numeracy", cognitive, "QID44", "QID45", "QID46", "QID47", "QID48", "QID49", "QID50", "QID51"),
        _ids("Deductive certainty", cognitive, "QID217", "QID218", "QID219", "QID220"),
        _ids("Wason selection", cognitive, "QID221"),
        _ids("Forward Flow", "Forward Flow", "QID10"),
        _ids(INSTRUCTION, "Economic preferences - intro", "QID266", "QID242"),
        _ids(INSTRUCTION, economic, "QID243", "QID249"),
        _ids("Ultimatum game (sender)", economic, "QID224"),
        _ids("Ultimatum game (receiver)", economic, "QID225", "QID226", "QID227", "QID228", "QID229", "QID230"),
        _ids("Mental accounting", economic, "QID149", "QID150", "QID151", "QID152"),
        _ids("Discount", economic, "QID84", "QID244", "QID245"),
        _ids("Present bias", economic, "QID246", "QID247", "QID248"),
        _ids("Risk aversion", economic, "QID250", "QID251", "QID252"),
        _ids("Loss aversion", economic, "QID276", "QID277", "QID278", "QID279"),
        _ids("Trust game (sender)", economic, "QID117"),
        _ids("Trust game (receiver)", economic, "QID118", "QID119", "QID120", "QID121", "QID122"),
        _ids("Trust game (sender) thought listing", economic, "QID271"),
        _ids("Trust game (receiver) thought listing", economic, "QID272"),
        _ids("Dictator game", economic, "QID231"),
        _ids("Dictator game thought listing", economic, "QID275"),
    ]
    for group in groups:
        overlap = set(expected).intersection(group)
        if overlap:
            raise AssertionError(f"duplicate expected keys: {sorted(overlap)}")
        expected.update(group)
    return expected


_EXPERIMENT_BLOCKS = {
    "Absolute vs. relative - calculator": "Absolute vs. relative savings",
    "Absolute vs. relative - jacket": "Absolute vs. relative savings",
    "Allais Form 1": "Allais problem",
    "Allais Form 2": "Allais problem",
    "Anchoring - African countries high": "Anchoring",
    "Anchoring - African countries low": "Anchoring",
    "Anchoring - redwood high": "Anchoring",
    "Anchoring - redwood low": "Anchoring",
    "Base-rate 30 engineers": "Base rate",
    "Base-rate 70 engineers": "Base rate",
    "Disease - gain": "Framing",
    "Disease-loss": "Framing",
    "False consensus": "False consensus",
    "Less is More Gamble A": "Less is more",
    "Less is More Gamble B": "Less is more",
    "Less is More Gamble C": "Less is more",
    "Linda -no conjunction": "Conjunction (Linda)",
    "Linda-conjunction": "Conjunction (Linda)",
    "Myside Ford": "Myside bias",
    "Myside German": "Myside bias",
    "Outcome bias - failure": "Outcome bias",
    "Outcome bias - success": "Outcome bias",
    "Probability matching vs. maximizing - Problem 1": "Probability matching",
    "Probability matching vs. maximizing - Problem 2": "Probability matching",
    "Sunk cost - no": "Sunk cost",
    "Sunk cost - yes": "Sunk cost",
    "WTA/WTP Thaler - WTP noncertainty": "WTA/WTP",
    "WTA/WTP Thaler problem - WTA certainty": "WTA/WTP",
    "WTA/WTP Thaler problem - WTP certainty": "WTA/WTP",
}

_HIGHWAY_STEM = (
    "Imagine that highway safety experts have determined that a substantial number "
    "of people are at risk of dying in a type of automobile fire."
)
_SEATBELT_STEM = "You have recently graduated from university, obtained a good job, and are buying a new car."


def _assert_experiment_instruments(focus: list[dict[str, str]]) -> None:
    experiments = [row for row in focus if row["family"] == "experiments"]
    assert len(experiments) == 39
    seen: dict[str, str] = {}
    for row in experiments:
        block = row["block"]
        if block.startswith("Proportion dominance 1"):
            assert row["instrument"] == _HIGHWAY_STEM
        elif block.startswith("Proportion dominance 2"):
            assert row["instrument"] == _SEATBELT_STEM
        else:
            assert row["instrument"] == _EXPERIMENT_BLOCKS[block]
        previous = seen.get(block)
        if previous is not None:
            assert previous == row["instrument"]
        seen[block] = row["instrument"]
    assert set(seen) == set(_EXPERIMENT_BLOCKS) | {
        "Proportion dominance 1A",
        "Proportion dominance 1B",
        "Proportion dominance 1C",
        "Proportion dominance 2A",
        "Proportion dominance 2B",
        "Proportion dominance 2C",
    }
