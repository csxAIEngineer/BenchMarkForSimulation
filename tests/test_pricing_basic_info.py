"""Exact basic-info cards, fixed-seed sampling, and distribution counts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from popbench.datasets.bdm_lettuce.fields import OUTCOME_COLUMNS, SURVEY_FIELDS
from popbench.pricing_basic_info.assemble import assemble_dataset
from popbench.pricing_basic_info.cards import CardBuildError, build_card
from popbench.pricing_basic_info.report import distribution_report
from popbench.pricing_basic_info.sample import cochran_n0, finite_population_n, planned_sample_size, stratified_sample
from popbench.pricing_basic_info.spec import (
    BDM_BASIC_COLUMNS,
    BDM_OUTCOME_COLUMNS,
    CARD_KEYS,
    FIELD_KEYS,
    MISSING_TOKEN,
    STATUS_MISSING,
    STATUS_PRESENT,
    TWIN_BASIC_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("QID12", "QID20", "QID21", "QID24")
STEMS = {
    "QID12": "What is the sex that you were assigned at birth?",
    "QID20": "In politics today, do you consider yourself a",
    "QID21": "Last year, what was your total family income from all sources, before taxes?",
    "QID24": "What is your current employment status?",
}
RULE = {"z": "1.96", "proportion": "0.5", "margin_of_error": "0.10"}

# Read from wave1_3_response_label.csv pid 1732 at Twin-2K revision
# f883165a3026fde855dfd448e0cd16443ab257b6. The older atus fixture for this
# pid fills other QIDs with codebook examples; those examples are not this row.
PID_1732 = {
    "QID11": "Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH)",
    "QID12": "Female",
    "QID13": "30-49",
    "QID14": "Some college, no degree",
    "QID15": "Other",
    "QID16": "Yes",
    "QID17": "Married",
    "QID18": "Nothing in particular",
    "QID19": "Never",
    "QID20": "Independent",
    "QID21": "Less than $30,000",
    "QID22": "Moderate",
    "QID23": "3",
    "QID24": "Self-employed",
}


def _row(pid: str, **fields: str) -> dict[str, str]:
    row = {"pid": pid}
    row.update(fields)
    return row


def _outcomes(pids: list[str], answer: str = "No, I would not purchase the product") -> dict:
    stimulus = "SECRET_STIMULUS Land O Lakes priced at $8.45 for this respondent only"
    return {
        pid: [
            {
                "answer": answer,
                "question_id": "QID9_1",
                "question_text": stimulus,
                "source_column": "wave4_Q_wave4_A",
            }
        ]
        for pid in pids
    }


def _assemble(rows: list[dict[str, str]], *, seed: int = 42, n: int | None = 2):
    pids = [row["pid"] for row in rows]
    return assemble_dataset(
        rows,
        dataset="twin2k_500",
        source_file="wave1_3_response_label.csv",
        columns=COLUMNS,
        stratify_column="QID21",
        question_text=STEMS,
        outcomes=_outcomes(pids),
        outcome_ids=("QID9_1",),
        forbidden_columns=("QID9_1", "persona_summary"),
        seed=seed,
        sample_rule=RULE,
        sample_size_override=n,
        eligibility_note="fixture",
    )


def test_bdm_columns_follow_the_existing_background_catalog_and_not_its_paraphrases():
    assert BDM_BASIC_COLUMNS == tuple(column for column, _key, _label in SURVEY_FIELDS)
    assert BDM_OUTCOME_COLUMNS == OUTCOME_COLUMNS


def test_card_copies_cells_and_keeps_refusal_and_blank_distinct():
    card = build_card(
        _row("7", QID12="Female", QID20="Prefer not to answer", QID21="  ", QID24="Part-time employment"),
        dataset="twin2k_500",
        source_file="wave1_3_response_label.csv",
        columns=COLUMNS,
        question_text=STEMS,
    )
    assert set(card) == set(CARD_KEYS)
    by_column = {field["source_column"]: field for field in card["basic_info"]}
    assert set(by_column["QID12"]) == set(FIELD_KEYS)
    assert by_column["QID12"]["value"] == "Female"
    assert by_column["QID12"]["status"] == STATUS_PRESENT
    assert by_column["QID20"]["value"] == "Prefer not to answer"
    assert by_column["QID20"]["source_question_text"] == STEMS["QID20"]
    assert by_column["QID21"]["value"] == MISSING_TOKEN
    assert by_column["QID21"]["status"] == STATUS_MISSING
    text = str(card["text"])
    assert "QID24: Part-time employment" in text
    assert "QID21: MISSING" in text
    assert "20" not in text.split("QID24: ", 1)[1]
    assert "over $50,000" not in text
    assert "Part-time employment" in text
    assert text.index("## Identity") < text.index("## Basics")
    assert "consider yourself a Republican" not in text
    assert "Prefer not to answer" in text


def test_card_drops_outcome_stimulus_and_blocks_a_newline_rewrite():
    row = _row("7", QID12="Female", QID20="Independent", QID21="Less than $30,000", QID24="Student")
    row["QID9_1"] = "SECRET_STIMULUS Land O Lakes priced at $8.45"
    row["persona_summary"] = "A generated biography that must not be copied."
    card = build_card(
        row,
        dataset="twin2k_500",
        source_file="wave1_3_response_label.csv",
        columns=COLUMNS,
        question_text=STEMS,
    )
    assert "SECRET_STIMULUS" not in str(card["text"])
    assert "biography" not in str(card["text"])
    assert [field["source_column"] for field in card["basic_info"]] == list(COLUMNS)
    with pytest.raises(CardBuildError, match="newline"):
        build_card(
            _row("8", QID12="Male\nextra", QID20="Independent", QID21="x", QID24="Student"),
            dataset="twin2k_500",
            source_file="wave1_3_response_label.csv",
            columns=COLUMNS,
            question_text=STEMS,
        )


def test_assemble_is_byte_identical_and_excludes_only_missing_outcomes():
    rows = [
        _row("2", QID12="Male", QID20="Republican", QID21="$100,000 or more", QID24="Full-time employment"),
        _row("10", QID12="Female", QID20="", QID21="", QID24="Home-maker"),
        _row("3", QID12="Female", QID20="Democrat", QID21="Less than $30,000", QID24="Unemployed"),
    ]
    first = _assemble(rows, seed=42, n=2)
    second = _assemble(rows, seed=42, n=2)
    assert first == second
    cards = [json.loads(line) for line in first["cards.jsonl"].splitlines()]
    assert [card["subject_id"] for card in cards] == ["2", "3", "10"]
    blank = next(card for card in cards if card["subject_id"] == "10")
    income = next(field for field in blank["basic_info"] if field["source_column"] == "QID21")
    assert income["value"] == MISSING_TOKEN
    assert "Home-maker" in blank["text"]
    assert "SECRET_STIMULUS" not in blank["text"]
    assert "## Life history" not in blank["text"]
    missing_outcome_rows = rows + [_row("11", QID12="Male", QID20="Independent", QID21="$100,000 or more", QID24="Student")]
    documents = assemble_dataset(
        missing_outcome_rows,
        dataset="twin2k_500",
        source_file="wave1_3_response_label.csv",
        columns=COLUMNS,
        stratify_column="QID21",
        question_text=STEMS,
        outcomes=_outcomes(["2", "10", "3"]),
        outcome_ids=("QID9_1",),
        forbidden_columns=("QID9_1",),
        seed=42,
        sample_rule=RULE,
        sample_size_override=2,
        eligibility_note="fixture",
    )
    exclusions = json.loads(documents["exclusions.json"])
    assert exclusions["excluded"] == [{"reason": "missing pricing outcomes", "subject_id": "11"}]
    assert exclusions["included"] == 3
    assert "11" not in documents["cards.jsonl"]


def test_same_seed_selects_the_same_ids_and_reports_counts_without_a_test():
    rows = []
    for index in range(1, 21):
        income = "Less than $30,000" if index <= 15 else "$100,000 or more"
        rows.append(_row(str(index), QID12="Female" if index % 2 else "Male", QID20="Independent", QID21=income, QID24="Student"))
    draw_a = stratified_sample(rows, stratify_column="QID21", sample_size=8, seed=42, dataset="twin2k_500")
    draw_b = stratified_sample(rows, stratify_column="QID21", sample_size=8, seed=42, dataset="twin2k_500")
    draw_c = stratified_sample(rows, stratify_column="QID21", sample_size=8, seed=7, dataset="twin2k_500")
    assert draw_a == draw_b
    assert draw_a["subject_ids"] != draw_c["subject_ids"]
    assert sum(draw_a["allocation"].values()) == 8
    assert draw_a["allocation"]["Less than $30,000"] == 6
    assert draw_a["allocation"]["$100,000 or more"] == 2
    report = distribution_report(rows, draw_a["subject_ids"], ["QID12", "QID21"])
    assert "p_value" not in json.dumps(report)
    assert "chi" not in json.dumps(report).lower()
    income = report["fields"]["QID21"]
    counts = {level["value"]: level["population_count"] for level in income["levels"]}
    assert counts == {"$100,000 or more": 5, "Less than $30,000": 15}
    sample_counts = {level["value"]: level["sample_count"] for level in income["levels"]}
    assert sample_counts == {"$100,000 or more": 2, "Less than $30,000": 6}
    assert income["total_variation_distance"] == "0.00000000"
    skewed = distribution_report(rows, [str(index) for index in range(1, 9)], ["QID21"])
    assert skewed["fields"]["QID21"]["total_variation_distance"] != "0.00000000"
    assert int(float(skewed["fields"]["QID21"]["max_abs_proportion_difference"]) * 100) > 0


def test_default_sample_size_is_the_finite_population_cochran_rule():
    assert cochran_n0("1.96", "0.5", "0.10") == 97
    assert finite_population_n(97, 2058) == 93
    assert finite_population_n(97, 610) == 84
    twin = planned_sample_size(2058)
    bdm = planned_sample_size(610)
    assert twin["sample_size"] == 93
    assert bdm["sample_size"] == 84
    assert planned_sample_size(2058, override=40)["sample_size"] == 40
    with pytest.raises(ValueError):
        planned_sample_size(10, override=11)


def test_package_source_does_not_import_a_model_or_the_expanded_card():
    root = ROOT / "src" / "popbench" / "pricing_basic_info"
    modules = list(root.glob("*.py"))
    assert modules
    imported: set[str] = set()
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                imported.add(node.module)
    assert "openai" not in imported
    assert "popbench.models.atus_expanded" not in imported
    assert "popbench.models.atus_expanded_skill" not in imported


def test_committed_pid_1732_card_matches_the_label_file_snapshot():
    path = ROOT / "experiments" / "pricing-basic-info" / "twin2k_500" / "cards.jsonl"
    manifest_path = ROOT / "experiments" / "pricing-basic-info" / "manifest.json"
    assert path.is_file(), "run scripts/build_pricing_basic_info.py before this test"
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        card = json.loads(line)
        if card["subject_id"] == "1732":
            found = card
            break
    assert found is not None
    assert [field["source_column"] for field in found["basic_info"]] == list(TWIN_BASIC_COLUMNS)
    for field in found["basic_info"]:
        assert field["value"] == PID_1732[field["source_column"]]
        assert field["status"] == STATUS_PRESENT
        assert f"{field['source_column']}: {field['value']}" in found["text"]
    assert "South" not in found["text"]
    assert "White" not in found["text"]
    assert "Never been married" not in found["text"]
    assert "## Life history" not in found["text"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["committed_outputs"]["twin2k_500/cards.jsonl"]["sha256"]
    import hashlib

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == recorded
    sample = json.loads((path.parent / "sample.json").read_text(encoding="utf-8"))
    assert sample["seed"] == 42
    assert sample["sample_size"] == 93
    assert sample["population_size"] == 2058
    bdm_sample = json.loads((ROOT / "experiments" / "pricing-basic-info" / "bdm_lettuce" / "sample.json").read_text(encoding="utf-8"))
    assert bdm_sample["sample_size"] == 84
    assert bdm_sample["population_size"] == 610
    bdm_cards = (ROOT / "experiments" / "pricing-basic-info" / "bdm_lettuce" / "cards.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(bdm_cards[0])
    assert first["subject_id"] == "1"
    age = next(field for field in first["basic_info"] if field["source_column"] == "survey_q01_age")
    resources = next(field for field in first["basic_info"] if field["source_column"] == "survey_q13_lack_of_resources")
    assert age["value"] == "32"
    assert resources["value"] == MISSING_TOKEN
    assert resources["status"] == STATUS_MISSING
    assert "Very important" in first["text"]
    assert "treatment" not in first["text"]
