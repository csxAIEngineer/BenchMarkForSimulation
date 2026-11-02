"""Canadian BDM lettuce loader and persona card. Bids stay on Gold."""

import json

import pytest

from popbench.cli import main
from popbench.datasets import resolve_dataset
from popbench.datasets.bdm_lettuce.fields import wtp_prompt
from popbench.datasets.bdm_lettuce.loader import (
    PUBLISHED_N,
    bundle_from_records,
    download,
    load,
    record_from_row,
)
from popbench.datasets.io import DatasetError, write_jsonl
from popbench.models.bdm_survey_persona import build
from popbench.models.errors import PersonaMethodError
from popbench.schema import Case, Gold, Question, Subject

BID = "8.88"
CSV = """pid,local_field,local_greenhouse,local_vertical,nonlocal_field,nonlocal_greenhouse,nonlocal_vertical,survey_q01_age,survey_q02_sex,survey_q15_income,survey_q16_education,survey_q17_household_size,survey_q06_primary_grocery,conclusion_selected_product,conclusion_generated_price,conclusion_purchased_product
7,{bid},3.10,4.20,1.00,1.50,2.00,32,Female,"$100,000-$149,999 CAD",Completed university,3,Yes,local_vertical,1.25,FALSE
8,2.00,2.10,2.20,2.30,2.40,2.50,41,Male,"$0-$14,999 CAD",Completed high school,2,No,local_field,0.50,TRUE
""".format(bid=BID)

def test_alias_and_published_size():
    assert resolve_dataset("canada_bdm") == "bdm_lettuce"
    assert resolve_dataset("canadian_bdm") == "bdm_lettuce"
    assert PUBLISHED_N == 610


def test_record_keeps_bids_off_the_persona():
    import csv
    import io

    row = next(csv.DictReader(io.StringIO(CSV)))
    record = record_from_row(row)
    assert record["id"] == "7"
    assert record["demographics"]["age"] == "32"
    assert record["demographics"]["income"] == "$100,000-$149,999 CAD"
    assert "local_field" not in record["demographics"]
    assert BID not in record["demographic_text"]
    assert "not a long narrative persona" in record["demographic_text"]
    bundle = bundle_from_records([record])
    assert bundle.unit == "individual"
    assert len(bundle.cases) == 7
    bids = {case.question.id: case.gold.answer for case in bundle.cases}
    assert bids["local_field"] == BID
    assert bids["conclusion_purchased_product"] == "FALSE"
    field = next(case for case in bundle.cases if case.question.id == "local_field")
    assert BID not in field.question.prompt
    assert BID not in (field.subject.demographic_text or "")
    purchase = next(case for case in bundle.cases if case.question.id == "conclusion_purchased_product")
    assert "1.25" in purchase.question.prompt
    assert BID not in purchase.question.prompt


def test_persona_method_does_not_read_gold_or_other_datasets():
    record = record_from_row(_first_row())
    case = bundle_from_records([record]).cases[0]
    poisoned = Subject(
        id=case.subject.id,
        dataset="bdm_lettuce",
        unit="individual",
        demographics=dict(case.subject.demographics),
        demographic_text=f"leaked bid {BID}",
    )
    case = Case(
        dataset="bdm_lettuce",
        subject=poisoned,
        question=case.question,
        gold=Gold(answer=BID),
    )
    package = build(case)
    assert package.method == "bdm_survey_persona"
    assert "## Basics" in package.user
    assert "## Life history" in package.user
    assert "Childhood:" in package.user
    assert "Age: 32" in package.user
    assert "Sex: Female" in package.user
    assert BID not in package.user
    assert BID not in package.system
    assert "1.25" not in package.user.split("Question:")[0]
    assert case.question.prompt in package.user
    again = build(case, seed=0)
    assert package.user == again.user
    variants = {build(case, seed=seed).user for seed in range(8)}
    assert len(variants) > 1

    other = Case(
        dataset="twin2k_500",
        subject=Subject(id="1", dataset="twin2k_500", unit="individual", demographics={"age": "32"}),
        question=Question(id="Q", prompt="Pick one", kind="text"),
    )
    with pytest.raises(PersonaMethodError):
        build(other)
    empty = Case(
        dataset="bdm_lettuce",
        subject=Subject(id="9", dataset="bdm_lettuce", unit="individual", demographics={}),
        question=Question(id="local_field", prompt=wtp_prompt("local field-grown lettuce"), kind="numeric"),
    )
    with pytest.raises(PersonaMethodError, match="does not invent"):
        build(empty)


def test_download_caches_the_csv_and_mock_simulate_runs(tmp_path, monkeypatch):
    bodies = {
        "1035856": CSV.encode("utf-8"),
        "1051155": b"readme",
        "1051154": b"docx",
    }

    def fake_get(url: str) -> bytes:
        for key, payload in bodies.items():
            if key in url:
                return payload
        raise AssertionError(url)

    monkeypatch.setattr("popbench.datasets.bdm_lettuce.loader._http_get", fake_get)
    path = download(tmp_path, limit=1)
    assert path.is_file()
    csv_path = tmp_path / "bdm_lettuce" / "20240723_VFExperiment_01.csv"
    assert csv_path.read_bytes().startswith(b"pid,")
    bundle = load(tmp_path)
    assert len(bundle.subjects) == 1
    assert bundle.cases[0].gold.answer == BID
    assert BID not in (bundle.subjects[0].demographic_text or "")

    with pytest.raises(DatasetError, match="expected 610"):
        download(tmp_path)

    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / "data" / "bdm_lettuce" / "respondents.jsonl", [record_from_row(_first_row())])
    code = main(
        [
            "simulate",
            "--dataset",
            "canada_bdm",
            "--method",
            "bdm_survey_persona",
            "--provider",
            "mock",
            "--limit",
            "2",
            "--data-dir",
            "data",
            "--out",
            "runs/bdm",
        ]
    )
    assert code == 0
    lines = (tmp_path / "runs" / "bdm" / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    saved = [json.loads(line) for line in lines]
    assert saved[0]["method"] == "bdm_survey_persona"
    assert saved[0]["parsed"]["answer"] == "mock"
    raw = (tmp_path / "runs" / "bdm" / "responses.jsonl").read_text(encoding="utf-8")
    assert BID not in raw


def _first_row() -> dict[str, str]:
    import csv
    import io

    return next(csv.DictReader(io.StringIO(CSV)))
