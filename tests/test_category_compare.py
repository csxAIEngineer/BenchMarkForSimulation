"""ATUS personas scored separately on each Twin-2K paper category."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from popbench.cli import _categories_root, main
from popbench.dao.atus_categories import category_records
from popbench.dao.interview import InterviewError
from popbench.dao.io import write_jsonl
from popbench.dao.survey import PAPER_FAMILIES, build_closed_items, items_by_family
from popbench.dao.twin2k import catalog_dir
from popbench.dao.visitors import VisitorError
from popbench.evaluate.categories import (
    build_category_comparison,
    evaluate_category_run,
    render_category_comparison,
    write_category_comparison,
)
from popbench.simulate.client import ModelSettings
from popbench.simulate.survey import answer_survey

ROOT = Path(__file__).resolve().parents[1]

CLOSED_ITEM_COUNTS = {
    "demographics": 14,
    "personality": 275,
    "cognitive": 63,
    "economic": 176,
    "heuristics": 20,
    "experiments": 66,
    "pricing": 40,
}


def test_items_by_family_restarts_turns_inside_each_category():
    items = [
        {"item_id": "a", "family": "demographics", "turn": 9},
        {"item_id": "b", "family": "pricing", "turn": 10},
        {"item_id": "c", "family": "demographics", "turn": 11},
    ]
    grouped = items_by_family(items)
    assert [row["turn"] for row in grouped["demographics"]] == [1, 2]
    assert [row["item_id"] for row in grouped["demographics"]] == ["a", "c"]
    assert grouped["pricing"][0]["turn"] == 1
    assert items[0]["turn"] == 9
    assert grouped["cognitive"] == []
    with pytest.raises(InterviewError, match="unknown family"):
        items_by_family([{"item_id": "z", "family": "bfi44", "turn": 1}])


def test_category_records_use_the_persona_card_and_only_these_items():
    personas = [
        {
            "id": "atus-0-0000",
            "text": "Age: 40\nSex: Female",
            "demographics": {"age_band": "30-49", "sex": "Female"},
        }
    ]
    items = [
        {
            "turn": 1,
            "item_id": "Q9",
            "kind": "choice",
            "family": "pricing",
            "block": "Product Preferences - Pricing",
            "instrument": "",
            "prompt": "Would you buy this?",
            "options": ["Yes", "No"],
        }
    ]
    records = category_records(personas, items)
    assert records[0]["id"] == "atus-0-0000"
    assert records[0]["persona"]["text"] == "Age: 40\nSex: Female"
    assert [turn["item_id"] for turn in records[0]["turns"]] == ["Q9"]
    assert records[0]["turns"][0]["family"] == "pricing"


def test_comparison_ranks_inside_each_category_without_a_pooled_score(tmp_path: Path):
    summaries = [
        _summary("small-model", "heuristics", tvd=0.1, n_items=2),
        _summary("big-model", "heuristics", tvd=0.5, n_items=2),
        _summary("small-model", "personality", tvd=0.9, n_items=200),
        _summary("big-model", "personality", tvd=0.2, n_items=200),
    ]
    report = build_category_comparison(summaries)
    assert "mean_tvd" not in report
    assert "ranked" not in report
    assert [row["family"] for row in report["categories"]] == ["personality", "heuristics"]
    by_family = {row["family"]: row for row in report["categories"]}
    assert [row["model"] for row in by_family["heuristics"]["ranked"]] == [
        "small-model",
        "big-model",
    ]
    assert [row["model"] for row in by_family["personality"]["ranked"]] == [
        "big-model",
        "small-model",
    ]
    text = render_category_comparison(report)
    assert "合计" not in text
    assert "人格" in text
    assert "非实验启发式" in text
    assert "0.100（1）" in text
    assert "0.200（1）" in text
    written = write_category_comparison(tmp_path, summaries)
    assert written["kind"] == "atus_category_comparison"
    assert (tmp_path / "compare.md").is_file()


def test_answer_survey_records_an_atus_category(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_chat(_settings, messages):
        assert "Age: 40" in messages[0]["content"]
        return json.dumps({"answer": "Yes", "rationale": "fits"})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    records = category_records(
        [
            {
                "id": "atus-0-0000",
                "text": "Age: 40",
                "demographics": {"age_band": "30-49", "sex": "Female"},
            }
        ],
        [
            {
                "turn": 1,
                "item_id": "Q9",
                "kind": "choice",
                "family": "pricing",
                "block": "Product Preferences - Pricing",
                "instrument": "",
                "prompt": "Would you buy this?",
                "options": ["Yes", "No"],
            }
        ],
    )
    settings = ModelSettings(
        base_url="https://example.invalid",
        api_key="test",
        model="gpt-5.5",
        temperature=None,
        max_tokens=None,
        transport="cr",
    )
    answer_survey(records, tmp_path, settings=settings, panel="atus", seed=0, family="pricing")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["panel"] == "atus"
    assert meta["family"] == "pricing"
    assert meta["seed"] == 0
    items = records[0]["turns"]
    items[0]["encoding"] = "single"
    items[0]["column"] = "Q9"
    summary = evaluate_category_run(
        tmp_path,
        items,
        {
            "n_respondents": 4,
            "items": [
                {
                    "item_id": "Q9",
                    "kind": "choice",
                    "family": "pricing",
                    "n": 4,
                    "shares": [0.25, 0.75],
                    "mean_code": 1.75,
                }
            ],
        },
        family="pricing",
        seed=0,
    )
    assert summary["model"] == "gpt-5.5"
    assert summary["label"] == "定价"
    assert summary["n_items"] == 1
    assert summary["mean_tvd"] == pytest.approx(0.75)
    assert summary["mean_mae"] is None


def _write_atus_adults(data_dir: Path, rows: list[dict]) -> None:
    dest = data_dir / "visitors" / "atus" / "sum"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "atussum_2023.dat", index=False)


def _write_cards(data_dir: Path, cards: list[dict]) -> None:
    path = data_dir / "visitors" / "by_population" / "adults" / "personas.jsonl"
    write_jsonl(path, cards)


def test_category_panels_reuse_written_persona_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = catalog_dir(ROOT / "data") / "question_catalog.json"
    twin_root = ROOT / "data" / "twin2k"
    if not catalog.is_file():
        pytest.skip("Twin-2K question catalog is not on disk")
    (tmp_path / "twin2k").symlink_to(twin_root, target_is_directory=True)
    _write_atus_adults(
        tmp_path,
        [
            {"TUCASEID": 101, "TEAGE": 40, "TESEX": 2, "TUFINLWGT": 10},
            {"TUCASEID": 202, "TEAGE": 41, "TESEX": 2, "TUFINLWGT": 12},
        ],
    )
    card_a = "CARD-A life history for 101"
    card_b = "CARD-B life history for 202"
    _write_cards(
        tmp_path,
        [
            {
                "id": "visitor-atus-101",
                "TUCASEID": 101,
                "persona_text": card_a,
                "demographics": {"age": 40, "age_band": "30-49", "sex": "Female"},
            },
            {
                "id": "visitor-atus-202",
                "TUCASEID": 202,
                "persona_text": card_b,
                "demographics": {"age": 41, "age_band": "30-49", "sex": "Female"},
            },
        ],
    )

    def forbid_expand(*_args, **_kwargs):
        raise AssertionError("expand_persona should not run")

    monkeypatch.setattr("popbench.simulate.persona_background.expand_persona", forbid_expand)
    from popbench.dao.atus_categories import build_atus_category_panels
    from popbench.dao.atus_twin2k import build_atus_personas

    out = tmp_path / "out"
    built = build_atus_category_panels(
        tmp_path,
        n=2,
        seed=0,
        out_dir=out,
        families=["heuristics", "pricing"],
    )
    personas = build_atus_personas(tmp_path, n=2, seed=0)
    assert [person["id"] for person in built["personas"]] == [person["id"] for person in personas]
    texts = {person["id"]: person["text"] for person in built["personas"]}
    assert texts == {"visitor-atus-101": card_a, "visitor-atus-202": card_b}
    by_family = {panel["family"]: panel for panel in built["panels"]}
    assert len(by_family["heuristics"]["items"]) == CLOSED_ITEM_COUNTS["heuristics"]
    assert len(by_family["pricing"]["items"]) == CLOSED_ITEM_COUNTS["pricing"]
    assert {item["family"] for item in by_family["heuristics"]["items"]} == {"heuristics"}
    heuristic_ids = {turn["item_id"] for turn in by_family["heuristics"]["records"][0]["turns"]}
    pricing_ids = {turn["item_id"] for turn in by_family["pricing"]["records"][0]["turns"]}
    assert heuristic_ids.isdisjoint(pricing_ids)
    for panel in built["panels"]:
        assert [record["persona"]["text"] for record in panel["records"]] == [
            texts[record["id"]] for record in panel["records"]
        ]
    assert (
        by_family["heuristics"]["records"][0]["persona"]["text"]
        == by_family["pricing"]["records"][0]["persona"]["text"]
    )
    assert by_family["heuristics"]["records"][0]["turns"][0]["turn"] == 1
    assert len(by_family["heuristics"]["baseline"]["items"]) == CLOSED_ITEM_COUNTS["heuristics"]
    assert (out / "families" / "pricing" / "human_baseline.json").is_file()
    manifest = json.loads((out / "families" / "heuristics" / "manifest.json").read_text())
    assert manifest["population"] is None
    assert manifest["sampling"] == "age_band_x_sex"


def test_missing_persona_card_names_simulate_population_personas(tmp_path: Path):
    from popbench.dao.atus_twin2k import build_atus_personas

    rows = [
        {"TUCASEID": 101, "TEAGE": 40, "TESEX": 2, "TUFINLWGT": 10},
        {"TUCASEID": 202, "TEAGE": 41, "TESEX": 2, "TUFINLWGT": 12},
    ]
    _write_atus_adults(tmp_path, rows)
    with pytest.raises(VisitorError, match="simulate-population-personas"):
        build_atus_personas(tmp_path, n=2, seed=0)
    _write_cards(
        tmp_path,
        [
            {
                "id": "visitor-atus-101",
                "TUCASEID": 101,
                "persona_text": "only one card",
                "demographics": {"age": 40, "age_band": "30-49", "sex": "Female"},
            }
        ],
    )
    with pytest.raises(VisitorError, match="simulate-population-personas"):
        build_atus_personas(tmp_path, n=2, seed=0)


def test_closed_items_match_the_seven_category_counts():
    path = catalog_dir(ROOT / "data") / "question_catalog.json"
    if not path.is_file():
        pytest.skip("Twin-2K question catalog is not on disk")
    catalog = json.loads(path.read_text())
    grouped = items_by_family(build_closed_items(catalog))
    assert list(grouped) == list(PAPER_FAMILIES)
    assert {family: len(rows) for family, rows in grouped.items()} == CLOSED_ITEM_COUNTS
    for family, rows in grouped.items():
        assert rows
        assert {row["family"] for row in rows} == {family}
        assert [row["turn"] for row in rows] == list(range(1, len(rows) + 1))


def test_population_panels_use_the_factor_directory(tmp_path: Path):
    catalog = catalog_dir(ROOT / "data") / "question_catalog.json"
    twin_root = ROOT / "data" / "twin2k"
    if not catalog.is_file():
        pytest.skip("Twin-2K question catalog is not on disk")
    (tmp_path / "twin2k").symlink_to(twin_root, target_is_directory=True)
    levels = {
        "bachelor_or_higher": [(11, 10.0)],
        "high_school": [(22, 10.0)],
        "less_than_high_school": [(33, 10.0)],
        "some_college": [(44, 10.0)],
    }
    _write_population_levels(tmp_path, "education_level", levels)
    summary = tmp_path / "visitors" / "atus" / "sum" / "atussum_2023.dat"
    summary.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"TUCASEID": 999, "TEAGE": 40, "TESEX": 1, "TUFINLWGT": 99}]
    ).to_csv(summary, index=False)
    _write_cards(
        tmp_path,
        [
            _population_card(case_id)
            for case_id in (11, 22, 33, 44, 999)
        ],
    )
    from popbench.dao.atus_categories import build_atus_category_panels

    out = tmp_path / "out"
    built = build_atus_category_panels(
        tmp_path,
        n=4,
        seed=0,
        out_dir=out,
        families=["heuristics", "pricing"],
        population="education_level",
    )
    assert [person["TUCASEID"] for person in built["personas"]] == [11, 22, 33, 44]
    assert built["population"] == "education_level"
    by_family = {panel["family"]: panel for panel in built["panels"]}
    assert [record["id"] for record in by_family["heuristics"]["records"]] == [
        record["id"] for record in by_family["pricing"]["records"]
    ]
    for family in ("heuristics", "pricing"):
        manifest = json.loads((out / "families" / family / "manifest.json").read_text())
        assert manifest["population"] == "education_level"
        assert manifest["sampling"] == "education_level"


def test_comparison_names_the_population_factor(tmp_path: Path):
    summaries = [
        _summary("small-model", "heuristics", tvd=0.1, n_items=2),
        _summary("big-model", "heuristics", tvd=0.5, n_items=2),
    ]
    report = build_category_comparison(summaries, population="education_level")
    assert report["population"] == "education_level"
    text = render_category_comparison(report)
    assert "data/atus/by_population/education_level/" in text
    written = write_category_comparison(tmp_path, summaries, population="education_level")
    assert written["population"] == "education_level"
    assert "education_level" in (tmp_path / "compare.md").read_text()
    plain = render_category_comparison(build_category_comparison(summaries))
    assert "by_population" not in plain
    assert "population" not in build_category_comparison(summaries)


def test_population_run_directory_names_the_factor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        run_dir=None,
        population="education_level",
        per_age_band=None,
        n=50,
        seed=0,
    )
    assert _categories_root(args) == (
        tmp_path / "runs" / "compare" / "atus-categories-education_level-n50-seed0"
    )
    args.population = None
    assert _categories_root(args) == tmp_path / "runs" / "compare" / "atus-categories-n50-seed0"
    args.per_age_band = 6
    args.seed = 1
    assert _categories_root(args) == tmp_path / "runs" / "compare" / "atus-categories-age-n6-seed1"


def test_temperature_and_recent_name_a_separate_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        run_dir=None,
        population="education_level",
        per_age_band=None,
        n=50,
        seed=0,
        temperature=None,
        recent=False,
    )
    assert _categories_root(args) == (
        tmp_path / "runs" / "compare" / "atus-categories-education_level-n50-seed0"
    )
    args.temperature = 0.8
    args.recent = True
    assert _categories_root(args) == (
        tmp_path / "runs" / "compare" / "atus-categories-education_level-n50-seed0-t0.8-recent"
    )
    args.batch = True
    assert _categories_root(args) == (
        tmp_path
        / "runs"
        / "compare"
        / "atus-categories-education_level-n50-seed0-t0.8-recent-batch"
    )
    args.run_dir = tmp_path / "kept"
    assert _categories_root(args) == tmp_path / "kept"


def test_comparison_records_temperature_and_recent_only_when_passed(tmp_path: Path):
    summaries = [
        _summary("small-model", "demographics", tvd=0.2, n_items=14),
        _summary("big-model", "demographics", tvd=0.4, n_items=14),
    ]
    plain = build_category_comparison(summaries)
    assert "temperature" not in plain
    assert "recent" not in plain
    assert "batch" not in plain
    plain_text = render_category_comparison(plain)
    assert "温度" not in plain_text
    assert "recent：" not in plain_text
    recent = {
        "draw": 'sha256("education-categories-recent:" + person_id) % 3',
        "kinds": ["中彩票", "日常琐事", "核爆炸"],
        "by_id": {"visitor-atus-1": "中彩票", "visitor-atus-2": "核爆炸"},
    }
    report = build_category_comparison(summaries, temperature=0.8, recent=recent, batch=True)
    assert report["temperature"] == 0.8
    assert report["batch"] is True
    assert report["recent"]["by_id"] == recent["by_id"]
    text = render_category_comparison(report)
    assert "- 请求：同一类里空白 instrument 的题目合成一次请求。\n" in text
    assert "- 温度：0.8\n" in text
    assert (
        "- recent：中彩票、日常琐事、核爆炸，按 "
        '`sha256("education-categories-recent:" + person_id) % 3` 抽取。\n'
    ) in text
    written = write_category_comparison(
        tmp_path,
        summaries,
        temperature=0.8,
        recent=recent,
    )
    saved = json.loads((tmp_path / "compare.json").read_text())
    assert saved["temperature"] == 0.8
    assert saved["recent"]["kinds"] == ["中彩票", "日常琐事", "核爆炸"]
    assert saved["recent"]["by_id"]["visitor-atus-1"] == "中彩票"
    assert "- 温度：0.8" in (tmp_path / "compare.md").read_text()
    assert written["temperature"] == 0.8


def test_population_and_per_age_band_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "compare-categories",
                "--population",
                "education_level",
                "--per-age-band",
                "5",
            ]
        )
    assert caught.value.code != 0
    assert "not allowed" in capsys.readouterr().err


def _population_card(case_id: int) -> dict:
    return {
        "id": f"visitor-atus-{case_id}",
        "TUCASEID": case_id,
        "persona_text": f"card {case_id}",
        "demographics": {"age": 40, "age_band": "30-49", "sex": "Female"},
    }


def _write_population_levels(
    data_dir: Path,
    factor: str,
    levels: dict[str, list[tuple[int, float]]],
) -> None:
    folder = data_dir / "atus" / "by_population" / factor
    folder.mkdir(parents=True, exist_ok=True)
    for level, rows in levels.items():
        pd.DataFrame(
            [
                {
                    "TUCASEID": case_id,
                    "TUFINLWGT": weight,
                    "TEAGE": 40,
                    "sex": "Female",
                    "age_band": "30-49",
                }
                for case_id, weight in rows
            ]
        ).to_csv(folder / f"{level}.csv", index=False)


def _summary(model: str, family: str, *, tvd: float, n_items: int) -> dict:
    return {
        "model": model,
        "family": family,
        "mean_tvd": tvd,
        "mean_mae": None,
        "n_items": n_items,
        "n_choice": n_items,
        "n_numeric": 0,
        "n_invalid": 0,
        "n_people": 4,
        "seed": 0,
    }
