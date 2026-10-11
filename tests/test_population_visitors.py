"""Persona cards for every adult in each ATUS population group."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from popbench.cli import main
from popbench.dao.atus_populations import (
    ACTIVITY_LABELS,
    build_population_visitors,
    population_visitors_dir,
)
from popbench.dao.io import read_jsonl, write_jsonl
from popbench.dao.visitors import VisitorError
from popbench.simulate.client import ModelSettings
from popbench.simulate.visitors import simulate_population_personas


def test_build_population_visitors_writes_every_adult(tmp_path: Path):
    _write_group(
        tmp_path,
        "sex",
        "Male",
        [
            _adult(20, sex="Male", school="", work=400, personal_care=500),
            _adult(10, sex="Male", school="enrolled", work=100, personal_care=600),
        ],
    )
    _write_group(
        tmp_path,
        "sex",
        "Female",
        [_adult(30, sex="Female", age=22, age_band="18-29", school="not_enrolled", work=20)],
    )

    summary = build_population_visitors(tmp_path)

    assert summary["n_groups"] == 2
    assert summary["n_cards"] == 3
    assert summary["dir"] == population_visitors_dir(tmp_path)
    males = read_jsonl(tmp_path / "visitors" / "by_population" / "sex" / "Male" / "visitors.jsonl")
    females = read_jsonl(tmp_path / "visitors" / "by_population" / "sex" / "Female" / "visitors.jsonl")
    assert [row["demographics"]["TUCASEID"] for row in males] == [10, 20]
    assert [row["id"] for row in females] == ["visitor-sex-Female-30"]
    blank = males[1]
    assert blank["routine"]["work"] == 400
    assert blank["routine"]["personal_care"] == 500
    assert "School enrollment" not in blank["persona"]["text"]
    assert "State:" not in blank["persona"]["text"]
    assert "Personal income" not in blank["persona"]["text"]
    assert "Usual weekly work hours" not in blank["persona"]["text"]
    assert "Diary day (ATUS minutes, this respondent):" in blank["persona"]["text"]
    assert "- work: 400 minutes" in blank["persona"]["text"]
    assert blank["demographics"]["school"] is None
    assert blank["source"] == "atus-population"
    assert blank["persona"]["condition"] == "atus_population"
    enrolled = males[0]["persona"]["text"]
    assert "School enrollment: enrolled" in enrolled
    assert not (tmp_path / "visitors" / "by_population" / "race_ethnicity").exists()
    manifest = json.loads((tmp_path / "visitors" / "by_population" / "manifest.json").read_text())
    assert manifest["n_cards"] == 3
    assert (tmp_path / "visitors" / "by_population" / "sex" / "Male" / "example.md").is_file()


def test_build_population_visitors_requires_the_split(tmp_path: Path):
    with pytest.raises(VisitorError, match="split-populations"):
        build_population_visitors(tmp_path)


def test_simulate_population_personas_requires_built_cards(tmp_path: Path):
    with pytest.raises(VisitorError, match="build-population-visitors"):
        simulate_population_personas(tmp_path)


def test_simulate_population_personas_writes_gpt_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_population_panel(tmp_path)
    _write_gpt_cache(tmp_path)
    calls: list[str] = []

    def fake_settings() -> ModelSettings:
        return ModelSettings(
            base_url="https://api.deepseek.com",
            api_key="test",
            model="deepseek-flash",
            temperature=0,
            transport="deepseek",
        )

    def fake_chat(settings: ModelSettings, messages: list[dict[str, str]], *, attempts: int = 6) -> str:
        assert settings.model == "deepseek-flash"
        assert settings.transport == "deepseek"
        prompt = messages[-1]["content"]
        calls.append(prompt)
        if "Age: 23" in prompt:
            childhood = "Grew up sharing a bedroom in a walk-up."
        else:
            childhood = "Grew up helping in a family kitchen."
        return json.dumps(
            {
                "childhood": childhood,
                "schooling": "Finished the schooling that matches the education fact.",
                "work_path": "Pay stays modest and weekly hours are not on a fixed schedule.",
                "family": "Household arrangements follow the marital fact given above.",
                "moves": "The current state is unknown.",
                "turning_point": "A schedule change shows up in the diary minutes.",
            }
        )

    monkeypatch.setattr("popbench.simulate.visitors.model_settings_from_env", fake_settings)
    monkeypatch.setattr("popbench.simulate.visitors.chat_with_retry", fake_chat)

    assert main(["simulate-population-personas", "--data-dir", str(tmp_path)]) == 0
    assert len(calls) == 2

    folder = tmp_path / "visitors" / "by_population" / "sex" / "Male"
    rows = read_jsonl(folder / "visitors.jsonl")
    cards = (folder / "persona_cards.txt").read_text()
    example = (folder / "example.md").read_text()
    first, second = rows
    assert first["persona_text"].startswith("## Basics")
    assert "## Life history" in first["persona_text"]
    assert "## Typical day (ATUS minutes)" in first["persona_text"]
    assert "State: Unknown" in first["persona_text"]
    assert "Personal income: $50,000 or less" in first["persona_text"]
    assert "Usual weekly work hours: unknown" in first["persona_text"]
    assert "Education: high_school" in first["persona_text"]
    assert "Marital status: Married" in first["persona_text"]
    assert "Race: white_nonhispanic" in first["persona_text"]
    assert "- personal_care: 500 minutes" in first["persona_text"]
    assert "- work: 400 minutes" in first["persona_text"]
    assert "Grew up sharing a bedroom in a walk-up." in first["persona_text"]
    assert "Kept from the GPT-5.5 cache." not in first["persona_text"]
    assert first["persona"]["text"] == first["persona_text"]
    assert first["background"]["model"] == "deepseek-flash"
    assert first["background"]["seed"] == 0
    assert "Education: bachelor_or_higher" in second["persona_text"]
    assert "Marital status: Unknown" in second["persona_text"]
    assert "Race: hispanic" in second["persona_text"]
    assert "- work: 10 minutes" in second["persona_text"]
    assert "Grew up helping in a family kitchen." in second["persona_text"]
    assert second["background"]["model"] == "deepseek-flash"
    assert "You are being visited for a short interview" in cards
    assert "You are being visited for a short interview" in example
    assert "# visitor=visitor-atus-10 panel_index=0/2 draw_seed=0 state=None" in example
    assert "visitor-atus-20" not in example
    assert "# visitor=visitor-atus-20 panel_index=1/2 draw_seed=0 state=None" in cards
    assert example.strip() == cards.split("# visitor=visitor-atus-20")[0].strip()
    assert first["persona_id"] == "visitor-atus-10"
    assert first["background"]["visitor_id"] == "visitor-atus-10"

    age_rows = read_jsonl(
        tmp_path / "visitors" / "by_population" / "age_band" / "18-29" / "visitors.jsonl"
    )
    assert len(age_rows) == 1
    assert age_rows[0]["id"] == "visitor-age_band-18-29-10"
    assert age_rows[0]["persona_id"] == "visitor-atus-10"
    assert age_rows[0]["persona_text"] == first["persona_text"]
    adults = read_jsonl(tmp_path / "visitors" / "by_population" / "adults" / "personas.jsonl")
    assert [row["id"] for row in adults] == ["visitor-atus-10", "visitor-atus-20"]

    assert main(["simulate-population-personas", "--data-dir", str(tmp_path)]) == 0
    assert len(calls) == 2
    again = read_jsonl(folder / "visitors.jsonl")
    assert again[0]["persona_text"] == first["persona_text"]
    assert again[1]["persona_text"] == second["persona_text"]


def _write_gpt_cache(data_dir: Path) -> None:
    folder = data_dir / "visitors" / "by_population" / ".persona_cache" / "gpt-5.5"
    folder.mkdir(parents=True)
    text = "\n".join(
        [
            "## Basics",
            "Age: 23",
            "Sex: Male",
            "State: Unknown",
            "Education: high_school",
            "Marital status: Married",
            "Race: white_nonhispanic",
            "Personal income: $50,000 or less",
            "Usual weekly work hours: unknown",
            "",
            "## Life history",
            "Childhood: Kept from the GPT-5.5 cache.",
            "",
            "## Typical day (ATUS minutes)",
            "- personal_care: 500 minutes",
            "- work: 400 minutes",
        ]
    )
    payload = {
        "raw": "{}",
        "background": {
            "seed": 0,
            "visitor_id": "visitor-sex-Male-10",
            "model": "gpt-5.5",
            "childhood": "Kept from the GPT-5.5 cache.",
            "schooling": "Finished high school.",
            "work_path": "Modest pay.",
            "family": "Lives with a spouse.",
            "moves": "Current state is unknown.",
            "turning_point": "A day off shows up in the minutes.",
        },
        "persona_text": text,
    }
    (folder / "cached.json").write_text(json.dumps(payload) + "\n")


def _write_population_panel(data_dir: Path) -> None:
    folder = data_dir / "visitors" / "by_population" / "sex" / "Male"
    folder.mkdir(parents=True)
    write_jsonl(
        folder / "visitors.jsonl",
        [
            _visitor_row(
                10,
                age=23,
                education="high_school",
                partner="spouse",
                race="white_nonhispanic",
                work=400,
                personal_care=500,
            ),
            _visitor_row(
                20,
                age=41,
                education="bachelor_or_higher",
                partner="none",
                race="hispanic",
                work=10,
                personal_care=20,
            ),
        ],
    )
    manifest = {
        "source": "atus-population",
        "n_groups": 2,
        "n_cards": 3,
        "groups": [
            {
                "factor": "sex",
                "level": "Male",
                "n_people": 2,
                "panel": "sex/Male/visitors.jsonl",
            },
            {
                "factor": "age_band",
                "level": "18-29",
                "n_people": 1,
                "panel": "age_band/18-29/visitors.jsonl",
            },
        ],
    }
    (data_dir / "visitors" / "by_population" / "manifest.json").write_text(
        json.dumps(manifest) + "\n"
    )
    age = data_dir / "visitors" / "by_population" / "age_band" / "18-29"
    age.mkdir(parents=True)
    same_adult = _visitor_row(
        10,
        age=23,
        education="high_school",
        partner="spouse",
        race="white_nonhispanic",
        work=400,
        personal_care=500,
        factor="age_band",
        level="18-29",
    )
    write_jsonl(age / "visitors.jsonl", [same_adult])


def _visitor_row(
    case_id: int,
    *,
    age: int,
    education: str,
    partner: str,
    race: str,
    work: int,
    personal_care: int,
    factor: str = "sex",
    level: str = "Male",
) -> dict:
    routine = {name: 0.0 for name in ACTIVITY_LABELS}
    routine["work"] = float(work)
    routine["personal_care"] = float(personal_care)
    visitor_id = f"visitor-{factor}-{level}-{case_id}"
    return {
        "id": visitor_id,
        "source": "atus-population",
        "persona": {
            "id": visitor_id,
            "condition": "atus_population",
            "text": "base",
            "age": age,
            "sex": "Male",
            "state": None,
        },
        "routine": routine,
        "demographics": {
            "TUCASEID": case_id,
            "age": age,
            "sex": "Male",
            "education_level": education,
            "employment": "full_time",
            "children": "no_household_child",
            "partner": partner,
            "metro": "metropolitan",
            "school": "not_enrolled",
            "race_ethnicity": race,
        },
    }


def _write_group(data_dir: Path, factor: str, level: str, rows: list[dict]) -> None:
    folder = data_dir / "atus" / "by_population" / factor
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / f"{level}.csv", index=False)


def _adult(
    case_id: int,
    *,
    sex: str,
    age: int = 40,
    age_band: str = "30-49",
    school: str = "",
    work: int = 0,
    personal_care: int = 0,
) -> dict:
    row = {
        "TUCASEID": case_id,
        "TUFINLWGT": 1.0,
        "TEAGE": age,
        "sex": sex,
        "age_band": age_band,
        "education_level": "high_school",
        "employment": "full_time",
        "children": "no_household_child",
        "partner": "spouse",
        "metro": "metropolitan",
        "school": school,
        "race_ethnicity": "white_nonhispanic",
    }
    for name in ACTIVITY_LABELS:
        row[name] = 0
    row["work"] = work
    row["personal_care"] = personal_care
    return row
