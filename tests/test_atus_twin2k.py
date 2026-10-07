"""ATUS persona cards answering a short Twin-2K multiple-choice set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from popbench.dao.atus_twin2k import (
    PROBE_N_TURNS,
    build_atus_twin2k,
    sample_atus_adults,
    select_probe_items,
)
from popbench.dao.interview import OPTIONS, build_instrument
from popbench.dao.schema import InterviewRecord, InterviewTurn, PersonaCard
from popbench.evaluate.atus_twin2k import (
    evaluate_atus_twin2k,
    render_atus_twin2k_comparison,
    write_atus_twin2k_comparison,
)
from popbench.simulate.client import ModelSettings
from popbench.simulate.interview import run_records, turns_user_message


def _atus_frame() -> pd.DataFrame:
    rows = []
    case = 1
    for age, sex in (
        (22, 1),
        (22, 1),
        (40, 1),
        (40, 1),
        (55, 1),
        (55, 1),
        (70, 1),
        (70, 1),
        (24, 2),
        (24, 2),
        (35, 2),
        (35, 2),
        (60, 2),
        (60, 2),
        (80, 2),
        (80, 2),
    ):
        rows.append(
            {
                "TUCASEID": case,
                "TEAGE": age,
                "TESEX": sex,
                "TUFINLWGT": 10 + case,
                "PEEDUCA": 43,
                "PTDTRACE": 1,
                "TELFS": 1,
                "TRSPPRES": 1,
                "TRERNWA": 200_000,
                "TRCHILDNUM": 0,
                "TEHRUSLT": 40,
                "GTMETSTA": 1,
                "TUDIARYDAY": 3,
                "t010101": 400,
                "t050101": 300,
            }
        )
        case += 1
    return pd.DataFrame(rows)


def test_sample_atus_adults_is_stable_and_stratified():
    frame = _atus_frame()
    first = sample_atus_adults(frame, n=8, seed=0)
    second = sample_atus_adults(frame, n=8, seed=0)
    other = sample_atus_adults(frame, n=8, seed=1)
    assert first["TUCASEID"].tolist() == second["TUCASEID"].tolist()
    assert first["TUCASEID"].tolist() != other["TUCASEID"].tolist()
    assert set(first["age_band"] + "|" + first["sex"]) == {
        "18-29|Male",
        "18-29|Female",
        "30-49|Male",
        "30-49|Female",
        "50-64|Male",
        "50-64|Female",
        "65+|Male",
        "65+|Female",
    }


def test_select_probe_items_keeps_one_forward_item_per_scale():
    items = []
    for number, scale in (
        (1, "extraversion"),
        (7, "agreeableness"),
        (3, "conscientiousness"),
        (4, "neuroticism"),
        (5, "openness"),
    ):
        items.append(
            {
                "turn": number,
                "item_id": f"QID25_{number}",
                "column": f"QID25_{number}",
                "scale": scale,
                "reverse": False,
                "prompt": f"Item {number}.",
            }
        )
    items.append(
        {
            "turn": 45,
            "item_id": "QID235_1",
            "column": "QID235_1",
            "scale": "green_values",
            "reverse": False,
            "prompt": "Green.",
        }
    )
    chosen = select_probe_items(items)
    assert len(chosen) == PROBE_N_TURNS
    assert [item["scale"] for item in chosen] == [
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
        "green_values",
    ]
    assert [item["turn"] for item in chosen] == [1, 2, 3, 4, 5, 6]
    assert chosen[0]["source_turn"] == 1


def test_probe_prompt_hides_gold_and_counts_six_questions():
    message = turns_user_message(
        {
            "turn": 2,
            "prompt": "I see myself as someone who is talkative.",
            "options": ["Disagree strongly", "Agree strongly"],
            "gold_answer": "SECRET GOLD",
            "gold_code": 5,
        },
        n_turns=6,
    )
    assert "Question 2 of 6." in message
    assert "SECRET GOLD" not in message


def test_lower_twin2k_distance_ranks_first(tmp_path):
    summaries = [
        _summary("gpt-5.5", 0.40, 1.2),
        _summary("glm-5.3", 0.10, 0.4),
        _summary("deepseek-v4-pro", 0.10, 0.2),
    ]
    report = write_atus_twin2k_comparison(tmp_path / "compare", summaries)
    assert [row["model"] for row in report["ranked"]] == [
        "deepseek-v4-pro",
        "glm-5.3",
        "gpt-5.5",
    ]
    text = render_atus_twin2k_comparison(report)
    assert text.index("deepseek-v4-pro") < text.index("glm-5.3")
    assert "Twin-2K" in text


def test_run_records_sends_the_persona_card(tmp_path, monkeypatch):
    seen: dict[str, str] = {}

    def fake_chat(_settings, messages):
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[-1]["content"]
        return json.dumps({"answer": "Agree a little", "rationale": "fits the diary"})

    monkeypatch.setattr("popbench.simulate.interview.chat_with_retry", fake_chat)
    record = InterviewRecord(
        id="atus-0-0000",
        region="US",
        language="en",
        source="ATUS-2023",
        instrument="twin2k_probe6",
        persona=PersonaCard(
            id="atus-0-0000",
            condition="atus_life_history",
            text="## Basics\nAge: 40\nSex: Female",
        ),
        turns=(
            InterviewTurn(
                turn=1,
                item_id="QID25_1",
                scale="extraversion",
                prompt="I see myself as someone who is talkative.",
                options=OPTIONS,
                reverse=False,
            ),
        ),
    )
    settings = ModelSettings(
        base_url="https://example.invalid",
        api_key="test",
        model="gpt-5.5",
        temperature=None,
        max_tokens=None,
        transport="cr",
    )
    run_records([record], tmp_path / "run", settings=settings, seed=0)
    assert "Age: 40" in seen["system"]
    assert "Question 1 of 1." in seen["user"]
    assert "talkative" in seen["user"]
    summary = evaluate_atus_twin2k(
        tmp_path / "run",
        [record],
        {
            "items": [
                {
                    "item_id": "QID25_1",
                    "turn": 1,
                    "scale": "extraversion",
                    "n": 10,
                    "shares": [0.2, 0.2, 0.2, 0.2, 0.2],
                    "mean_code": 3.0,
                }
            ],
            "scales": {
                scale: {"n": 1, "mean": 3.0, "std": 1.0}
                for scale in (
                    "extraversion",
                    "agreeableness",
                    "conscientiousness",
                    "neuroticism",
                    "openness",
                    "green_values",
                )
            },
        },
        seed=0,
    )
    assert summary["model"] == "gpt-5.5"
    assert summary["items"][0]["n"] == 1
    assert (tmp_path / "run" / "option_shares.json").is_file()


def test_build_atus_twin2k_from_cached_surveys(tmp_path):
    data_dir = Path("data")
    summary = data_dir / "visitors" / "atus" / "sum" / "atussum_2023.dat"
    catalog = data_dir / "twin2k" / "question_catalog_and_human_response_csv" / "question_catalog.json"
    if not summary.is_file() or not catalog.is_file():
        pytest.skip("ATUS and Twin-2K caches are not on disk")
    built = build_atus_twin2k(data_dir, n=4, seed=0, out_dir=tmp_path)
    again = build_atus_twin2k(data_dir, n=4, seed=0, out_dir=tmp_path / "again")
    assert [record.id for record in built["records"]] == [record.id for record in again["records"]]
    assert built["records"][0].persona.text == again["records"][0].persona.text
    record = built["records"][0]
    assert len(record.turns) == PROBE_N_TURNS
    assert all(turn.gold_code is None for turn in record.turns)
    assert "Age:" in record.persona.text
    assert "Diary day (ATUS minutes)" in record.persona.text
    assert "Household (ATUS respondent)" in record.persona.text
    assert len(built["baseline"]["items"]) == PROBE_N_TURNS
    assert len(built["baseline"]["items"][0]["shares"]) == 5
    instrument = build_instrument(json.loads(catalog.read_text()))
    assert [item["item_id"] for item in select_probe_items(instrument)] == [
        turn.item_id for turn in record.turns
    ]


def _summary(model: str, tvd: float, error: float) -> dict:
    return {
        "model": model,
        "mean_tvd": tvd,
        "mean_abs_code_error": error,
        "n_invalid": 0,
        "n_answers": 6,
        "n_people": 4,
        "seed": 0,
        "items": [
            {
                "item_id": "QID25_1",
                "scale": "extraversion",
                "scale_zh": "外向性",
                "prompt": "I see myself as someone who is talkative.",
                "tvd": tvd,
            }
        ],
    }
