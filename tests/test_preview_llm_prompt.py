"""Tests for the real-dataset LLM prompt preview tool."""

from pathlib import Path
import sys

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from preview_llm_prompt import (  # noqa: E402
    build_llm_payload,
    pick_visitor,
    preview_random_llm_prompt,
)

ROOT = TESTS_DIR.parent
DATA = ROOT / "data"


@pytest.mark.skipif(
    not (DATA / "visitors" / "acs" / "acs_income_2018.parquet").is_file(),
    reason="ACS/ATUS visitor cache not present",
)
def test_preview_random_llm_prompt_from_real_panel():
    payload = preview_random_llm_prompt(DATA, seed=11, n_panel=50, panel_seed=0)
    assert payload["visitor_id"]
    assert payload["panel_size"] >= 1
    assert 0 <= payload["panel_index"] < payload["panel_size"]
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    system = messages[0]["content"]
    assert "## Basics" in system or "Life history" in system
    assert "Visit question 1 of" in messages[1]["content"]
    assert "===== SYSTEM =====" in payload["preview_text"]
    assert payload["background"]["visitor_id"] == payload["visitor_id"]


@pytest.mark.skipif(
    not (DATA / "visitors" / "v0" / "visitors_n50.jsonl").is_file()
    and not (DATA / "visitors" / "acs" / "acs_income_2018.parquet").is_file(),
    reason="visitor panel/cache missing",
)
def test_same_seed_picks_same_visitor():
    a = preview_random_llm_prompt(DATA, seed=3, n_panel=50, panel_seed=0)
    b = preview_random_llm_prompt(DATA, seed=3, n_panel=50, panel_seed=0)
    assert a["visitor_id"] == b["visitor_id"]
    assert a["panel_index"] == b["panel_index"]
    assert a["preview_text"] == b["preview_text"]


def test_pick_visitor_is_in_range():
    rows = [{"id": f"v{i}"} for i in range(10)]
    row, index = pick_visitor(rows, seed=0)
    assert row["id"] == rows[index]["id"]
    assert 0 <= index < 10


def test_build_llm_payload_shape():
    row = {
        "id": "visitor-test-0001",
        "state": "Ohio",
        "demographics": {
            "age": 41,
            "sex": "Female",
            "education": "some_college",
            "marital_status": "Married",
            "race": "White alone",
            "income_over_50k": False,
            "work_hours": 28,
        },
        "routine": {"work": 120.0, "leisure": 200.0, "personal_care": 500.0},
        "persona": {"text": "Age: 41"},
    }
    payload = build_llm_payload(row, seed=2)
    assert payload["messages"][0]["role"] == "system"
    assert "Ohio" in payload["messages"][0]["content"]
    assert payload["turn"]["turn"] == 1
