from popbench.simulate.persona_background import expand_persona
from popbench.simulate.visitors import system_message


def test_expand_persona_is_deterministic_for_same_seed():
    demographics = {
        "age": 40,
        "sex": "Female",
        "state": "Ohio",
        "education": "bachelor_or_higher",
        "marital_status": "Married",
        "race": "White alone",
        "income_over_50k": True,
        "work_hours": 40,
    }
    routine = {"work": 200.0, "leisure": 120.0, "personal_care": 500.0}
    first = expand_persona(
        visitor_id="visitor-0-0001",
        demographics=demographics,
        routine=routine,
        base_text="",
        seed=7,
    )
    second = expand_persona(
        visitor_id="visitor-0-0001",
        demographics=demographics,
        routine=routine,
        base_text="",
        seed=7,
    )
    assert first["background"] == second["background"]
    assert first["persona_text"] == second["persona_text"]
    assert "Life history" in first["persona_text"]
    assert "Typical day (ATUS minutes)" in first["persona_text"]


def test_expand_persona_changes_with_seed_or_id():
    demographics = {
        "age": 25,
        "sex": "Male",
        "state": "Texas",
        "education": "high_school",
        "marital_status": "Never married",
        "race": "Black or African American alone",
        "income_over_50k": False,
        "work_hours": 20,
    }
    a = expand_persona(
        visitor_id="a",
        demographics=demographics,
        routine={"work": 100.0},
        base_text="",
        seed=0,
    )
    b = expand_persona(
        visitor_id="a",
        demographics=demographics,
        routine={"work": 100.0},
        base_text="",
        seed=1,
    )
    c = expand_persona(
        visitor_id="b",
        demographics=demographics,
        routine={"work": 100.0},
        base_text="",
        seed=0,
    )
    assert a["background"] != b["background"]
    assert a["background"] != c["background"]


def test_schooling_matches_education_band():
    demographics = {
        "age": 30,
        "sex": "Female",
        "state": "California",
        "education": "less_than_high_school",
        "marital_status": "Never married",
        "race": "Asian alone",
        "income_over_50k": False,
        "work_hours": 15,
    }
    expanded = expand_persona(
        visitor_id="x",
        demographics=demographics,
        routine={},
        base_text="",
        seed=3,
    )
    schooling = expanded["background"]["schooling"].lower()
    assert "bachelor" not in schooling
    assert "college" not in schooling or "credential" in schooling


def test_system_message_mentions_life_history():
    text = system_message("## Basics\nAge: 20\n## Life history\nChildhood: x")
    assert "life history" in text
    assert "Age: 20" in text
