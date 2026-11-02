"""The only persona method is atus_expanded."""

from pathlib import Path

import pytest

from popbench.datasets.simbench.loader import bundle_from_records as simbench_bundle
from popbench.datasets.twin2k_500.loader import bundle_from_records as twin_bundle
from popbench.models import METHODS, build_prompt
from popbench.models.errors import PersonaMethodError

ROOT = Path(__file__).resolve().parents[1]


def _twin():
    bundle = twin_bundle(
        [
            {
                "id": "7",
                "demographics": {
                    "age": "40",
                    "sex": "Female",
                    "education": "high_school",
                    "marital_status": "Married",
                    "income_over_50k": "true",
                    "work_hours": "40",
                    "state": "Ohio",
                },
                "demographic_text": "Age: 40\nSex: Female",
                "persona_text": "Full card about this respondent.",
                "persona_summary": "A short published summary.",
                "routine": {"work": 480, "sleep": 420},
                "questions": [
                    {
                        "id": "Q9",
                        "prompt": "Pick one",
                        "options": ["red", "blue"],
                        "kind": "multiple_choice",
                        "gold": "SECRET_GOLD",
                    }
                ],
            }
        ]
    )
    return bundle.cases[0]


def _group():
    bundle = simbench_bundle(
        [
            {
                "group_key": "ESS:population",
                "group_prompt": "You are an ESS respondent in Europe.",
                "variables": {},
                "question_id": "trust",
                "prompt": "Trust most people?",
                "options": ["A", "B"],
                "distribution": {"A": 0.4, "B": 0.6},
                "group_size": 10,
            }
        ]
    )
    return bundle.cases[0]


def test_only_atus_expanded_is_registered():
    assert list(METHODS) == ["atus_expanded"]
    assert (ROOT / "src" / "popbench" / "models" / "atus_expanded").is_dir()
    for removed in (
        "demographics_only",
        "full_persona",
        "persona_summary",
        "group_prompt",
        "bdm_survey_persona",
    ):
        assert not (ROOT / "src" / "popbench" / "models" / removed).exists()


def test_models_do_not_import_http_clients():
    root = ROOT / "src" / "popbench" / "models"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "import httpx" not in text
    assert "cr_api" not in text
    assert "load_dataset" not in text


def test_atus_expanded_is_deterministic_and_keeps_the_same_person():
    case = _twin()
    first = build_prompt("atus_expanded", case, seed=3)
    second = build_prompt("atus_expanded", case, seed=3)
    other = build_prompt("atus_expanded", case, seed=4)
    assert first.user == second.user
    assert first.user != other.user
    assert "## Basics" in first.user
    assert "## Life history" in first.user
    assert "Age: 40" in first.user
    assert "Education: high_school" in first.user
    assert "work:" in first.user
    assert "SECRET_GOLD" not in first.user
    assert "Full card" not in first.user


def test_atus_expanded_refuses_simbench_groups():
    with pytest.raises(PersonaMethodError, match="SimBench"):
        build_prompt("atus_expanded", _group())


def test_missing_demographics_are_rejected():
    bundle = twin_bundle(
        [
            {
                "id": "9",
                "questions": [{"id": "Q", "prompt": "Hello", "kind": "text"}],
            }
        ]
    )
    with pytest.raises(PersonaMethodError, match="does not invent"):
        build_prompt("atus_expanded", bundle.cases[0])


def test_unknown_method_is_rejected():
    with pytest.raises(KeyError, match="demographics_only"):
        build_prompt("demographics_only", _twin())
