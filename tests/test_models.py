"""Persona methods build prompt packages and do not call an LLM."""

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


def test_each_method_has_its_own_directory():
    for name in METHODS:
        assert (ROOT / "src" / "popbench" / "models" / name).is_dir()


def test_models_do_not_import_http_clients():
    root = ROOT / "src" / "popbench" / "models"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "import httpx" not in text
    assert "cr_api" not in text
    assert "load_dataset" not in text


def test_individual_methods_omit_gold_and_refuse_groups():
    case = _twin()
    group = _group()
    for name in ("demographics_only", "full_persona", "persona_summary", "atus_expanded"):
        kwargs = {"seed": 0} if name == "atus_expanded" else {}
        package = build_prompt(name, case, **kwargs)
        assert package.method == name
        assert "SECRET_GOLD" not in package.user
        assert package.response_mode == "answer"
        with pytest.raises(PersonaMethodError):
            build_prompt(name, group, **kwargs)


def test_method_text_uses_only_its_own_fields():
    case = _twin()
    demographic = build_prompt("demographics_only", case)
    assert "Age: 40" in demographic.user
    assert "Full card" not in demographic.user
    assert "short published summary" not in demographic.user.casefold()

    full = build_prompt("full_persona", case)
    assert "Full card about this respondent." in full.user
    assert "Age: 40" not in full.user

    summary = build_prompt("persona_summary", case)
    assert "A short published summary." in summary.user
    assert "Full card" not in summary.user


def test_atus_expanded_is_deterministic_and_keeps_the_same_person():
    case = _twin()
    first = build_prompt("atus_expanded", case, seed=3)
    second = build_prompt("atus_expanded", case, seed=3)
    other = build_prompt("atus_expanded", case, seed=4)
    assert first.user == second.user
    assert first.user != other.user
    assert "Age: 40" in first.user
    assert "work:" in first.user
    assert "SECRET_GOLD" not in first.user


def test_missing_persona_fields_name_the_dataset_to_fetch():
    bundle = twin_bundle(
        [
            {
                "id": "9",
                "questions": [{"id": "Q", "prompt": "Hello", "kind": "text"}],
            }
        ]
    )
    case = bundle.cases[0]
    with pytest.raises(PersonaMethodError, match="persona_text"):
        build_prompt("full_persona", case)
    with pytest.raises(PersonaMethodError, match="persona_summary"):
        build_prompt("persona_summary", case)
    with pytest.raises(PersonaMethodError, match="demographic"):
        build_prompt("demographics_only", case)
    with pytest.raises(PersonaMethodError, match="does not invent"):
        build_prompt("atus_expanded", case)


def test_group_prompt_asks_for_a_distribution_and_refuses_people():
    package = build_prompt("group_prompt", _group())
    assert package.response_mode == "distribution"
    assert "Do not invent an individual respondent" in package.system
    assert "ESS respondent" in package.user
    assert "SECRET" not in package.user
    with pytest.raises(PersonaMethodError):
        build_prompt("group_prompt", _twin())
