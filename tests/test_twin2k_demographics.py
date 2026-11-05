"""Twin-2K QID demographics must reach the atus_expanded persona card.

Pricing subjects store the wave-1 block as QID11–QID24. The card reads
age, sex, education, income, and the other canonical fields. Subject 1732
is the reported pricing example: Female, 30-49, some college, household
income under $30,000.
"""

from popbench.datasets.twin2k_500.loader import bundle_from_records
from popbench.models import build_prompt
from popbench.schema import Case, Question, Subject

# Fields the pricing cache actually has for pid 1732, plus the rest of the
# wave-1 block so the card is not half empty. Only QID12, QID13, QID14, and
# QID21 are taken from the reported subject; the others are fixture answers
# in the Twin-2K codebook.
SUBJECT_1732 = {
    "id": "1732",
    "demographics": {
        "QID11": "South",
        "QID12": "Female",
        "QID13": "30-49",
        "QID14": "Some college, no degree",
        "QID15": "White",
        "QID16": "Yes",
        "QID17": "Never been married",
        "QID18": "Agnostic",
        "QID19": "Seldom",
        "QID20": "Democrat",
        "QID21": "Less than $30,000",
        "QID22": "Liberal",
        "QID23": "2",
        "QID24": "Part-time employment",
    },
    "questions": [{"id": "QID9_1", "prompt": "Would you buy this?", "kind": "text"}],
}

HIGH_INCOME = {
    "id": "2001",
    "demographics": {
        "QID11": "Northeast",
        "QID12": "Male",
        "QID13": "65+",
        "QID14": "College graduate/some postgrad",
        "QID15": "Asian",
        "QID17": "Married",
        "QID21": "$100,000 or more",
        "QID24": "Full-time employment",
    },
    "questions": [{"id": "QID9_1", "prompt": "Would you buy this?", "kind": "text"}],
}


def _persona(record: dict[str, object]) -> str:
    case = bundle_from_records([record]).cases[0]
    package = build_prompt("atus_expanded", case, seed=0)
    return package.user.split("Question:")[0]


def _basics(persona: str) -> str:
    start = persona.index("## Basics")
    end = persona.index("## Life history")
    return persona[start:end]


def test_subject_1732_basics_use_qid_demographics():
    basics = _basics(_persona(SUBJECT_1732))
    assert "Age: 30-49" in basics
    assert "Sex: Female" in basics
    assert "Education: Some college, no degree" in basics
    assert "Marital status: Never been married" in basics
    assert "Race: White" in basics
    assert "Region: South" in basics
    assert "State: Unknown" not in basics
    assert "Household income: Less than $30,000" in basics
    assert "Personal income: $50,000 or less" in basics
    assert "Personal income: over $50,000" not in basics
    assert "Employment: Part-time employment" in basics
    assert "Usual weekly work hours: 20" in basics
    assert "Age: unknown" not in basics
    assert "Sex: Unknown" not in basics


def test_high_income_twin_is_not_forced_under_50k():
    basics = _basics(_persona(HIGH_INCOME))
    assert "Age: 65+" in basics
    assert "Sex: Male" in basics
    assert "Household income: $100,000 or more" in basics
    assert "Personal income: over $50,000" in basics
    assert "Usual weekly work hours: 40" in basics
    assert "Region: Northeast" in basics


def test_different_twin2k_people_do_not_share_basics():
    low = _basics(_persona(SUBJECT_1732))
    high = _basics(_persona(HIGH_INCOME))
    assert low != high


def test_income_brackets_around_50k_follow_the_lower_bound():
    under = _basics(
        _persona(
            {
                "id": "30",
                "demographics": {"QID12": "Female", "QID21": "$30,000-$50,000"},
                "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
            }
        )
    )
    over = _basics(
        _persona(
            {
                "id": "50",
                "demographics": {"QID12": "Female", "QID21": "$50,000-$75,000"},
                "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
            }
        )
    )
    assert "Household income: $30,000-$50,000" in under
    assert "Personal income: $50,000 or less" in under
    assert "Household income: $50,000-$75,000" in over
    assert "Personal income: over $50,000" in over


def test_card_maps_raw_qids_without_the_loader():
    """Pricing code can hand the jsonl demographics dict straight to the card."""
    case = Case(
        dataset="twin2k_500",
        subject=Subject(
            id="1732",
            dataset="twin2k_500",
            unit="individual",
            demographics=dict(SUBJECT_1732["demographics"]),
        ),
        question=Question(id="QID9_1", prompt="Would you buy this?", kind="text"),
    )
    persona = build_prompt("atus_expanded", case, seed=0).user.split("Question:")[0]
    basics = _basics(persona)
    assert "Sex: Female" in basics
    assert "Age: 30-49" in basics
    assert "Household income: Less than $30,000" in basics
    assert "Usual weekly work hours: 20" in basics


def test_loader_keeps_qids_and_adds_canonical_keys():
    subject = bundle_from_records([SUBJECT_1732]).subjects[0]
    assert subject.demographics["QID12"] == "Female"
    assert subject.demographics["QID21"] == "Less than $30,000"
    assert subject.demographics["sex"] == "Female"
    assert subject.demographics["age"] == "30-49"
    assert subject.demographics["education"] == "Some college, no degree"
    assert subject.demographics["income"] == "Less than $30,000"
    assert subject.demographics["region"] == "South"
    assert subject.demographics["employment"] == "Part-time employment"
    assert "work_hours" not in subject.demographics


def test_explicit_canonical_fields_win_over_qids():
    persona = _persona(
        {
            "id": "7",
            "demographics": {
                "age": "40",
                "sex": "Female",
                "state": "Ohio",
                "education": "high_school",
                "marital_status": "Married",
                "income_over_50k": "true",
                "work_hours": "40",
                "QID12": "Male",
                "QID13": "65+",
                "QID21": "Less than $30,000",
                "QID24": "Part-time employment",
            },
            "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
        }
    )
    basics = _basics(persona)
    assert "Age: 40" in basics
    assert "Sex: Female" in basics
    assert "State: Ohio" in basics
    assert "Personal income: over $50,000" in basics
    assert "Usual weekly work hours: 40" in basics
    assert "Sex: Male" not in basics
    assert "Less than $30,000" not in basics


def test_life_history_follows_mapped_education_and_marital_status():
    persona = _persona(SUBJECT_1732)
    assert "Never been married" in persona
    assert "Lives independently" in persona or "Has stayed single" in persona
    assert "Family life is private" not in persona
    assert "community college" in persona or "never completed a bachelor" in persona

    graduate = _persona(HIGH_INCOME)
    assert "community college" not in graduate
    assert "never completed a bachelor" not in graduate
    assert "bachelor" in graduate or "Finished college" in graduate


def test_associate_degree_and_partner_status_pick_real_templates():
    associate = _persona(
        {
            "id": "assoc",
            "demographics": {"QID14": "Associate’s degree", "QID17": "Living with a partner"},
            "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
        }
    )
    assert "Education: Associate’s degree" in associate
    assert "Marital status: Living with a partner" in associate
    assert "Education path is uneven" not in associate
    assert "Family life is private" not in associate
    assert "community college" in associate or "never completed a bachelor" in associate


def test_employment_hours_are_mapped_only_when_the_label_says():
    retired = _basics(
        _persona(
            {
                "id": "r",
                "demographics": {"QID24": "Retired"},
                "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
            }
        )
    )
    freelance = _basics(
        _persona(
            {
                "id": "s",
                "demographics": {"QID24": "Self-employed"},
                "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
            }
        )
    )
    assert "Employment: Retired" in retired
    assert "Usual weekly work hours: 0" in retired
    assert "Employment: Self-employed" in freelance
    assert "Usual weekly work hours: unknown" in freelance
    student = _basics(
        _persona(
            {
                "id": "stu",
                "demographics": {"QID24": "Full-time student"},
                "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
            }
        )
    )
    assert "Employment: Full-time student" in student
    assert "Usual weekly work hours: 0" in student


def test_atus_row_without_qids_still_prints_state_unknown():
    basics = _basics(
        _persona(
            {
                "id": "atus",
                "demographics": {"age": "34", "sex": "Female"},
                "questions": [{"id": "Q", "prompt": "Buy?", "kind": "text"}],
            }
        )
    )
    assert "Age: 34" in basics
    assert "Sex: Female" in basics
    assert "State: Unknown" in basics
    assert "Personal income: $50,000 or less" in basics
