"""Question batching: one persona, one provider call, one result row per question."""

from popbench.comms.batch import pack_batch
from popbench.comms.client import LLMClient
from popbench.comms.providers.mock import MockProvider
from popbench.datasets.twin2k_500.loader import bundle_from_records
from popbench.models import build_prompt
from popbench.pipeline import simulate


def _twin(n_questions: int = 4, n_people: int = 1):
    people = []
    for person in range(n_people):
        people.append(
            {
                "id": str(person + 1),
                "demographics": {"age": "34", "sex": "Female"},
                "demographic_text": "Age: 34\nSex: Female",
                "questions": [
                    {
                        "id": f"Q{index}",
                        "prompt": f"Question {index}",
                        "options": ["red", "blue"],
                        "kind": "multiple_choice",
                        "gold": "SECRET_GOLD",
                    }
                    for index in range(n_questions)
                ],
            }
        )
    return bundle_from_records(people)


def test_batch_puts_the_persona_once_and_hides_gold():
    packages = [build_prompt("atus_expanded", case, seed=0) for case in _twin(3).cases]
    _system, user, question_ids = pack_batch(packages)
    assert user.count("Persona:") == 1
    assert "Age: 34" in user
    assert question_ids == ("Q0", "Q1", "Q2")
    assert user.count("question_id:") == 3
    assert "SECRET_GOLD" not in user
    assert '"answers"' in _system


def test_mock_batch_uses_fewer_calls_than_questions():
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", temperature=0)
    results = simulate(
        _twin(4),
        method="atus_expanded",
        client=client,
        batch_size=2,
    )
    assert provider.calls == 2
    assert [row.question_id for row in results] == ["Q0", "Q1", "Q2", "Q3"]
    assert all(row.parsed == {"answer": "mock", "rationale": "mock reply from mock-llm"} for row in results)
    assert all("SECRET_GOLD" not in row.raw for row in results)


def test_one_batch_when_batch_size_covers_the_subject():
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", temperature=0)
    results = simulate(_twin(4), method="atus_expanded", client=client, batch_size=20)
    assert provider.calls == 1
    assert len(results) == 4


def test_subjects_are_not_mixed_into_one_call():
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", temperature=0)
    simulate(_twin(2, n_people=2), method="atus_expanded", client=client, batch_size=10)
    assert provider.calls == 2


def test_distribution_batch_parses_each_question():
    """Distribution replies stay supported even without a group persona method."""
    from popbench.schema import PromptPackage

    persona = "You are an ESS respondent in Europe."
    packages = (
        PromptPackage(
            dataset="simbench",
            subject_id="ESS:population",
            question_id="trust",
            method="atus_expanded",
            system="",
            user="",
            response_mode="distribution",
            persona=persona,
            question_body="Trust most people?\n\nOptions:\n- A\n- B",
        ),
        PromptPackage(
            dataset="simbench",
            subject_id="ESS:population",
            question_id="fair",
            method="atus_expanded",
            system="",
            user="",
            response_mode="distribution",
            persona=persona,
            question_body="Are people fair?\n\nOptions:\n- Yes\n- No",
        ),
    )
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", temperature=None)
    results = client.complete_batch(packages)
    assert provider.calls == 1
    assert results[0].parsed["distribution"] == {"A": 0.5, "B": 0.5}
    assert results[1].parsed["distribution"] == {"Yes": 0.5, "No": 0.5}
    assert "0.4" not in results[0].raw


def test_batch_cache_is_one_entry_for_the_whole_request(tmp_path):
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", cache_dir=tmp_path, temperature=0)
    bundle = _twin(4)
    first = simulate(bundle, method="atus_expanded", client=client, batch_size=4)
    second = simulate(bundle, method="atus_expanded", client=client, batch_size=4)
    assert provider.calls == 1
    assert all(row.cached for row in second)
    assert [row.question_id for row in first] == [row.question_id for row in second]
    assert len(list(tmp_path.glob("*.json"))) == 1
