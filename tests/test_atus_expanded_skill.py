"""Survey-twin skill on top of the shared atus_expanded card."""

from popbench.comms.client import LLMClient
from popbench.comms.providers.mock import MockProvider
from popbench.comms.sample import apply_outer_sample, clamp_prob, sample_rng
from popbench.datasets.twin2k_500.loader import bundle_from_records
from popbench.models import build_prompt
from popbench.models.atus_expanded_skill.tendency import tendency_card
from popbench.pipeline import simulate
from popbench.schema import PromptPackage


def _case(
    person: str = "7",
    question: str = "Q1",
    kind: str = "multiple_choice",
    options=("red", "blue"),
    demographics: dict | None = None,
):
    demo = {"age": "40", "sex": "Female", "education": "high_school"}
    if demographics:
        demo.update(demographics)
    bundle = bundle_from_records(
        [
            {
                "id": person,
                "demographics": demo,
                "questions": [
                    {
                        "id": question,
                        "prompt": "Pick one",
                        "options": list(options),
                        "kind": kind,
                        "gold": "SECRET_GOLD",
                    }
                ],
            }
        ]
    )
    return bundle.cases[0]


def test_skill_prompt_reuses_the_card_and_embeds_the_skill():
    demo = {"price_importance": "Extremely important"}
    first = build_prompt("atus_expanded_skill", _case("7", "Q1", demographics=demo), seed=0)
    second = build_prompt("atus_expanded_skill", _case("7", "Q2", demographics=demo), seed=0)
    other = build_prompt("atus_expanded_skill", _case("8", "Q1", demographics=demo), seed=0)
    plain = build_prompt("atus_expanded", _case("7", "Q1", demographics=demo), seed=0)
    card_demo = {"age": "40", "sex": "Female", "education": "high_school", **demo}
    anchor = tendency_card("7", card_demo, seed=0)
    assert "## Basics" in first.user
    assert "## Life history" in first.user
    assert "类人问卷孪生" in first.user
    assert "buy_prob" in first.user
    assert "probs" in first.user
    assert "谁高谁低" in first.user
    assert "公不公平" in first.user
    assert "没写宠物" in first.user
    assert "明确线索" in first.user
    assert "随机倾向" in first.user
    assert "个人倾向" in first.user
    assert "市场调查" in first.system
    assert "便宜就买" in first.system
    assert "平均受访者" in first.system
    assert "抽到的个人倾向" not in first.system
    assert "抽到的个人倾向" not in first.user
    assert "SECRET_GOLD" not in first.user
    assert "Extremely important" in first.user
    assert "价格敏感：" not in first.user
    assert anchor == tendency_card("8", card_demo, seed=0)
    assert anchor in first.user
    assert anchor in second.user
    assert anchor in other.user
    assert first.user != other.user
    assert "buy_prob" not in plain.user
    assert first.outer_sample is True
    assert plain.outer_sample is False


def test_outer_sampling_is_deterministic_and_clamped():
    assert clamp_prob(0.01) == 0.15
    assert clamp_prob(0.99) == 0.85
    package = PromptPackage(
        dataset="twin2k_500",
        subject_id="7",
        question_id="price-1",
        method="atus_expanded_skill",
        system="",
        user="",
        response_mode="answer",
        options=("buy", "skip"),
        kind="multiple_choice",
        outer_sample=True,
    )
    first, error = apply_outer_sample({"buy_prob": 0.99}, package, seed=3)
    second, _ = apply_outer_sample({"buy_prob": 0.99}, package, seed=3)
    assert error is None
    assert first == second
    assert first["buy_prob"] == 0.85
    assert first["answer"] in {"buy", "skip"}
    choice = PromptPackage(
        dataset="twin2k_500",
        subject_id="7",
        question_id="Q1",
        method="atus_expanded_skill",
        system="",
        user="",
        response_mode="answer",
        options=("red", "blue"),
        kind="multiple_choice",
        outer_sample=True,
    )
    picked, choice_error = apply_outer_sample({"probs": {"red": 0.2, "blue": 0.8}}, choice, seed=1)
    again, _ = apply_outer_sample({"probs": {"red": 0.2, "blue": 0.8}}, choice, seed=1)
    assert choice_error is None
    assert picked == again
    assert picked["answer"] in {"red", "blue"}
    assert sample_rng(1, "7", "choice", "Q1").random() == sample_rng(1, "7", "choice", "Q1").random()


def test_mock_batch_samples_pricing_choice_and_keeps_numeric(tmp_path):
    people = []
    for person, question, kind, options in (
        ("7", "price-1", "multiple_choice", ["buy", "skip"]),
        ("7", "color", "multiple_choice", ["red", "blue"]),
        ("7", "bid", "numeric", []),
    ):
        people.append(
            {
                "id": person,
                "demographics": {"age": "40", "sex": "Female"},
                "questions": [
                    {
                        "id": question,
                        "prompt": f"Item {question}",
                        "options": options,
                        "kind": kind,
                        "gold": "SECRET_GOLD",
                    }
                ],
            }
        )
    from popbench.datasets.twin2k_500.loader import bundle_from_records

    bundle = bundle_from_records(people)
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", cache_dir=tmp_path, temperature=0)
    results = simulate(bundle, method="atus_expanded_skill", client=client, batch_size=10, seed=4)
    assert provider.calls == 1
    by_id = {row.question_id: row for row in results}
    assert by_id["price-1"].parsed["answer"] in {"buy", "skip"}
    assert by_id["price-1"].parsed["buy_prob"] == 0.42
    assert by_id["color"].parsed["answer"] in {"red", "blue"}
    assert by_id["bid"].parsed["answer"] == "mock"
    assert all("SECRET_GOLD" not in row.raw for row in results)
    cached = simulate(bundle, method="atus_expanded_skill", client=client, batch_size=10, seed=4)
    assert provider.calls == 1
    assert [row.parsed["answer"] for row in cached] == [row.parsed["answer"] for row in results]
