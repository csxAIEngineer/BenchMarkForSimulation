"""Persona-conditioned tendency: same fields lock the same axes."""

from popbench.datasets.twin2k_500.loader import bundle_from_records
from popbench.models import build_prompt
from popbench.models.atus_expanded_skill.tendency import persona_tendency, tendency_card


def _levels(demographics: dict[str, str], **kwargs: str) -> dict[str, str]:
    return persona_tendency(demographics, **kwargs).levels()


def test_same_fields_are_deterministic_and_ignore_person_id():
    demo = {"age": "40", "education": "high_school", "income": "$100,000 or more"}
    assert persona_tendency(demo).levels() == persona_tendency(dict(demo)).levels()
    assert tendency_card("7", demo, seed=3) == tendency_card("7", demo, seed=3)
    assert tendency_card("7", demo, seed=3) == tendency_card("8", demo, seed=9)


def test_income_sets_price_sensitivity_and_is_not_nudged_away():
    rich = {"income": "$100,000 or more", "age": "40", "education": "high_school"}
    poor = {"income": "Less than $30,000", "age": "40", "education": "high_school"}
    assert _levels(rich)["价格敏感"] == "低"
    assert _levels(poor)["价格敏感"] == "高"
    assert _levels(rich)["对看起来贵的日用品"] == "不在乎"
    assert _levels(poor)["对看起来贵的日用品"] == "死抠"
    for person in ("7", "8", "9"):
        assert "价格敏感：低" in tendency_card(person, rich, seed=1)
        assert "价格敏感：高" in tendency_card(person, poor, seed=1)
        assert "$100,000 or more" in tendency_card(person, rich, seed=1)


def test_education_and_age_move_novelty_in_opposite_directions():
    high = _levels({"age": "40", "education": "Completed university"})
    low = _levels({"age": "40", "education": "Completed high school"})
    young = _levels({"age": "18-29", "education": "some college"})
    older = _levels({"age": "65+", "education": "some college"})
    assert high["对新奇零食 / 电子小件"] == "想试试"
    assert low["对新奇零食 / 电子小件"] == "排斥"
    assert young["对新奇零食 / 电子小件"] == "想试试"
    assert older["对新奇零食 / 电子小件"] == "排斥"


def test_income_flag_without_a_dollar_string():
    assert _levels({"income_over_50k": "false"})["价格敏感"] == "高"
    assert _levels({"income_over_50k": "true"})["价格敏感"] == "中"
    assert "收入是否超过5万美元：否" in tendency_card("1", {"income_over_50k": "false"}, seed=0)


def test_qid_income_and_stated_price_are_persona_fields():
    qid = _levels({"QID21": "$100,000 or more", "QID4": "College graduate/some postgrad"})
    assert qid["价格敏感"] == "低"
    assert "态度题" not in qid
    cautious = _levels({"income": "$100,000 or more", "price_importance": "Extremely important"})
    relaxed = _levels({"income": "$100,000 or more", "price_importance": "Not important"})
    assert cautious["价格敏感"] == "中"
    assert relaxed["价格敏感"] == "低"


def test_politics_is_an_anchor_and_empty_cards_stay_neutral():
    liberal = persona_tendency({"political": "Very liberal", "age": "32"})
    assert liberal.levels()["态度题"] == "偏自由"
    assert any(line.startswith("政治倾向：") for line in liberal.anchors)
    empty = _levels({})
    assert empty == {
        "价格敏感": "中",
        "对新奇零食 / 电子小件": "无所谓",
        "对看起来贵的日用品": "一般",
        "未知品类": "五五开",
    }
    blank = tendency_card("7", {}, seed=0)
    assert "没有收入、学历、年龄或写明的偏好" in blank
    shown = _shown(blank)
    diffs = [label for label, level in empty.items() if shown[label] != level]
    assert len(diffs) == 1


def test_nudge_touches_only_an_unlocked_axis():
    demo = {"age": "40"}
    locked = _levels(demo)
    cards = [tendency_card(str(person), demo, seed=2) for person in range(6)]
    for card in cards:
        shown = _shown(card)
        diffs = [label for label in locked if shown[label] != locked[label]]
        assert len(diffs) == 1
        assert diffs[0] in {"价格敏感", "对看起来贵的日用品"}
        assert shown["对新奇零食 / 电子小件"] == locked["对新奇零食 / 电子小件"]
        assert shown["未知品类"] == locked["未知品类"]
        assert "未锁定轴的一档抖动：" in card
    assert len(set(cards)) > 1


def test_outcome_bids_and_gold_are_not_anchors():
    card = tendency_card(
        "7",
        {
            "income": "$100,000-$149,999 CAD",
            "local_field": "8.88",
            "conclusion_purchased_product": "FALSE",
        },
        seed=0,
    )
    assert "8.88" not in card
    assert "FALSE" not in card
    assert "$100,000-$149,999 CAD" in card


def test_option_bullets_do_not_override_the_selected_answer():
    text = "What is your age?\nOptions:\n- 18-29\n- 65+\nAnswer: 65+\n"
    assert _levels({}, demographic_text=text)["对新奇零食 / 电子小件"] == "排斥"


def test_demographic_text_fills_income_when_the_key_is_absent():
    text = "Last year, what was your total family income from all sources, before taxes?\nAnswer: $100,000 or more"
    levels = _levels({}, demographic_text=text)
    assert levels["价格敏感"] == "低"
    bundle = bundle_from_records(
        [
            {
                "id": "7",
                "demographics": {"age": "40"},
                "demographic_text": text,
                "questions": [{"id": "Q1", "prompt": "Pick one", "options": ["red", "blue"], "kind": "multiple_choice"}],
            }
        ]
    )
    package = build_prompt("atus_expanded_skill", bundle.cases[0], seed=0)
    assert "$100,000 or more" in package.user
    assert "价格敏感：低" in package.user
    assert package.outer_sample is True


def _shown(card: str) -> dict[str, str]:
    levels: dict[str, str] = {}
    reading = False
    for line in card.splitlines():
        if line == "作答用：":
            reading = True
            continue
        if reading and line.startswith("- "):
            label, value = line[2:].split("：", 1)
            levels[label] = value
            continue
        if reading:
            break
    return levels
