"""Card anchors keep cross-person order. An id hash must not redraw it."""

from popbench.datasets.twin2k_500.loader import bundle_from_records
from popbench.models import build_prompt
from popbench.models.atus_expanded_skill.tendency import persona_tendency, tendency_card

_AXIS_BANS = ("价格敏感：", "问卷认真度", "冲动：", "死抠", "五五开", "想试试", "排斥", "偏自由", "未锁定轴")


def test_same_fields_are_deterministic_and_ignore_person_id():
    demo = {"age": "40", "education": "high_school", "income": "$100,000 or more"}
    assert persona_tendency(demo) == persona_tendency(dict(demo))
    assert tendency_card("7", demo, seed=3) == tendency_card("7", demo, seed=3)
    assert tendency_card("7", demo, seed=3) == tendency_card("8", demo, seed=9)


def test_stated_price_stays_verbatim_and_is_not_a_three_level_lock():
    cautious = {
        "income": "$100,000 or more",
        "price_importance": "Extremely important",
    }
    relaxed = {"income": "$100,000 or more", "price_importance": "Not important"}
    cautious_card = tendency_card("1", cautious, seed=0)
    relaxed_card = tendency_card("9", relaxed, seed=4)
    assert ("对价格的在意", "Extremely important") in persona_tendency(cautious).clear
    assert ("对价格的在意", "Not important") in persona_tendency(relaxed).clear
    assert "Extremely important" in _clear(cautious_card)
    assert "Not important" in _clear(relaxed_card)
    assert "$100,000 or more" in _soft(cautious_card)
    assert "$100,000 or more" not in _clear(cautious_card)
    assert tendency_card("1", cautious, seed=0) == tendency_card("999", cautious, seed=7)
    for card in (cautious_card, relaxed_card):
        _assert_no_axis_lottery(card)


def test_income_education_and_age_are_soft_cues_not_behavior_bins():
    rich = tendency_card(
        "7",
        {"income": "$100,000 or more", "age": "40", "education": "high_school"},
        seed=1,
    )
    poor = tendency_card(
        "8",
        {"income": "Less than $30,000", "age": "65+", "education": "Completed high school"},
        seed=1,
    )
    assert "$100,000 or more" in _soft(rich)
    assert "Less than $30,000" in _soft(poor)
    assert "65+" in _soft(poor)
    assert "Completed high school" in _soft(poor)
    assert "残差：关闭" in rich
    assert "0.04" not in rich
    _assert_no_axis_lottery(rich)
    _assert_no_axis_lottery(poor)


def test_income_flag_without_a_dollar_string_stays_soft():
    under = persona_tendency({"income_over_50k": "false"})
    over = persona_tendency({"income_over_50k": "true"})
    assert ("收入", "不超过 5 万美元") in under.soft
    assert ("收入", "超过 5 万美元") in over.soft
    both = persona_tendency({"income": "$100,000 or more", "income_over_50k": "false"})
    assert ("收入", "$100,000 or more") in both.soft
    assert all("5 万" not in value for _, value in both.soft)


def test_qid_income_and_education_are_not_price_or_novelty_locks():
    tendency = persona_tendency(
        {"QID21": "$100,000 or more", "QID4": "College graduate/some postgrad"}
    )
    assert ("收入", "$100,000 or more") in tendency.soft
    assert ("学历", "College graduate/some postgrad") in tendency.soft
    assert tendency.clear == ()
    card = tendency_card("7", {"QID21": "$100,000 or more"}, seed=0)
    _assert_no_axis_lottery(card)


def test_politics_and_novelty_are_verbatim_clear_cues():
    tendency = persona_tendency(
        {
            "political": "Very liberal",
            "age": "32",
            "try_novel_food_technology": "Not likely",
        }
    )
    assert ("政治倾向", "Very liberal") in tendency.clear
    assert ("尝新意愿", "Not likely") in tendency.clear
    assert ("年龄", "32") in tendency.soft
    card = tendency_card("7", {"political": "Very liberal", "age": "32"}, seed=0)
    assert "Very liberal" in _clear(card)
    assert "32" in _soft(card)
    _assert_no_axis_lottery(card)


def test_bare_numbers_stay_prior_answers_unless_the_field_is_age():
    tendency = persona_tendency({"age": "40", "QID5": "45"})
    assert ("年龄", "40") in tendency.soft
    assert ("既往回答 QID5", "45") in tendency.clear
    counted = "How many years have you shopped here?\nAnswer: 20\n"
    echoed = persona_tendency({}, demographic_text=counted)
    assert ("年龄", "20") not in echoed.soft
    assert any(value == "20" for _, value in echoed.clear)
    asked = "What is your age?\nAnswer: 40\n"
    assert ("年龄", "40") in persona_tendency({}, demographic_text=asked).soft


def test_first_stated_politics_wins_when_text_disagrees():
    text = "Political leaning: Very conservative"
    tendency = persona_tendency({"political": "Very liberal"}, demographic_text=text)
    assert [value for label, value in tendency.clear if label == "政治倾向"] == ["Very liberal"]


def test_identity_answers_do_not_become_clear_rank_anchors():
    tendency = persona_tendency({"QID2": "Female", "QID8": "Married", "QID9": "Often"})
    values = [value for _, value in (*tendency.clear, *tendency.soft)]
    assert "Female" not in values
    assert "Married" not in values
    assert ("既往回答 QID9", "Often") in tendency.clear
    text = "Pick a group\nAnswer: Female\n"
    echoed = persona_tendency({}, demographic_text=text)
    assert all(value != "Female" for _, value in echoed.clear)


def test_structural_fields_are_not_promoted_over_a_stated_habit():
    tendency = persona_tendency(
        {"primary_grocery_shopper": "Yes", "household_size": "3", "sex": "Female"}
    )
    assert ("谁买菜", "Yes") in tendency.clear
    assert all(value != "3" for _, value in (*tendency.clear, *tendency.soft))
    assert all("Female" not in value for _, value in (*tendency.clear, *tendency.soft))


def test_empty_card_keeps_only_a_tiny_residual():
    cards = [tendency_card(f"p{index}", {}, seed=index) for index in range(12)]
    assert len(set(cards)) > 1
    for card in cards:
        assert "明确线索：无" in card
        assert "软线索：无" in card
        assert "0.04" in card
        assert card.endswith("不要答成平均受访者，也不要每题重新发明一套价值观。")
        _assert_no_axis_lottery(card)
    anchored = {"age": "40"}
    assert len({tendency_card(str(index), anchored, seed=index) for index in range(6)}) == 1


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
    tendency = persona_tendency({}, demographic_text=text)
    assert ("年龄", "65+") in tendency.soft
    assert "18-29" not in tendency_card("7", {}, demographic_text=text, seed=0)


def test_demographic_text_fills_income_and_prior_answers():
    income = (
        "Last year, what was your total family income from all sources, before taxes?\n"
        "Answer: $100,000 or more"
    )
    assert ("收入", "$100,000 or more") in persona_tendency({}, demographic_text=income).soft
    prior = "How do you feel about trying new snacks?\nOptions:\n- Never\n- Often\nAnswer: Often\n"
    echoed = persona_tendency({}, demographic_text=prior)
    assert any(value == "Often" for _, value in echoed.clear)
    assert all(value != "Never" for _, value in echoed.clear)
    bundle = bundle_from_records(
        [
            {
                "id": "7",
                "demographics": {"age": "40"},
                "demographic_text": income,
                "questions": [
                    {
                        "id": "Q1",
                        "prompt": "Pick one",
                        "options": ["red", "blue"],
                        "kind": "multiple_choice",
                    }
                ],
            }
        ]
    )
    package = build_prompt("atus_expanded_skill", bundle.cases[0], seed=0)
    assert "$100,000 or more" in package.user
    assert "价格敏感：低" not in package.user
    assert "平均受访者" in package.system
    assert package.outer_sample is True


def test_prior_answers_are_capped_and_keep_card_order():
    blocks = [
        f"Hobby item {index}?\nOptions:\n- yes\n- no\nAnswer: choice-{index}" for index in range(12)
    ]
    tendency = persona_tendency({}, demographic_text="\n".join(blocks))
    assert len(tendency.clear) == 8
    assert [value for _, value in tendency.clear] == [f"choice-{index}" for index in range(8)]


def _clear(card: str) -> str:
    return _between(card, "明确线索", "软线索")


def _soft(card: str) -> str:
    return _between(card, "软线索", "残差")


def _between(card: str, start: str, end: str) -> str:
    head = card.split(start, 1)[1]
    return head.split(end, 1)[0]


def _assert_no_axis_lottery(card: str) -> None:
    for banned in _AXIS_BANS:
        assert banned not in card
