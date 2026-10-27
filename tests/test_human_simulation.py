"""Human-like survey prompting helpers."""

from __future__ import annotations

import json

from popbench.simulate.human_simulation import (
    BUY_OPTION,
    SKIP_OPTION,
    clamp_buy_prob,
    human_system_message,
    is_pricing_purchase,
    parse_buy_probs,
    parse_soft_probs,
    purchase_answer,
    sample_purchase,
    soft_choice_answer,
    tendency_card,
    tendency_fields,
)
from popbench.simulate.survey import batch_message


def test_tendency_card_is_stable_per_person_and_differs_across_people():
    a = tendency_fields("person-a", 0)
    b = tendency_fields("person-b", 0)
    assert tendency_fields("person-a", 0) == a
    assert a != b
    assert "Personal leanings" in tendency_card("person-a", 0)


def test_human_system_message_embeds_persona_and_leanings():
    text = human_system_message("Age: 40\nSex: Female", person_id="p1", seed=0)
    assert "market survey" in text
    assert "Age: 40" in text
    assert "Personal leanings" in text


def test_buy_prob_clamp_and_bernoulli_are_deterministic():
    assert clamp_buy_prob(0.01) == 0.15
    assert clamp_buy_prob(0.99) == 0.85
    first = sample_purchase(0.5, person_id="p1", item_id="QID9_1", seed=0)
    second = sample_purchase(0.5, person_id="p1", item_id="QID9_1", seed=0)
    assert first is second


def test_parse_buy_probs_and_purchase_answer():
    turn = {
        "item_id": "QID9_1",
        "family": "pricing",
        "options": [BUY_OPTION, SKIP_OPTION],
    }
    raw = json.dumps({"gut": "whatever", "buy_prob": 0.02})
    row = parse_buy_probs(raw, [turn])[0]
    assert row["buy_prob"] == 0.15
    answer = purchase_answer(
        turn,
        buy_prob=row["buy_prob"],
        gut=row["gut"],
        person_id="p1",
        seed=0,
    )
    assert answer["ok"] is True
    assert answer["value"] in (1, 2)
    cached = json.loads(answer["model_raw"])
    assert cached["buy_prob"] == 0.15


def test_soft_choice_round_trip():
    turn = {
        "item_id": "QID287_1",
        "family": "experiments",
        "kind": "choice",
        "options": ["A", "B", "C", "D", "E"],
    }
    raw = json.dumps({"gut": "meh", "probs": [0.1, 0.2, 0.3, 0.25, 0.15]})
    row = parse_soft_probs(raw, [turn])[0]
    answer = soft_choice_answer(
        turn,
        probs=row["probs"],
        gut=row["gut"],
        person_id="p1",
        seed=0,
    )
    assert answer["value"] in range(1, 6)
    assert is_pricing_purchase(turn) is False


def test_pricing_batch_message_asks_for_buy_prob():
    turn = {
        "turn": 1,
        "item_id": "QID9_1",
        "kind": "choice",
        "family": "pricing",
        "instrument": "pricing",
        "prompt": "Would you purchase butter at $7.39?",
        "options": [BUY_OPTION, SKIP_OPTION],
    }
    text = batch_message([turn], n_items=1, human_simulation=True)
    assert "buy_prob" in text
    assert "Yes, I would purchase the product" in text
