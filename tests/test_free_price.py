"""Free-price demand summaries. No model calls and no raw Twin files."""

from __future__ import annotations

from decimal import Decimal

import pytest

from popbench.evaluate.demand import (
    adjacent_upward_jumps,
    build_offers,
    classify_free_endpoint,
    equal_product_curve,
    free_endpoint,
    free_to_first_positive,
)
from popbench.evaluate.options import YES_OPTION, NO_OPTION
from popbench.evaluate.stimulus import is_free_price, parse_stimulus, tier_index

YES = YES_OPTION
NO = NO_OPTION


def _text(product: str, price: str) -> str:
    return (
        "Please consider the following product category: Snacks. "
        "Suppose you are in a grocery store, and you see the following product in that category: "
        f"{product}. The product is priced at: ${price}. "
        "Would you or would you not purchase this product?"
    )


def test_zero_price_is_parsed_currency_and_missing_text_is_not_free() -> None:
    parsed = parse_stimulus(_text("Oats", "0.00"))
    assert parsed is not None
    assert parsed["price_usd"] == Decimal("0.00")
    assert is_free_price(parsed["price_usd"]) is True
    assert parse_stimulus(_text("Oats", "0")) is None
    assert parse_stimulus("priced at: $0.00") is None
    assert parse_stimulus(_text("Oats", "1.45"))["price_usd"] == Decimal("1.45")
    assert tier_index(Decimal("1.45"), Decimal("14.54")) == 1
    assert tier_index(Decimal("0.00"), Decimal("14.54")) == 0
    assert tier_index(Decimal("1.00"), Decimal("14.54")) is None


def _offers():
    outcomes = [
        {
            "subject_id": "2",
            "items": [
                {"answer": NO, "question_id": "QID9_1", "question_text": _text("Oats", "0.00")},
                {"answer": YES, "question_id": "QID9_2", "question_text": _text("Oats", "1.00")},
            ],
        },
        {
            "subject_id": "1",
            "items": [
                {"answer": YES, "question_id": "QID9_1", "question_text": _text("Oats", "0.00")},
                {"answer": NO, "question_id": "QID9_2", "question_text": _text("Oats", "1.00")},
            ],
        },
    ]
    responses = [
        {"answer": "No", "parse_error": None, "question_id": "QID9_1", "subject_id": "1"},
        {"answer": "Yes", "parse_error": None, "question_id": "QID9_2", "subject_id": "1"},
        {"answer": "Would not purchase", "parse_error": None, "question_id": "QID9_1", "subject_id": "2"},
        {"answer": "Would", "parse_error": None, "question_id": "QID9_2", "subject_id": "2"},
    ]
    stimuli = {}
    for row in outcomes:
        for item in row["items"]:
            stimuli[(row["subject_id"], item["question_id"])] = parse_stimulus(item["question_text"])
    return outcomes, responses, stimuli


def test_free_offers_pair_by_id_and_do_not_code_unmapped_as_no() -> None:
    outcomes, responses, stimuli = _offers()
    forward = build_offers(outcomes, responses, ["1", "2"], stimuli)
    backward = build_offers(list(reversed(outcomes)), list(reversed(responses)), ["2", "1"], stimuli)
    assert free_endpoint(forward) == free_endpoint(backward)
    summary = free_endpoint(forward)
    assert summary["assigned_offers"] == 2
    assert summary["both_mapped_offers"] == 2
    assert summary["paired_human_rate"] == 0.5
    assert summary["paired_model_rate"] == 0.0
    assert summary["paired_gap_percentage_points"] == -50
    assert summary["paired_accuracy"] == 0.5
    # Subject 2's positive-price reply is unmapped. It is unknown, not a rejection.
    positive = [offer for offer in forward if offer["price_usd"] == Decimal("1.00") and offer["subject_id"] == "2"]
    assert positive[0]["model_status"] == "unmapped"
    assert positive[0]["model_code"] is None


def test_duplicate_conflict_is_not_a_purchase() -> None:
    outcomes, responses, stimuli = _offers()
    responses = responses + [
        {"answer": "Yes", "parse_error": None, "question_id": "QID9_1", "subject_id": "2"},
    ]
    offers = build_offers(outcomes, responses, ["1", "2"], stimuli)
    summary = free_endpoint(offers)
    assert summary["model_status"]["conflict"] == 1
    assert summary["both_mapped_offers"] == 1
    assert summary["model"]["unknown"] == 1
    assert summary["model"]["rate_mapped"] == 0.0
    assert summary["model"]["rate_lower_unknown_as_no"] == 0.0
    assert summary["model"]["rate_upper_unknown_as_yes"] == 0.5


def test_upward_jump_and_absent_intermediate_tier() -> None:
    monotonic = [
        {"product": "A", "tier": 0, "purchase_rate": 0.8, "n_mapped": 5},
        {"product": "A", "tier": 1, "purchase_rate": 0.8, "n_mapped": 5},
        {"product": "A", "tier": 2, "purchase_rate": 0.2, "n_mapped": 5},
    ]
    assert adjacent_upward_jumps(monotonic) == []
    rising = [
        {"product": "A", "tier": 0, "purchase_rate": 0.2, "n_mapped": 4},
        {"product": "A", "tier": 1, "purchase_rate": 0.7, "n_mapped": 4},
    ]
    jumps = adjacent_upward_jumps(rising)
    assert len(jumps) == 1
    assert jumps[0]["increase"] == pytest.approx(0.5)
    gapped = [
        {"product": "A", "tier": 0, "purchase_rate": 0.2, "n_mapped": 4},
        {"product": "A", "tier": 2, "purchase_rate": 0.9, "n_mapped": 4},
    ]
    assert adjacent_upward_jumps(gapped) == []
    summary = free_to_first_positive(gapped)
    assert summary["rate_increase_to_first_positive"] == 0.7
    assert summary["max_positive_minus_free"] == 0.7


def test_equal_product_weight_is_not_offer_weighted_composition() -> None:
    rows = [
        {"product": "A", "tier": 0, "purchase_rate": 1.0, "n_mapped": 1},
        {"product": "B", "tier": 0, "purchase_rate": 0.0, "n_mapped": 9},
    ]
    curve = equal_product_curve(rows)
    assert curve[0]["purchase_rate"] == 0.5
    assert curve[0]["n_mapped"] == 10
    assert classify_free_endpoint(0.1, 0.02, 0.2) == "confirmed_in_this_sample"
    assert classify_free_endpoint(-0.1, -0.2, -0.01) == "not_confirmed"
    assert classify_free_endpoint(0.1, -0.05, 0.2) == "inconclusive"
    assert classify_free_endpoint(None, None, None) == "inconclusive"
