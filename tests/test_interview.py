import pytest

from popbench.interview import (
    SCALE_ZH,
    allocate_quotas,
    keyed_score,
    scale_means,
    score_simulation,
)
from popbench.metrics import jensen_shannon, total_variation_distance
from popbench.simulate import parse_choice


def test_reverse_keyed_score_flips_the_agreement_scale():
    assert keyed_score(5, reverse=False) == 5
    assert keyed_score(1, reverse=True) == 5
    assert keyed_score(2, reverse=True) == 4


def test_scale_mean_uses_keyed_items():
    means = scale_means({"extraversion": [5, 5], "openness": [1, 3]})
    assert means["extraversion"] == 5
    assert means["openness"] == 2


def test_quotas_follow_weights_and_sum_to_n():
    quotas = allocate_quotas({"a": 50, "b": 50}, 10)
    assert sum(quotas.values()) == 10
    assert quotas["a"] == 5
    assert quotas["b"] == 5


def test_identical_shares_have_zero_distance():
    shares = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert total_variation_distance(shares, shares) == 0
    assert jensen_shannon(shares, shares) == 0


def test_parse_choice_accepts_the_option_text_and_its_number():
    options = ("Disagree strongly", "Agree strongly")
    parsed = parse_choice('{"answer": "Agree strongly", "rationale": "Fits."}', options)
    assert parsed.answer == "Agree strongly"
    numbered = parse_choice('{"answer": "1"}', options)
    assert numbered.answer == "Disagree strongly"


def test_simulation_matches_human_shares_when_the_panel_copies_them():
    items = [
        {"item_id": "Q1", "turn": 1, "scale": "extraversion", "reverse": False},
        {"item_id": "Q2", "turn": 2, "scale": "extraversion", "reverse": True},
    ]
    baseline = {
        "items": [
            {"item_id": "Q1", "shares": [0.0, 0.0, 0.0, 0.0, 1.0], "mean_code": 5},
            {"item_id": "Q2", "shares": [1.0, 0.0, 0.0, 0.0, 0.0], "mean_code": 1},
        ],
        "scales": {
            scale: {"mean": 5.0 if scale == "extraversion" else 3.0, "std": 0.0, "n": 1}
            for scale in SCALE_ZH
        },
    }
    # Only the scales that score_simulation initializes from SCALE_ZH are reported.
    # Feed a response whose keyed mean is 5, matching the human extraversion mean.
    responses = [
        {
            "id": "p",
            "answers": [
                {"item_id": "Q1", "code": 5, "ok": True},
                {"item_id": "Q2", "code": 1, "ok": True},
            ],
        }
    ]
    summary = score_simulation(responses, baseline, items)
    assert summary["mean_tvd"] == pytest.approx(0)
    assert summary["scales"]["extraversion"]["absolute_error"] == pytest.approx(0)
    assert set(summary["scales"]) == set(SCALE_ZH)


def test_parse_choice_rejects_non_json():
    with pytest.raises(ValueError):
        parse_choice("I agree", ("Agree strongly",))
