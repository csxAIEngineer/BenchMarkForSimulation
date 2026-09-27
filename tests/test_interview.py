import pytest

from popbench.dao.interview import SCALE_ZH, keyed_score, scale_means
from popbench.dao.personas import allocate_quotas
from popbench.dao.schema import InterviewRecord
from popbench.evaluate.interview import score_simulation
from popbench.evaluate.metrics import (
    closeness_score,
    cronbach_alpha,
    jensen_shannon,
    kolmogorov_smirnov,
    opinion_alignment,
    ordinal_accuracy,
    pearson,
    quadratic_weighted_kappa,
    total_variation_distance,
    wasserstein_1d,
)
from popbench.simulate.client import parse_choice
from popbench.simulate.interview import turns_user_message


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
    assert wasserstein_1d(shares, shares) == 0
    assert kolmogorov_smirnov(shares, shares) == 0
    assert opinion_alignment(shares, shares) == 1


def test_closeness_score_is_100_for_humans_and_0_for_random_answers():
    assert closeness_score(0, perfect=0, random_floor=0.4) == 100
    assert closeness_score(0.4, perfect=0, random_floor=0.4) == 0
    assert closeness_score(1, perfect=1, random_floor=0.2) == 100
    assert closeness_score(0.2, perfect=1, random_floor=0.2) == 0
    assert closeness_score(0.6, perfect=1, random_floor=0.2) == pytest.approx(50)


def test_ordinal_metrics_treat_a_far_miss_as_worse_than_a_near_miss():
    far = [1.0, 0.0, 0.0, 0.0, 0.0]
    near = [0.0, 1.0, 0.0, 0.0, 0.0]
    end = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert wasserstein_1d(near, far) == pytest.approx(1)
    assert wasserstein_1d(end, far) == pytest.approx(4)
    assert opinion_alignment(end, far) == pytest.approx(0)
    assert kolmogorov_smirnov(end, far) == pytest.approx(1)
    assert ordinal_accuracy([5], [1], span=4) == 0
    assert ordinal_accuracy([5], [4], span=4) == pytest.approx(0.75)
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], categories=5) == pytest.approx(1)
    alpha = cronbach_alpha([[1, 1], [2, 2], [3, 3], [4, 4]])
    assert alpha == pytest.approx(1)


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


def test_turn_prompt_does_not_carry_the_gold_answer():
    message = turns_user_message(
        {
            "turn": 3,
            "prompt": "I see myself as someone who is talkative.",
            "options": ["Disagree strongly", "Agree strongly"],
            "gold_answer": "SECRET GOLD",
            "gold_code": 5,
        }
    )
    assert "Question 3 of 50." in message
    assert "SECRET GOLD" not in message
    assert "gold" not in message.casefold()


def test_interview_record_round_trip():
    raw = {
        "id": "twin2k:p1",
        "region": "US",
        "language": "en",
        "source": "LLM-Digital-Twin/Twin-2K-500",
        "instrument": "bfi44_plus_green6",
        "persona": {"id": "p1", "condition": "demographics_only", "text": "Age: 40"},
        "turns": [
            {
                "turn": 1,
                "item_id": "QID25_1",
                "scale": "extraversion",
                "prompt": "I see myself as someone who is talkative.",
                "options": ["Disagree strongly", "Agree strongly"],
                "gold_answer": "Agree strongly",
                "gold_code": 2,
                "label_text": None,
                "reverse": False,
            }
        ],
        "scale_scores": {"extraversion": 2.0},
    }
    assert InterviewRecord.from_dict(raw).to_dict() == raw
