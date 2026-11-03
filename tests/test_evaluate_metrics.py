"""Mode agreement (Top-1 accuracy) on categorical answer shares."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from popbench.evaluate import distribution_top1_accuracy  # noqa: E402
from popbench.evaluate.metrics import total_variation_distance  # noqa: E402
from score_cr_skill_compare_metrics import mean_tvd_by_item  # noqa: E402


def test_binary_top1_is_which_side_is_higher() -> None:
    assert distribution_top1_accuracy([0.2, 0.8], [0.45, 0.55]) == 1.0
    assert distribution_top1_accuracy([0.7, 0.3], [0.4, 0.6]) == 0.0


def test_tie_matches_only_on_a_shared_mode() -> None:
    assert distribution_top1_accuracy([1, 1], [1, 2]) == 1.0
    assert distribution_top1_accuracy([1, 1], [2, 1]) == 1.0
    assert distribution_top1_accuracy([1, 1], [1, 1]) == 1.0
    assert distribution_top1_accuracy([2, 1, 2], [1, 3, 1]) == 0.0
    assert distribution_top1_accuracy([1, 1, 1], [0.2, 0.2, 0.6]) == 1.0


def test_exact_near_tie_is_not_a_mode_tie() -> None:
    assert distribution_top1_accuracy([0.5000001, 0.4999999], [0.4, 0.6]) == 0.0


def test_counts_normalize_like_tvd() -> None:
    assert distribution_top1_accuracy([3, 1], [6, 2]) == 1.0
    assert total_variation_distance([3, 1], [6, 2]) == 0.0


def test_rejects_bad_share_vectors() -> None:
    with pytest.raises(ValueError, match="same length"):
        distribution_top1_accuracy([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="empty"):
        distribution_top1_accuracy([], [])
    with pytest.raises(ValueError, match="negative"):
        distribution_top1_accuracy([1, -1], [1, 0])


def test_mean_top1_acc_averages_item_modes() -> None:
    pairs = {
        "price": [("buy", "buy"), ("buy", "not"), ("not", "buy")],
        "color": [("red", "blue"), ("red", "blue")],
    }
    result = mean_tvd_by_item(pairs)
    assert result["mean_top1_acc"] == 0.5
    assert result["n_items_top1"] == 2
    assert result["n_items_tvd"] == 2
    assert result["mean_tvd"] == pytest.approx(0.5)


def test_mean_top1_acc_is_none_without_items() -> None:
    result = mean_tvd_by_item({})
    assert result["mean_top1_acc"] is None
    assert result["mean_tvd"] is None
    assert result["n_items_top1"] == 0
    assert result["n_items_tvd"] == 0
