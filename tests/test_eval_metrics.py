"""Tests for behavioral-fit metrics and cross-model robustness."""

from pathlib import Path

import pytest

from popbench.evaluate.metrics import (
    cross_model_variance,
    distribution_correlation,
    mean_squared_error,
    response_amplitude_coverage,
)
from popbench.evaluate.robustness import RobustnessError, audit_cross_model_variance
from popbench.evaluate.visitors import behavioral_fit


def test_distribution_correlation_perfect_match():
    shares = [0.1, 0.2, 0.4, 0.2, 0.1]
    assert distribution_correlation(shares, shares) == pytest.approx(1.0)


def test_mse_zero_on_identical_vectors():
    assert mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mean_squared_error([0.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_response_amplitude_coverage_full_and_partial():
    human = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert response_amplitude_coverage([1.0, 2.0, 3.0, 4.0, 5.0], human) == pytest.approx(1.0)
    assert response_amplitude_coverage([2.0, 3.0], human) == pytest.approx(0.4)
    assert response_amplitude_coverage([3.0, 3.0], [3.0, 3.0]) == pytest.approx(1.0)


def test_cross_model_variance_zero_when_identical():
    series = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    stats = cross_model_variance(series)
    assert stats["mean_variance"] == pytest.approx(0.0)
    assert stats["mean_std"] == pytest.approx(0.0)


def test_cross_model_variance_positive_when_models_differ():
    series = [[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]]
    stats = cross_model_variance(series)
    assert stats["mean_variance"] > 0
    assert stats["max_std"] > stats["mean_std"] - 1e-9


def test_behavioral_fit_on_toy_shares():
    from popbench.simulate.visitors import VISIT_TURNS

    baseline_items = []
    sim_items = []
    responses = [{"answers": []}]
    for turn in VISIT_TURNS:
        human_shares = [0.2, 0.2, 0.2, 0.2, 0.2]
        sim_shares = [0.2, 0.2, 0.2, 0.2, 0.2]
        codes = [1, 2, 3, 4, 5]
        baseline_items.append(
            {
                "item_id": turn.item_id,
                "turn": turn.turn,
                "shares": human_shares,
                "codes": codes,
                "mean_code": 3.0,
            }
        )
        sim_items.append(
            {
                "item_id": turn.item_id,
                "turn": turn.turn,
                "shares": sim_shares,
                "n": 5,
            }
        )
        responses[0]["answers"].append(
            {"item_id": turn.item_id, "code": 3, "ok": True}
        )
    fit = behavioral_fit(
        {"items": sim_items},
        responses,
        {"items": baseline_items, "n_respondents": 5},
    )
    assert fit["share_mse"] == pytest.approx(0.0)
    assert fit["response_amplitude_coverage"] is not None


def test_audit_cross_model_requires_two_runs(tmp_path):
    with pytest.raises(RobustnessError):
        audit_cross_model_variance([tmp_path / "only"])


def test_audit_cross_model_variance_from_option_shares(tmp_path):
    from popbench.simulate.visitors import VISIT_OPTIONS, VISIT_TURNS
    import json

    def _write_run(path: Path, model: str, mean_shift: float) -> None:
        path.mkdir()
        items = []
        for turn in VISIT_TURNS:
            # Point mass that shifts with the model.
            code = 1 + int(mean_shift) % 5
            shares = [0.0] * len(VISIT_OPTIONS)
            shares[code - 1] = 1.0
            items.append(
                {
                    "item_id": turn.item_id,
                    "turn": turn.turn,
                    "shares": shares,
                    "options": list(VISIT_OPTIONS),
                    "n": 1,
                }
            )
        (path / "option_shares.json").write_text(
            json.dumps({"items": items}, indent=2) + "\n"
        )
        (path / "meta.json").write_text(json.dumps({"model": model, "panel": "visitors"}) + "\n")

    run_a = tmp_path / "model-a"
    run_b = tmp_path / "model-b"
    _write_run(run_a, "model-a", 0)
    _write_run(run_b, "model-b", 2)
    out = tmp_path / "robustness"
    summary = audit_cross_model_variance([run_a, run_b], out_dir=out)
    assert summary["n_runs"] == 2
    assert summary["mean_code_variance"]["mean_variance"] > 0
    assert (out / "robustness_summary.json").is_file()
    assert (out / "report.md").is_file()
