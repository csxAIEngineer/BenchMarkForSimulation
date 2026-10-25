"""Paired TVD and share-MSE differences, with fake answers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from popbench.dao.io import write_jsonl
from popbench.evaluate.metrics import mean_squared_error, total_variation_distance
from popbench.evaluate.paired_effect import (
    COMPARE_MODEL_IDS,
    DEMOGRAPHIC_ITEM_IDS,
    FIXED_N,
    FIXED_SEEDS,
    ChoiceItem,
    align_by_id,
    answer_changes,
    arm_distances,
    assemble_model_endpoint,
    batch_run_dir,
    check_batch_arm,
    choice_items,
    codes_by_person,
    cross_seed_sign_flip,
    load_arm,
    option_shares,
    paired_distances,
    permute_paired_arms,
    render_paired_report,
    require_fixed_seeds,
    require_same_human_shares,
    score_fixed_batch,
    sign_flip_mean,
    write_paired_report,
)


def test_mean_distance_is_the_unweighted_mean_of_item_shares():
    # Two options and four options, so a pooled MSE would not match the item average.
    items = [
        ChoiceItem("Q2", 2, (0.5, 0.5)),
        ChoiceItem("Q4", 4, (0.5, 0.5, 0.0, 0.0)),
    ]
    answers = [[1, 1], [1, 1]]
    shares_q2 = option_shares([1, 1], 2)
    shares_q4 = option_shares([1, 1], 4)
    assert shares_q2 == [1.0, 0.0]
    assert shares_q4 == [1.0, 0.0, 0.0, 0.0]
    tvd_q2 = total_variation_distance(shares_q2, [0.5, 0.5])
    tvd_q4 = total_variation_distance(shares_q4, [0.5, 0.5, 0.0, 0.0])
    mse_q2 = mean_squared_error(shares_q2, [0.5, 0.5])
    mse_q4 = mean_squared_error(shares_q4, [0.5, 0.5, 0.0, 0.0])
    assert tvd_q2 == pytest.approx(0.5)
    assert tvd_q4 == pytest.approx(0.5)
    assert mse_q2 == pytest.approx(0.25)
    assert mse_q4 == pytest.approx(0.125)
    scored = arm_distances(answers, items)
    assert scored["mean_tvd"] == pytest.approx((tvd_q2 + tvd_q4) / 2)
    assert scored["mean_share_mse"] == pytest.approx((mse_q2 + mse_q4) / 2)
    assert scored["mean_share_mse"] != pytest.approx((0.25 + 0.25 + 0.25 + 0.25) / 6)


def test_human_counts_are_normalized_before_both_distances():
    counts = ChoiceItem("Q", 2, (2, 2))
    proportions = ChoiceItem("Q", 2, (0.5, 0.5))
    answers = [[1], [2]]
    from_counts = arm_distances(answers, [counts])
    from_proportions = arm_distances(answers, [proportions])
    assert from_counts["mean_tvd"] == pytest.approx(from_proportions["mean_tvd"])
    assert from_counts["mean_share_mse"] == pytest.approx(0.0)
    assert from_proportions["mean_share_mse"] == pytest.approx(0.0)


def test_failed_answer_is_omitted_from_the_share():
    shares = option_shares([1, None], 2)
    assert shares == [1.0, 0.0]
    scored = arm_distances([[1], [None]], [ChoiceItem("Q", 2, (0.5, 0.5))])
    assert scored["items"][0]["n"] == 1
    assert scored["mean_tvd"] == pytest.approx(0.5)


def test_difference_is_treatment_minus_control():
    item = ChoiceItem("Q", 2, (1.0, 0.0))
    # Control is perfect. Treatment is the other option, so the difference is positive.
    paired = paired_distances([[1], [1]], [[2], [2]], [item])
    assert paired["mean_tvd_control"] == pytest.approx(0.0)
    assert paired["mean_tvd_treatment"] == pytest.approx(1.0)
    assert paired["mean_tvd_diff"] == pytest.approx(1.0)
    assert paired["mean_share_mse_diff"] == pytest.approx(1.0)
    assert paired["items"][0]["tvd_diff"] == pytest.approx(paired["mean_tvd_diff"])


def test_same_tvd_can_still_move_share_mse():
    # Absolute gaps (0.2, 0.2, 0) and (0.1, 0.1, 0.2) have the same TVD and different MSE.
    human = ChoiceItem("Q", 3, (0.4, 0.4, 0.2))
    control = [[1]] * 6 + [[2]] * 2 + [[3]] * 2
    treatment = [[1]] * 5 + [[2]] * 5
    paired = paired_distances(control, treatment, [human])
    assert paired["mean_tvd_diff"] == pytest.approx(0.0)
    assert paired["mean_share_mse_diff"] == pytest.approx((0.06 - 0.08) / 3)


def test_exact_permutation_p_for_a_complete_shift():
    # Three people, one 3-option item, human mass on option 1.
    # Control all answer 1 (TVD 0). Treatment all answer 2 (TVD 1, MSE 2/3).
    # Of the 8 swaps, only the identity and the full swap keep |diff| at the observed value.
    item = ChoiceItem("Q", 3, (1.0, 0.0, 0.0))
    result = permute_paired_arms([[1], [1], [1]], [[2], [2], [2]], [item])
    assert result["method"] == "exact"
    assert result["n_assignments"] == 8
    assert result["mean_tvd_diff"] == pytest.approx(1.0)
    assert result["mean_share_mse_diff"] == pytest.approx(2 / 3)
    assert result["p_mean_tvd"] == pytest.approx(2 / 8)
    assert result["p_mean_share_mse"] == pytest.approx(2 / 8)


def test_permutation_swaps_a_persons_whole_answer_row():
    # Person-level swaps never change |mean TVD diff|. Swapping one item inside
    # a person would. Both endpoints therefore stay at p = 1.
    items = [
        ChoiceItem("Q1", 2, (1.0, 0.0)),
        ChoiceItem("Q2", 2, (1.0, 0.0)),
    ]
    control = [[1, 2], [1, 1]]
    treatment = [[2, 1], [2, 2]]
    result = permute_paired_arms(control, treatment, items)
    assert result["mean_tvd_diff"] == pytest.approx(0.5)
    assert result["mean_share_mse_diff"] == pytest.approx(0.5)
    assert result["p_mean_tvd"] == pytest.approx(1.0)
    assert result["p_mean_share_mse"] == pytest.approx(1.0)


def test_identical_arms_have_zero_difference_and_p_one():
    item = ChoiceItem("Q", 2, (0.5, 0.5))
    result = permute_paired_arms([[1], [2]], [[1], [2]], [item])
    assert result["mean_tvd_diff"] == pytest.approx(0.0)
    assert result["mean_share_mse_diff"] == pytest.approx(0.0)
    assert result["p_mean_tvd"] == pytest.approx(1.0)
    assert result["p_mean_share_mse"] == pytest.approx(1.0)


def test_monte_carlo_uses_the_plus_one_formula_and_one_draw_for_both_endpoints():
    item = ChoiceItem("Q", 3, (1.0, 0.0, 0.0))
    control = [[1], [1], [1]]
    treatment = [[2], [2], [2]]
    # Identity is as extreme as the observed difference. Swapping only the first
    # person is not. One shared draw per replicate feeds both endpoints.
    rng = _RowsRng([[0.9, 0.9, 0.9], [0.0, 0.9, 0.9]])
    result = permute_paired_arms(control, treatment, [item], reps=2, rng=rng)
    assert rng.calls == 2
    assert result["method"] == "monte_carlo"
    assert result["reps"] == 2
    assert result["p_mean_tvd"] == pytest.approx(2 / 3)
    assert result["p_mean_share_mse"] == pytest.approx(2 / 3)


def test_large_arm_is_not_enumerated_without_a_draw():
    item = ChoiceItem("Q", 2, (1.0, 0.0))
    answers = [[1]] * 17
    with pytest.raises(ValueError, match="reps"):
        permute_paired_arms(answers, answers, [item])
    with pytest.raises(ValueError, match="rng"):
        permute_paired_arms(answers, answers, [item], reps=4)


def test_sign_flip_lists_mean_min_max_and_keeps_a_zero():
    report = sign_flip_mean([1.0, -3.0, 0.5])
    assert report["values"] == [1.0, -3.0, 0.5]
    assert report["mean"] == pytest.approx(-1.5 / 3)
    assert report["min"] == pytest.approx(-3.0)
    assert report["max"] == pytest.approx(1.0)
    zero = sign_flip_mean([1.0, 1.0, 0.0])
    assert zero["mean"] == pytest.approx(2 / 3)
    assert zero["n"] == 3
    assert zero["n_zero"] == 1
    assert zero["p"] == pytest.approx(0.5)


def test_eight_equal_seeds_need_a_unanimous_sign_to_fall_under_0_05():
    unanimous = sign_flip_mean([1.0] * 8)
    assert unanimous["p"] == pytest.approx(2 / 256)
    assert unanimous["p"] < 0.05
    split = sign_flip_mean([1.0] * 7 + [-1.0])
    assert split["p"] == pytest.approx(18 / 256)
    assert split["p"] > 0.05
    assert split["n_negative"] == 1


def test_a_near_zero_seed_can_still_move_the_mean():
    values = [10.0] * 7 + [-0.1]
    report = sign_flip_mean(values)
    assert report["n_negative"] == 1
    assert report["mean"] == pytest.approx(69.9 / 8)
    assert report["min"] == pytest.approx(-0.1)
    assert report["max"] == pytest.approx(10.0)
    assert report["p"] == pytest.approx(_brute_sign_flip(values))
    assert report["p"] == pytest.approx(4 / 256)
    assert report["p"] < 0.05


def test_cross_seed_sign_flip_uses_one_number_per_seed():
    seeds = [
        {"seed": 7, "n_people": 50, "mean_tvd_diff": 0.2, "mean_share_mse_diff": -0.01},
        {"seed": 11, "n_people": 50, "mean_tvd_diff": -0.1, "mean_share_mse_diff": 0.02},
    ]
    report = cross_seed_sign_flip(seeds)
    assert report["n_seeds"] == 2
    assert report["seeds"] == [7, 11]
    assert report["mean_tvd_diff"]["values"] == pytest.approx([0.2, -0.1])
    assert report["mean_share_mse_diff"]["values"] == pytest.approx([-0.01, 0.02])
    assert report["mean_tvd_diff"]["p"] == pytest.approx(sign_flip_mean([0.2, -0.1])["p"])
    assert "n_people" not in report["mean_tvd_diff"]


def test_align_by_id_follows_the_control_order():
    control = {"b": [1], "a": [2]}
    treatment = {"a": [2], "b": [1]}
    ids, left, right = align_by_id(control, treatment)
    assert ids == ["b", "a"]
    assert left == [[1], [2]]
    assert right == [[1], [2]]
    paired = paired_distances(left, right, [ChoiceItem("Q", 2, (0.5, 0.5))])
    assert paired["mean_tvd_diff"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="person ids"):
        align_by_id({"a": [1]}, {"b": [1]})


def test_codes_follow_the_item_list_and_drop_failed_answers():
    responses = [
        {
            "id": "p2",
            "answers": [
                {"item_id": "age", "value": 4, "ok": True},
                {"item_id": "sex", "value": None, "ok": False},
            ],
        },
        {
            "id": "p1",
            "answers": [
                {"item_id": "sex", "value": 1, "ok": True},
                {"item_id": "age", "value": 2, "ok": True},
            ],
        },
    ]
    coded = codes_by_person(responses, ["sex", "age"])
    assert coded == {"p2": [None, 4], "p1": [1, 2]}
    with pytest.raises(ValueError, match="duplicate"):
        codes_by_person(responses + [responses[0]], ["sex"])
    with pytest.raises(ValueError, match="missing"):
        codes_by_person(responses, ["sex", "income"])


def test_load_arm_reads_saved_responses(tmp_path):
    path = tmp_path / "responses.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "atus-1",
                "answers": [{"item_id": "QID12", "value": 2, "ok": True}],
            }
        ],
    )
    assert load_arm(path, ["QID12"]) == {"atus-1": [2]}


def test_fixed_seeds_reject_any_other_list():
    assert require_fixed_seeds(tuple(reversed(FIXED_SEEDS))) == FIXED_SEEDS
    assert FIXED_SEEDS == (11111, 7, 11, 23, 42, 101)
    assert FIXED_N == 25
    with pytest.raises(ValueError, match="fixed"):
        require_fixed_seeds(FIXED_SEEDS[:-1])
    with pytest.raises(ValueError, match="fixed"):
        require_fixed_seeds(FIXED_SEEDS + (1,))


def test_rejects_misaligned_people_and_bad_codes():
    item = ChoiceItem("Q", 2, (0.5, 0.5))
    with pytest.raises(ValueError, match="same people"):
        paired_distances([[1], [2]], [[1]], [item])
    with pytest.raises(ValueError, match="one code per item"):
        paired_distances([[1, 1]], [[1]], [item])
    with pytest.raises(ValueError, match="outside"):
        option_shares([3], 2)
    with pytest.raises(ValueError, match="negative"):
        arm_distances([[1], [2]], [ChoiceItem("Q", 2, (1.0, -1.0))])


class _RowsRng:
    """Yield scripted uniform draws. One call is one permutation replicate."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows
        self.calls = 0

    def random(self, size: int) -> np.ndarray:
        row = self._rows[self.calls]
        self.calls += 1
        assert len(row) == size
        return np.asarray(row, dtype=float)


def test_choice_items_keep_the_run_shares():
    items = [{"item_id": "QID12", "kind": "choice", "options": ["Male", "Female"]}]
    baseline = {"items": [{"item_id": "QID12", "kind": "choice", "shares": [2, 2]}]}
    chosen = choice_items(items, baseline)
    assert chosen[0].human_shares == (2.0, 2.0)
    assert arm_distances([[1], [2]], chosen)["mean_tvd"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="not a choice"):
        choice_items([{"item_id": "Q", "kind": "numeric", "options": []}], baseline)


def test_answer_change_skips_a_missing_code():
    rows = answer_changes([[1, 1], [1, None]], [[2, 1], [1, 2]], ["sex", "age"])
    assert rows[0]["n_changed"] == 1
    assert rows[0]["n_comparable"] == 2
    assert rows[0]["change_rate"] == pytest.approx(0.5)
    assert rows[1]["n_changed"] == 0
    assert rows[1]["n_comparable"] == 1
    assert rows[1]["change_rate"] == pytest.approx(0.0)


def test_same_human_shares_reject_a_different_baseline():
    left = [ChoiceItem("Q", 2, (0.5, 0.5))]
    require_same_human_shares(left, [ChoiceItem("Q", 2, (0.5, 0.5))])
    with pytest.raises(ValueError, match="human shares"):
        require_same_human_shares(left, [ChoiceItem("Q", 2, (0.5 + 1e-6, 0.5 - 1e-6))])


def test_assemble_orders_seeds_and_holds_calibration_out():
    rows = [
        {"seed": seed, "mean_tvd_diff": 0.2, "mean_share_mse_diff": -0.01}
        for seed in reversed(FIXED_SEEDS)
    ]
    report = assemble_model_endpoint(rows, {"mean_tvd_diff": 5.0, "mean_share_mse_diff": 5.0})
    assert [row["seed"] for row in report["seeds"]] == list(FIXED_SEEDS)
    assert report["sign_flip"]["mean_tvd_diff"]["n"] == 6
    assert report["sign_flip"]["mean_tvd_diff"]["p"] == pytest.approx(2 / 64)
    assert report["sign_flip"]["mean_tvd_diff"]["values"] == pytest.approx([0.2] * 6)
    assert report["calibration"]["in_sign_flip"] is False
    assert report["calibration"]["mean_tvd_diff"] == pytest.approx(5.0)
    assert 5.0 not in report["sign_flip"]["mean_tvd_diff"]["values"]
    with pytest.raises(ValueError, match="fixed"):
        assemble_model_endpoint(rows + [rows[0]], None)


def test_batch_paths_are_the_n25_batch_arms():
    root = Path("runs/compare")
    assert batch_run_dir(root, 42, treatment=False).name == (
        "atus-categories-education_level-n25-seed42-batch"
    )
    assert batch_run_dir(root, 7, treatment=True).name == (
        "atus-categories-education_level-n25-seed7-t0.8-recent-batch"
    )
    repeat = batch_run_dir(root, 11111, treatment=True, repeat=True)
    assert repeat.name == "atus-categories-education_level-n25-seed11111-t0.8-recent-batch-repeat"
    assert "n50" not in repeat.name
    with pytest.raises(ValueError, match="fixed"):
        batch_run_dir(root, 256, treatment=False)
    with pytest.raises(ValueError, match="second treatment"):
        batch_run_dir(root, 11111, treatment=False, repeat=True)


def test_check_batch_arm_rejects_a_request_that_is_not_this_contrast(tmp_path: Path):
    control = tmp_path / "control"
    _write_arm(control, seed=11111, temperature=None, recent=False, code=1, n_people=1, item_ids=("QID12",))
    check_batch_arm(control, temperature=None, recent=False, seed=11111, n_people=1)
    treatment = tmp_path / "treatment"
    _write_arm(treatment, seed=11111, temperature=0.8, recent=True, code=2, n_people=1, item_ids=("QID12",))
    check_batch_arm(treatment, temperature=0.8, recent=True, seed=11111, n_people=1)

    gpt_path = treatment / "gpt-5.5" / "demographics" / "meta.json"
    gpt = json.loads(gpt_path.read_text())
    gpt["request"]["extra_body"] = {}
    gpt_path.write_text(json.dumps(gpt))
    with pytest.raises(ValueError, match="reasoning_effort"):
        check_batch_arm(treatment, temperature=0.8, recent=True, seed=11111, n_people=1)
    gpt["request"]["extra_body"] = {"reasoning_effort": "none"}
    gpt_path.write_text(json.dumps(gpt))

    glm_path = treatment / "glm-5.3" / "demographics" / "meta.json"
    glm = json.loads(glm_path.read_text())
    glm["request"]["extra_body"] = {"reasoning_effort": "none"}
    glm_path.write_text(json.dumps(glm))
    with pytest.raises(ValueError, match="should not set reasoning_effort"):
        check_batch_arm(treatment, temperature=0.8, recent=True, seed=11111, n_people=1)
    glm["request"]["extra_body"] = {}
    glm["batch"] = False
    glm_path.write_text(json.dumps(glm))
    with pytest.raises(ValueError, match="batch"):
        check_batch_arm(treatment, temperature=0.8, recent=True, seed=11111, n_people=1)

    (control / "recent.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="should not have"):
        check_batch_arm(control, temperature=None, recent=False, seed=11111, n_people=1)


def test_score_fixed_batch_reports_both_endpoints_without_calling_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def fail(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("model request")

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fail)
    for seed in FIXED_SEEDS:
        _write_arm(batch_run_dir(tmp_path, seed, treatment=False), seed=seed, temperature=None, recent=False, code=1)
        _write_arm(batch_run_dir(tmp_path, seed, treatment=True), seed=seed, temperature=0.8, recent=True, code=2)
    _write_arm(
        batch_run_dir(tmp_path, 11111, treatment=True, repeat=True),
        seed=11111,
        temperature=0.8,
        recent=True,
        code=2,
    )
    report = score_fixed_batch(tmp_path, reps=2, rng=np.random.default_rng(0))
    assert calls == []
    assert report["reps"] == 2
    assert [row["seed"] for row in report["runs"]] == list(FIXED_SEEDS)
    assert set(report["models"]) == set(COMPARE_MODEL_IDS)
    model = report["models"]["gpt-5.5"]
    assert model["sign_flip"]["mean_tvd_diff"]["values"] == pytest.approx([1.0] * 6)
    assert model["sign_flip"]["mean_share_mse_diff"]["values"] == pytest.approx([1.0] * 6)
    assert model["sign_flip"]["mean_tvd_diff"]["p"] == pytest.approx(2 / 64)
    assert model["sign_flip"]["mean_share_mse_diff"]["p"] == pytest.approx(2 / 64)
    assert model["calibration"]["mean_tvd_diff"] == pytest.approx(0.0)
    assert model["calibration"]["mean_share_mse_diff"] == pytest.approx(0.0)
    assert model["calibration"]["in_sign_flip"] is False
    assert model["seeds"][0]["card"][0]["label"] == "性别"
    assert model["seeds"][0]["card"][0]["change_rate"] == pytest.approx(1.0)
    assert [row["label"] for row in model["seeds"][0]["focus"]] == ["家庭收入", "政治倾向", "党派", "宗教活动"]
    assert model["seeds"][0]["focus"][0]["n_changed"] == FIXED_N
    assert report["recent"]["7"]["counts"] == {"flash-00": 9, "flash-01": 8, "flash-02": 8}
    text = render_paired_report(report)
    assert "平均 TVD 差" in text
    assert "平均份额 MSE 差" in text
    assert "（2/64）" in text
    assert "不进入上面的 6 个 seed" in text
    assert "其中 0 个不大于这次校准" in text
    written = write_paired_report(report, tmp_path / "report")
    saved = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert saved["models"]["glm-5.3"]["sign_flip"]["mean_tvd_diff"]["n"] == 6
    assert len(DEMOGRAPHIC_ITEM_IDS) == 14


def _write_arm(
    root: Path,
    *,
    seed: int,
    temperature: float | None,
    recent: bool,
    code: int,
    n_people: int = FIXED_N,
    item_ids: tuple[str, ...] = DEMOGRAPHIC_ITEM_IDS,
    models: tuple[str, ...] = COMPARE_MODEL_IDS,
) -> None:
    items = [{"item_id": item_id, "kind": "choice", "options": ["A", "B"]} for item_id in item_ids]
    baseline = {
        "items": [{"item_id": item_id, "kind": "choice", "shares": [1.0, 0.0]} for item_id in item_ids]
    }
    folder = root / "families" / "demographics"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "items.json").write_text(json.dumps(items), encoding="utf-8")
    (folder / "human_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    ids = [f"s{seed}-{index}" for index in range(n_people)]
    if recent:
        kinds = ("flash-00", "flash-01", "flash-02")
        payload = {
            "kinds": list(kinds),
            "people": [{"id": person_id, "kind": kinds[index % 3]} for index, person_id in enumerate(ids)],
        }
        (root / "recent.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for model in models:
        arm = root / model / "demographics"
        arm.mkdir(parents=True, exist_ok=True)
        extra: dict[str, str] = {}
        if model == "gpt-5.5" and temperature is not None:
            extra = {"reasoning_effort": "none"}
        meta = {
            "model": model,
            "n_people": n_people,
            "n_turns": len(item_ids),
            "batch": True,
            "seed": seed,
            "family": "demographics",
            "request": {"temperature": temperature, "extra_body": extra},
        }
        (arm / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        write_jsonl(
            arm / "responses.jsonl",
            [
                {
                    "id": person_id,
                    "answers": [{"item_id": item_id, "value": code, "ok": True} for item_id in item_ids],
                }
                for person_id in ids
            ],
        )


def _brute_sign_flip(values: list[float]) -> float:
    n = len(values)
    threshold = abs(sum(values) / n)
    extreme = 0
    for mask in range(1 << n):
        signed = 0.0
        for index, value in enumerate(values):
            signed += -value if (mask >> index) & 1 else value
        if abs(signed / n) + 1e-12 >= threshold:
            extreme += 1
    return extreme / (1 << n)
