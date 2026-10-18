"""Paired effect of temperature 0.8 plus recent on the same people.

Differences are the treatment mean minus the control mean. A negative
difference means the treatment arm is closer to the Twin-2K shares. Total
variation and share MSE use the same option proportions: TVD is half the L1
gap, and MSE is the mean squared gap across that item's options. The reported
means are unweighted averages over items.

Within a seed, each person's two answer rows are swapped together and both
mean differences are recomputed. That asks whether the two answer sheets are
exchangeable. It does not ask whether these people represent ATUS.

Across seeds, each seed contributes one difference per endpoint. Those
differences are sign-flipped under symmetry about zero. People stay inside
their seed and are not pooled.

The six seeds and the panel size below were fixed before looking at results.
This module only scores answers that were already saved. It does not call a
model. Starting the arms waits for a separate confirmation, and then uses
exactly these seeds and this panel size. Both arms of a seed have to be asked
the same way. The one-question n=50 seed 11111 directories are a different
prompt and a different draw from a `--batch` n=25 run, and stay out of this
contrast.

``score_fixed_batch`` reads those n=25 batch directories, including each run's
human shares. The repeat of seed 11111 is reported beside the six seeds and
is not one of them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from popbench.dao.io import read_jsonl
from popbench.evaluate.metrics import mean_squared_error, total_variation_distance

# Six is the smallest list on which a two-sided sign flip can fall under 0.05
# when every seed has the same sign. 256 and 2026 were dropped before any of
# these seeds were collected under --batch. n=25 is a new draw: it is not the
# first 25 of the existing n=50 sample.
FIXED_SEEDS: tuple[int, ...] = (11111, 7, 11, 23, 42, 101)
FIXED_N: int = 25
# (exceed + 1) / (reps + 1) with this reps uses a denominator of 10000.
MONTE_CARLO_REPS = 9999
COMPARE_MODEL_IDS: tuple[str, ...] = ("gpt-5.5", "glm-5.3", "deepseek-v4-pro")
DEMOGRAPHIC_ITEM_IDS: tuple[str, ...] = tuple(f"QID{number}" for number in range(11, 25))
# Sex, age, education, and race are on the persona card.
CARD_ITEM_IDS: tuple[str, ...] = ("QID12", "QID13", "QID14", "QID15")
# Income, political views, party, and religious attendance.
FOCUS_ITEM_IDS: tuple[str, ...] = ("QID21", "QID22", "QID20", "QID19")
ITEM_LABELS: dict[str, str] = {
    "QID12": "性别",
    "QID13": "年龄",
    "QID14": "学历",
    "QID15": "种族",
    "QID19": "宗教活动",
    "QID20": "党派",
    "QID21": "家庭收入",
    "QID22": "政治倾向",
}

# 2**16 assignments is still a direct loop. Larger arms need a Monte Carlo draw.
EXACT_PERSON_LIMIT = 16
MAX_SIGN_FLIP_SEEDS = 20
_TIE = 1e-12


@dataclass(frozen=True)
class ChoiceItem:
    """One closed-ended item and the Twin-2K option shares."""

    item_id: str
    n_options: int
    human_shares: tuple[float, ...]


def require_fixed_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Return the planned seeds when `seeds` is exactly that set.

    Run order may differ. Adding, dropping, or replacing a seed is a different
    experiment. This does not start a model run.
    """
    got = tuple(int(seed) for seed in seeds)
    if len(got) != len(set(got)) or sorted(got) != sorted(FIXED_SEEDS):
        raise ValueError(f"seeds are fixed before looking at results; expected {list(FIXED_SEEDS)}")
    return FIXED_SEEDS


def option_shares(codes: Sequence[int | None], n_options: int) -> list[float]:
    """Proportion on codes 1..n_options. Missing answers are left out."""
    if n_options < 2:
        raise ValueError("a choice item needs at least two options")
    valid = [_as_code(code, n_options) for code in codes if code is not None]
    if not valid:
        raise ValueError("item has no valid answers")
    total = len(valid)
    return [valid.count(option) / total for option in range(1, n_options + 1)]


def arm_distances(
    answers: Sequence[Sequence[int | None]],
    items: Sequence[ChoiceItem],
) -> dict[str, Any]:
    """Mean TVD and mean share MSE for one arm. Lower is closer to Twin-2K."""
    prepared = _prepare_items(items)
    matrix = _code_matrix(answers, prepared)
    return _arm_report(matrix, prepared)


def paired_distances(
    control: Sequence[Sequence[int | None]],
    treatment: Sequence[Sequence[int | None]],
    items: Sequence[ChoiceItem],
) -> dict[str, Any]:
    """Treatment minus control on aligned people. Negative is closer to Twin-2K."""
    prepared = _prepare_items(items)
    left, right = _paired_matrices(control, treatment, prepared)
    control_arm = _arm_report(left, prepared)
    treatment_arm = _arm_report(right, prepared)
    return _pair_report(control_arm, treatment_arm)


def permute_paired_arms(
    control: Sequence[Sequence[int | None]],
    treatment: Sequence[Sequence[int | None]],
    items: Sequence[ChoiceItem],
    *,
    reps: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Swap each person's two rows and recompute both mean differences.

    The swap moves every item for that person together. Both endpoints use the
    same swaps. With at most 16 people and no `reps`, every assignment is
    enumerated. Larger arms need `reps` and `rng`. The Monte Carlo p-value is
    `(exceed + 1) / (reps + 1)`, counting the observed assignment in the +1.
    """
    prepared = _prepare_items(items)
    left, right = _paired_matrices(control, treatment, prepared)
    observed = _pair_report(_arm_report(left, prepared), _arm_report(right, prepared))
    n_people = left.shape[0]
    if reps is None:
        if n_people > EXACT_PERSON_LIMIT:
            raise ValueError(
                f"more than {EXACT_PERSON_LIMIT} people needs reps and rng; "
                "the assignments are not enumerated"
            )
        p_tvd, p_mse, n_assignments = _exact_permutation(left, right, prepared, observed)
        observed["method"] = "exact"
        observed["n_assignments"] = n_assignments
    else:
        if reps < 1:
            raise ValueError("reps must be at least 1")
        if rng is None:
            raise ValueError("monte carlo permutation needs rng")
        p_tvd, p_mse = _monte_carlo_permutation(left, right, prepared, observed, reps, rng)
        observed["method"] = "monte_carlo"
        observed["reps"] = reps
    observed["p_mean_tvd"] = p_tvd
    observed["p_mean_share_mse"] = p_mse
    return observed


def sign_flip_mean(values: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided sign flip of the mean. Every sign pattern is enumerated.

    The reference distribution is the observed differences with independent
    signs. A value of zero stays in the mean. The p-value is the share of
    patterns whose absolute mean is at least the observed absolute mean.
    """
    diffs = [float(value) for value in values]
    if not diffs:
        raise ValueError("sign flip needs at least one difference")
    if len(diffs) > MAX_SIGN_FLIP_SEEDS:
        raise ValueError(f"exact sign flip supports at most {MAX_SIGN_FLIP_SEEDS} seeds")
    if not all(math.isfinite(value) for value in diffs):
        raise ValueError("differences must be finite")
    n = len(diffs)
    observed = sum(diffs) / n
    threshold = abs(observed)
    extreme = 0
    total = 1 << n
    for mask in range(total):
        signed = 0.0
        for index, value in enumerate(diffs):
            signed += -value if (mask >> index) & 1 else value
        if abs(signed / n) + _TIE >= threshold:
            extreme += 1
    return {
        "values": diffs,
        "mean": observed,
        "min": min(diffs),
        "max": max(diffs),
        "p": extreme / total,
        "n": n,
        "n_positive": sum(value > 0 for value in diffs),
        "n_negative": sum(value < 0 for value in diffs),
        "n_zero": sum(value == 0 for value in diffs),
    }


def cross_seed_sign_flip(seed_diffs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sign-flip the seed-level differences. People are not pooled across seeds."""
    if not seed_diffs:
        raise ValueError("sign flip needs at least one seed")
    tvd: list[float] = []
    mse: list[float] = []
    seeds: list[Any] = []
    for row in seed_diffs:
        if "mean_tvd_diff" not in row or "mean_share_mse_diff" not in row:
            raise ValueError("each seed needs mean_tvd_diff and mean_share_mse_diff")
        tvd.append(float(row["mean_tvd_diff"]))
        mse.append(float(row["mean_share_mse_diff"]))
        seeds.append(row.get("seed"))
    return {
        "n_seeds": len(seed_diffs),
        "seeds": seeds,
        "mean_tvd_diff": sign_flip_mean(tvd),
        "mean_share_mse_diff": sign_flip_mean(mse),
    }


def codes_by_person(
    responses: Sequence[Mapping[str, Any]],
    item_ids: Sequence[str],
) -> dict[str, list[int | None]]:
    """Map person id to choice codes. A failed answer is None."""
    if not item_ids:
        raise ValueError("need at least one item")
    wanted = [str(item_id) for item_id in item_ids]
    coded: dict[str, list[int | None]] = {}
    for person in responses:
        person_id = str(person["id"])
        if person_id in coded:
            raise ValueError(f"duplicate person id {person_id}")
        by_item = {str(row["item_id"]): row for row in person["answers"]}
        codes: list[int | None] = []
        for item_id in wanted:
            row = by_item.get(item_id)
            if row is None:
                raise ValueError(f"{person_id} is missing {item_id}")
            value = row.get("value")
            codes.append(value if row.get("ok") and value is not None else None)
        coded[person_id] = codes
    if not coded:
        raise ValueError("responses are empty")
    return coded


def align_by_id(
    control: Mapping[str, Sequence[int | None]],
    treatment: Mapping[str, Sequence[int | None]],
) -> tuple[list[str], list[list[int | None]], list[list[int | None]]]:
    """Pair rows by person id. Row order follows the control arm."""
    if not control:
        raise ValueError("control arm is empty")
    missing = [person_id for person_id in control if person_id not in treatment]
    extra = [person_id for person_id in treatment if person_id not in control]
    if missing or extra:
        raise ValueError("control and treatment person ids differ")
    ids = [str(person_id) for person_id in control]
    return (
        ids,
        [list(control[person_id]) for person_id in control],
        [list(treatment[person_id]) for person_id in control],
    )


def load_arm(path: Path, item_ids: Sequence[str]) -> dict[str, list[int | None]]:
    """Read one saved responses file. Does not call a model."""
    return codes_by_person(read_jsonl(Path(path)), item_ids)


def _prepare_items(items: Sequence[ChoiceItem]) -> list[tuple[str, int, list[float]]]:
    if not items:
        raise ValueError("need at least one item")
    prepared = []
    for item in items:
        if item.n_options < 2:
            raise ValueError(f"{item.item_id} needs at least two options")
        if len(item.human_shares) != item.n_options:
            raise ValueError(f"{item.item_id} shares do not match its options")
        prepared.append((str(item.item_id), int(item.n_options), _proportions(item.human_shares)))
    return prepared


def _proportions(values: Sequence[float]) -> list[float]:
    shares = [float(value) for value in values]
    if any(value < 0 for value in shares):
        raise ValueError("shares cannot be negative")
    total = sum(shares)
    if total <= 0:
        raise ValueError("shares must sum to a positive number")
    return [value / total for value in shares]


def _as_code(value: Any, n_options: int) -> int:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise ValueError(f"choice code must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        number = int(value)
    else:
        raise ValueError(f"choice code must be an integer, got {value!r}")
    if number < 1 or number > n_options:
        raise ValueError(f"choice code {number} is outside 1..{n_options}")
    return number


def _code_matrix(
    answers: Sequence[Sequence[int | None]],
    items: Sequence[tuple[str, int, list[float]]],
) -> np.ndarray:
    if not answers:
        raise ValueError("an arm has no people")
    n_items = len(items)
    matrix = np.zeros((len(answers), n_items), dtype=int)
    for row_index, row in enumerate(answers):
        if len(row) != n_items:
            raise ValueError("each person needs one code per item")
        for column, code in enumerate(row):
            if code is None:
                continue
            matrix[row_index, column] = _as_code(code, items[column][1])
    return matrix


def _paired_matrices(
    control: Sequence[Sequence[int | None]],
    treatment: Sequence[Sequence[int | None]],
    items: Sequence[tuple[str, int, list[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(control) != len(treatment):
        raise ValueError("control and treatment must list the same people in the same order")
    return _code_matrix(control, items), _code_matrix(treatment, items)


def _column_shares(column: np.ndarray, n_options: int) -> list[float]:
    valid = [int(code) for code in column if int(code) > 0]
    if not valid:
        raise ValueError("item has no valid answers")
    total = len(valid)
    return [valid.count(option) / total for option in range(1, n_options + 1)]


def _arm_report(
    codes: np.ndarray,
    items: Sequence[tuple[str, int, list[float]]],
) -> dict[str, Any]:
    rows = []
    for column, (item_id, n_options, human_shares) in enumerate(items):
        shares = _column_shares(codes[:, column], n_options)
        n_valid = int(np.count_nonzero(codes[:, column]))
        rows.append(
            {
                "item_id": item_id,
                "n_options": n_options,
                "n": n_valid,
                "tvd": total_variation_distance(shares, human_shares),
                "share_mse": mean_squared_error(shares, human_shares),
            }
        )
    return {
        "n_people": int(codes.shape[0]),
        "n_items": len(rows),
        "mean_tvd": sum(row["tvd"] for row in rows) / len(rows),
        "mean_share_mse": sum(row["share_mse"] for row in rows) / len(rows),
        "items": rows,
    }


def _pair_report(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    items = []
    for left, right in zip(control["items"], treatment["items"], strict=True):
        items.append(
            {
                "item_id": left["item_id"],
                "n_options": left["n_options"],
                "n_control": left["n"],
                "n_treatment": right["n"],
                "tvd_control": left["tvd"],
                "tvd_treatment": right["tvd"],
                "tvd_diff": right["tvd"] - left["tvd"],
                "share_mse_control": left["share_mse"],
                "share_mse_treatment": right["share_mse"],
                "share_mse_diff": right["share_mse"] - left["share_mse"],
            }
        )
    return {
        "n_people": control["n_people"],
        "n_items": control["n_items"],
        "mean_tvd_control": control["mean_tvd"],
        "mean_tvd_treatment": treatment["mean_tvd"],
        "mean_tvd_diff": treatment["mean_tvd"] - control["mean_tvd"],
        "mean_share_mse_control": control["mean_share_mse"],
        "mean_share_mse_treatment": treatment["mean_share_mse"],
        "mean_share_mse_diff": treatment["mean_share_mse"] - control["mean_share_mse"],
        "items": items,
    }


def _diffs_under_swap(
    control: np.ndarray,
    treatment: np.ndarray,
    items: Sequence[tuple[str, int, list[float]]],
    swap: np.ndarray,
) -> tuple[float, float]:
    left = np.where(swap[:, None], treatment, control)
    right = np.where(swap[:, None], control, treatment)
    control_arm = _arm_report(left, items)
    treatment_arm = _arm_report(right, items)
    return (
        treatment_arm["mean_tvd"] - control_arm["mean_tvd"],
        treatment_arm["mean_share_mse"] - control_arm["mean_share_mse"],
    )


def _exact_permutation(
    control: np.ndarray,
    treatment: np.ndarray,
    items: Sequence[tuple[str, int, list[float]]],
    observed: Mapping[str, Any],
) -> tuple[float, float, int]:
    n_people = control.shape[0]
    indexes = np.arange(n_people)
    tvd_threshold = abs(float(observed["mean_tvd_diff"]))
    mse_threshold = abs(float(observed["mean_share_mse_diff"]))
    extreme_tvd = 0
    extreme_mse = 0
    n_assignments = 1 << n_people
    for mask in range(n_assignments):
        swap = ((mask >> indexes) & 1).astype(bool)
        tvd_diff, mse_diff = _diffs_under_swap(control, treatment, items, swap)
        if abs(tvd_diff) + _TIE >= tvd_threshold:
            extreme_tvd += 1
        if abs(mse_diff) + _TIE >= mse_threshold:
            extreme_mse += 1
    return extreme_tvd / n_assignments, extreme_mse / n_assignments, n_assignments


def _monte_carlo_permutation(
    control: np.ndarray,
    treatment: np.ndarray,
    items: Sequence[tuple[str, int, list[float]]],
    observed: Mapping[str, Any],
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n_people = control.shape[0]
    tvd_threshold = abs(float(observed["mean_tvd_diff"]))
    mse_threshold = abs(float(observed["mean_share_mse_diff"]))
    extreme_tvd = 0
    extreme_mse = 0
    for _ in range(reps):
        swap = np.asarray(rng.random(n_people)) < 0.5
        tvd_diff, mse_diff = _diffs_under_swap(control, treatment, items, swap)
        if abs(tvd_diff) + _TIE >= tvd_threshold:
            extreme_tvd += 1
        if abs(mse_diff) + _TIE >= mse_threshold:
            extreme_mse += 1
    return (extreme_tvd + 1) / (reps + 1), (extreme_mse + 1) / (reps + 1)


def choice_items(
    items: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> list[ChoiceItem]:
    """Choice items in run order, with the human shares stored for that run."""
    if not items:
        raise ValueError("need at least one item")
    gold = {str(row["item_id"]): row for row in baseline["items"]}
    chosen: list[ChoiceItem] = []
    for item in items:
        item_id = str(item["item_id"])
        if item.get("kind") != "choice":
            raise ValueError(f"{item_id} is not a choice item")
        row = gold.get(item_id)
        if row is None or row.get("kind") != "choice":
            raise ValueError(f"{item_id} has no human shares")
        options = list(item["options"])
        shares = tuple(float(value) for value in row["shares"])
        if len(options) < 2 or len(shares) != len(options):
            raise ValueError(f"{item_id} shares do not match its options")
        chosen.append(ChoiceItem(item_id, len(options), shares))
    return chosen


def load_choice_items(run_root: Path, family: str = "demographics") -> list[ChoiceItem]:
    """Read ``items.json`` and ``human_baseline.json`` from one saved run."""
    folder = Path(run_root) / "families" / family
    items_path = folder / "items.json"
    baseline_path = folder / "human_baseline.json"
    if not items_path.is_file():
        raise ValueError(f"missing {items_path}")
    if not baseline_path.is_file():
        raise ValueError(f"missing {baseline_path}")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    return choice_items(items, baseline)


def require_same_human_shares(left: Sequence[ChoiceItem], right: Sequence[ChoiceItem]) -> None:
    """The two arms have to be scored against the same Twin-2K shares."""
    if [item.item_id for item in left] != [item.item_id for item in right]:
        raise ValueError("control and treatment items differ")
    for control, treatment in zip(left, right, strict=True):
        if control.n_options != treatment.n_options:
            raise ValueError(f"{control.item_id} option counts differ")
        shares_differ = any(
            abs(float(a) - float(b)) > 1e-9 for a, b in zip(control.human_shares, treatment.human_shares)
        )
        if shares_differ:
            raise ValueError(f"{control.item_id} human shares differ between arms")


def answer_changes(
    control: Sequence[Sequence[int | None]],
    treatment: Sequence[Sequence[int | None]],
    item_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Count people whose two valid answers differ. A missing answer stays out."""
    if len(control) != len(treatment):
        raise ValueError("control and treatment must list the same people in the same order")
    rows = []
    n_people = len(control)
    for column, item_id in enumerate(item_ids):
        changed = 0
        comparable = 0
        for left, right in zip(control, treatment, strict=True):
            if len(left) != len(item_ids) or len(right) != len(item_ids):
                raise ValueError("each person needs one code per item")
            first, second = left[column], right[column]
            if first is None or second is None:
                continue
            comparable += 1
            if first != second:
                changed += 1
        rows.append(
            {
                "item_id": str(item_id),
                "n_changed": changed,
                "n_comparable": comparable,
                "n_people": n_people,
                "change_rate": (changed / comparable) if comparable else None,
            }
        )
    return rows


def score_response_files(
    control_path: Path,
    treatment_path: Path,
    items: Sequence[ChoiceItem],
    *,
    reps: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Pair saved answer files by person id and score both endpoints.

    Does not call a model. ``reps`` is required once an arm has more than 16 people.
    """
    item_ids = [item.item_id for item in items]
    control = load_arm(control_path, item_ids)
    treatment = load_arm(treatment_path, item_ids)
    _person_ids, left, right = align_by_id(control, treatment)
    scored = permute_paired_arms(left, right, items, reps=reps, rng=rng)
    scored["answer_changes"] = answer_changes(left, right, item_ids)
    return scored


def recent_kind_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    """How many people drew each recent event. The draw rule stays as stored."""
    people = payload["people"]
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for row in people:
        person_id = str(row["id"])
        if person_id in seen:
            raise ValueError(f"recent.json repeats {person_id}")
        seen.add(person_id)
        kind = str(row["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return {"n": len(people), "counts": counts}


def batch_run_dir(
    compare_dir: Path,
    seed: int,
    *,
    treatment: bool,
    repeat: bool = False,
) -> Path:
    """Directory name for one fixed n=25 batch arm. n=50 paths are not produced."""
    if repeat and not treatment:
        raise ValueError("the repeat arm is a second treatment, not a second control")
    seed = int(seed)
    if seed not in FIXED_SEEDS:
        raise ValueError(f"seed {seed} is outside the fixed list")
    if repeat and seed != 11111:
        raise ValueError("calibration repeats only seed 11111")
    stem = f"atus-categories-education_level-n{FIXED_N}-seed{seed}"
    if treatment:
        suffix = "-t0.8-recent-batch-repeat" if repeat else "-t0.8-recent-batch"
    else:
        suffix = "-batch"
    return Path(compare_dir) / f"{stem}{suffix}"


def check_batch_arm(
    run_root: Path,
    *,
    temperature: float | None,
    recent: bool,
    seed: int,
    n_people: int = FIXED_N,
    models: Sequence[str] = COMPARE_MODEL_IDS,
    family: str = "demographics",
) -> None:
    """Check one saved batch arm. Does not call a model or delete a cache."""
    run_root = Path(run_root)
    if not models:
        raise ValueError("need at least one model")
    items = load_choice_items(run_root, family)
    item_ids = [item.item_id for item in items]
    n_items = len(item_ids)
    recent_ids = _check_recent_file(run_root, recent=recent, n_people=n_people)
    seen_ids: set[str] | None = None
    for model in models:
        ids = _check_model_arm(
            run_root,
            model,
            family=family,
            item_ids=item_ids,
            n_items=n_items,
            n_people=n_people,
            seed=int(seed),
            temperature=temperature,
        )
        if seen_ids is None:
            seen_ids = ids
        elif seen_ids != ids:
            raise ValueError(f"{run_root} models do not share person ids")
    if recent and seen_ids != set(recent_ids):
        raise ValueError(f"{run_root} recent.json person ids differ from the answers")


def assemble_model_endpoint(
    seed_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Sign-flip the six seed differences. Calibration stays outside that list."""
    require_fixed_seeds([int(row["seed"]) for row in seed_rows])
    by_seed = {int(row["seed"]): row for row in seed_rows}
    ordered = [dict(by_seed[seed]) for seed in FIXED_SEEDS]
    signed = cross_seed_sign_flip(ordered)
    held_out = None
    if calibration is not None:
        held_out = {
            "seed": 11111,
            "contrast": "repeat minus first treatment",
            "mean_tvd_diff": float(calibration["mean_tvd_diff"]),
            "mean_share_mse_diff": float(calibration["mean_share_mse_diff"]),
            "in_sign_flip": False,
        }
    return {"seeds": ordered, "sign_flip": signed, "calibration": held_out}


def score_fixed_batch(
    compare_dir: Path,
    *,
    reps: int = MONTE_CARLO_REPS,
    rng: np.random.Generator | None = None,
    models: Sequence[str] = COMPARE_MODEL_IDS,
) -> dict[str, Any]:
    """Score the fixed six seeds and the one repeat arm from saved files.

    Does not call a model. People stay inside their seed. The repeat arm is
    the second temperature-0.8 draw minus the first, and it is not a seventh seed.
    """
    compare_dir = Path(compare_dir)
    if reps < 1:
        raise ValueError("reps must be at least 1")
    generator = np.random.default_rng(0) if rng is None else rng
    per_model: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    recent: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []
    for seed in FIXED_SEEDS:
        control_root = batch_run_dir(compare_dir, seed, treatment=False)
        treatment_root = batch_run_dir(compare_dir, seed, treatment=True)
        check_batch_arm(control_root, temperature=None, recent=False, seed=seed, models=models)
        check_batch_arm(treatment_root, temperature=0.8, recent=True, seed=seed, models=models)
        control_items = load_choice_items(control_root)
        treatment_items = load_choice_items(treatment_root)
        require_same_human_shares(control_items, treatment_items)
        _require_planned_items(control_items)
        recent[str(seed)] = recent_kind_counts(_read_recent(treatment_root))
        runs.append(
            {
                "seed": seed,
                "control": control_root.name,
                "treatment": treatment_root.name,
            }
        )
        for model in models:
            scored = score_response_files(
                control_root / model / "demographics" / "responses.jsonl",
                treatment_root / model / "demographics" / "responses.jsonl",
                control_items,
                reps=reps,
                rng=generator,
            )
            per_model[model].append(_seed_row(seed, scored))
    treatment_11111 = batch_run_dir(compare_dir, 11111, treatment=True)
    calibration_root = batch_run_dir(compare_dir, 11111, treatment=True, repeat=True)
    check_batch_arm(calibration_root, temperature=0.8, recent=True, seed=11111, models=models)
    _require_same_recent(treatment_11111, calibration_root)
    calibration_items = load_choice_items(calibration_root)
    require_same_human_shares(load_choice_items(treatment_11111), calibration_items)
    report_models = {}
    for model in models:
        left, right = _aligned_codes(
            treatment_11111 / model / "demographics" / "responses.jsonl",
            calibration_root / model / "demographics" / "responses.jsonl",
            calibration_items,
        )
        shift = paired_distances(left, right, calibration_items)
        report_models[model] = assemble_model_endpoint(
            per_model[model],
            {
                "mean_tvd_diff": shift["mean_tvd_diff"],
                "mean_share_mse_diff": shift["mean_share_mse_diff"],
            },
        )
    return {
        "n": FIXED_N,
        "seeds": list(FIXED_SEEDS),
        "family": "demographics",
        "population": "education_level",
        "reps": reps,
        "runs": runs,
        "calibration_run": calibration_root.name,
        "models": report_models,
        "recent": recent,
    }


def render_paired_report(report: Mapping[str, Any]) -> str:
    """Chinese report. Total variation and share MSE stay in separate sections."""
    seeds = "、".join(str(seed) for seed in report["seeds"])
    lines = [
        "# 温度 0.8 加 recent 的配对差",
        "",
        f"- 人：教育分层，每个 seed {report['n']} 人。seed 为 {seeds}。",
        "- 对照：省略温度，无 recent，14 道人口学题一次请求。",
        "- 处理：温度 0.8，加 recent，同样一次请求。",
        "- 差：处理均值减对照均值。负值表示处理更接近 Twin-2K。",
        "- 平均是 14 题的不加权平均。总变异距离和份额 MSE 分开检验。",
        "- 同一个 seed 里，置换问的是两份答案能否互换。人留在各自的 seed 里。",
        (
            f"- 每个 seed {report['n']} 人，超过 16，所以 seed 内置换是 {report['reps']} 次蒙特卡洛。"
            "p =（超过次数 + 1）/（次数 + 1）。"
        ),
        "- gpt-5.5 的对照是默认推理、省略温度。它的处理是关掉推理、温度 0.8、加 recent。",
        "- glm-5.3 和 deepseek-v4-pro 的处理只改温度并加 recent。",
        "",
        "校准是 seed 11111 的处理臂再答一遍，用重复作答减第一次处理。它单独列出，不进入上面的 6 个 seed。",
        "",
    ]
    for model, payload in report["models"].items():
        lines.extend(_render_model(str(model), payload))
    lines.extend(["## recent 配比", ""])
    lines.append("事件按人物 id 抽取。下面是每个 seed 抽到的人数。")
    lines.append("")
    for seed in report["seeds"]:
        mix = report["recent"][str(seed)]
        counts = "、".join(f"{kind} {count}" for kind, count in mix["counts"].items())
        lines.append(f"- seed {seed}：{mix['n']} 人，{counts}。")
    lines.append("")
    return "\n".join(lines)


def write_paired_report(report: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Write the scored report next to the runs. Does not call a model."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = _plain(report)
    json_path = target / "paired-effect.json"
    markdown_path = target / "paired-effect.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_paired_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _check_recent_file(run_root: Path, *, recent: bool, n_people: int) -> list[str]:
    path = run_root / "recent.json"
    if not recent:
        if path.exists():
            raise ValueError(f"control arm should not have {path}")
        return []
    if not path.is_file():
        raise ValueError(f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = payload["people"]
    if len(people) != n_people:
        raise ValueError(f"{path} has {len(people)} people, expected {n_people}")
    return [str(row["id"]) for row in people]


def _check_model_arm(
    run_root: Path,
    model: str,
    *,
    family: str,
    item_ids: Sequence[str],
    n_items: int,
    n_people: int,
    seed: int,
    temperature: float | None,
) -> set[str]:
    arm = run_root / model / family
    meta_path = arm / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != model:
        raise ValueError(f"{meta_path} model is {meta.get('model')!r}")
    if meta.get("n_people") != n_people:
        raise ValueError(f"{meta_path} n_people is {meta.get('n_people')}")
    if meta.get("n_turns") != n_items:
        raise ValueError(f"{meta_path} n_turns is {meta.get('n_turns')}")
    if meta.get("batch") is not True:
        raise ValueError(f"{meta_path} batch is not true")
    if meta.get("family") != family:
        raise ValueError(f"{meta_path} family is {meta.get('family')!r}")
    if meta.get("seed") != seed:
        raise ValueError(f"{meta_path} seed is {meta.get('seed')}")
    _check_request(model, meta.get("request") or {}, temperature)
    response_path = arm / "responses.jsonl"
    if not response_path.is_file():
        raise ValueError(f"missing {response_path}")
    responses = read_jsonl(response_path)
    if len(responses) != n_people:
        raise ValueError(f"{response_path} has {len(responses)} people, expected {n_people}")
    ids: list[str] = []
    for person in responses:
        person_id = str(person["id"])
        ids.append(person_id)
        answers = person["answers"]
        if [str(row["item_id"]) for row in answers] != list(item_ids):
            raise ValueError(f"{person_id} items differ from the run")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{model} repeats a person id")
    return set(ids)


def _check_request(model: str, request: Mapping[str, Any], temperature: float | None) -> None:
    got = request.get("temperature")
    if temperature is None:
        if got is not None:
            raise ValueError(f"{model} temperature should be omitted")
    elif got is None or abs(float(got) - float(temperature)) > 1e-9:
        raise ValueError(f"{model} temperature is {got}")
    extra = request.get("extra_body") or {}
    effort = extra.get("reasoning_effort")
    if model == "gpt-5.5" and temperature is not None:
        if effort != "none":
            raise ValueError("gpt-5.5 temperature requires reasoning_effort=none")
    elif effort:
        raise ValueError(f"{model} should not set reasoning_effort")


def _require_planned_items(items: Sequence[ChoiceItem]) -> None:
    got = tuple(item.item_id for item in items)
    if got != DEMOGRAPHIC_ITEM_IDS:
        raise ValueError("demographics items are not the planned 14 questions")


def _read_recent(run_root: Path) -> dict[str, Any]:
    return json.loads((Path(run_root) / "recent.json").read_text(encoding="utf-8"))


def _require_same_recent(left: Path, right: Path) -> None:
    if _recent_by_id(left) != _recent_by_id(right):
        raise ValueError("calibration recent events differ from the first treatment")


def _recent_by_id(run_root: Path) -> dict[str, str]:
    payload = _read_recent(run_root)
    recent_kind_counts(payload)
    return {str(row["id"]): str(row["kind"]) for row in payload["people"]}


def _aligned_codes(
    control_path: Path,
    treatment_path: Path,
    items: Sequence[ChoiceItem],
) -> tuple[list[list[int | None]], list[list[int | None]]]:
    item_ids = [item.item_id for item in items]
    _ids, left, right = align_by_id(load_arm(control_path, item_ids), load_arm(treatment_path, item_ids))
    return left, right


def _seed_row(seed: int, scored: Mapping[str, Any]) -> dict[str, Any]:
    changes = {row["item_id"]: row for row in scored["answer_changes"]}
    items = {row["item_id"]: row for row in scored["items"]}
    return {
        "seed": seed,
        "n_people": scored["n_people"],
        "n_items": scored["n_items"],
        "mean_tvd_control": scored["mean_tvd_control"],
        "mean_tvd_treatment": scored["mean_tvd_treatment"],
        "mean_tvd_diff": scored["mean_tvd_diff"],
        "mean_share_mse_control": scored["mean_share_mse_control"],
        "mean_share_mse_treatment": scored["mean_share_mse_treatment"],
        "mean_share_mse_diff": scored["mean_share_mse_diff"],
        "p_mean_tvd": scored["p_mean_tvd"],
        "p_mean_share_mse": scored["p_mean_share_mse"],
        "method": scored["method"],
        "reps": scored.get("reps"),
        "card": [_change_view(item_id, changes[item_id]) for item_id in CARD_ITEM_IDS],
        "focus": [_focus_view(item_id, items[item_id], changes[item_id]) for item_id in FOCUS_ITEM_IDS],
    }


def _change_view(item_id: str, change: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "label": ITEM_LABELS[item_id],
        "n_changed": change["n_changed"],
        "n_comparable": change["n_comparable"],
        "n_people": change["n_people"],
        "change_rate": change["change_rate"],
    }


def _focus_view(item_id: str, item: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "label": ITEM_LABELS[item_id],
        "tvd_diff": item["tvd_diff"],
        "share_mse_diff": item["share_mse_diff"],
        "n_changed": change["n_changed"],
        "n_comparable": change["n_comparable"],
    }


def _render_model(model: str, payload: Mapping[str, Any]) -> list[str]:
    lines = [f"## {model}", ""]
    lines.extend(_render_endpoint(payload, "平均 TVD 差", "mean_tvd_diff", "p_mean_tvd"))
    lines.extend(_render_endpoint(payload, "平均份额 MSE 差", "mean_share_mse_diff", "p_mean_share_mse"))
    calibration = payload["calibration"]
    lines.extend(
        [
            "### 校准",
            "",
            "重复作答减第一次处理。下面两个数不进入 6 个 seed 的符号翻转。",
            "",
            f"- 平均 TVD 差：{_fmt(calibration['mean_tvd_diff'])}",
            f"- 平均份额 MSE 差：{_fmt(calibration['mean_share_mse_diff'])}",
            "",
            _noise_line(
                "平均 TVD 差",
                payload["sign_flip"]["mean_tvd_diff"]["values"],
                calibration["mean_tvd_diff"],
            ),
            _noise_line(
                "平均份额 MSE 差",
                payload["sign_flip"]["mean_share_mse_diff"]["values"],
                calibration["mean_share_mse_diff"],
            ),
            "",
            "### 卡片上能对应的四题",
            "",
            "性别、年龄、学历、种族写在人物卡上。这四题是阴性对照。改答率高时，主终点的位移来自改写卡片上已有的答案。",
            "",
            _change_table(payload["seeds"]),
            "",
            "### 家庭收入、政治倾向、党派、宗教活动",
            "",
            _focus_table(payload["seeds"]),
            "",
        ]
    )
    return lines


def _render_endpoint(
    payload: Mapping[str, Any],
    title: str,
    diff_key: str,
    p_key: str,
) -> list[str]:
    flip = payload["sign_flip"][diff_key]
    lines = [
        f"### {title}",
        "",
        "| seed | 对照 | 处理 | 差 | 置换 p |",
        "| --- | --- | --- | --- | --- |",
    ]
    control_key = diff_key.replace("_diff", "_control")
    treatment_key = diff_key.replace("_diff", "_treatment")
    for row in payload["seeds"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    _fmt(row[control_key]),
                    _fmt(row[treatment_key]),
                    _fmt(row[diff_key]),
                    _fmt(row[p_key]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                f"6 个数：{', '.join(_fmt(value) for value in flip['values'])}。"
                f"均值 {_fmt(flip['mean'])}，最小 {_fmt(flip['min'])}，最大 {_fmt(flip['max'])}。"
            ),
            _sign_sentence(flip),
            "",
        ]
    )
    return lines


def _change_table(seeds: Sequence[Mapping[str, Any]]) -> str:
    header = "| 题 | " + " | ".join(str(row["seed"]) for row in seeds) + " |"
    rule = "| --- | " + " | ".join("---" for _ in seeds) + " |"
    lines = [header, rule]
    for index, card in enumerate(seeds[0]["card"]):
        cells = []
        for row in seeds:
            item = row["card"][index]
            rate = item["change_rate"]
            shown = "—" if rate is None else f"{item['n_changed']}/{item['n_comparable']} ({rate:.2f})"
            cells.append(shown)
        lines.append(f"| {card['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _focus_table(seeds: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| seed | 题 | TVD 差 | 份额 MSE 差 | 改答人数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in seeds:
        for item in row["focus"]:
            lines.append(
                f"| {row['seed']} | {item['label']} | {_fmt(item['tvd_diff'])} | "
                f"{_fmt(item['share_mse_diff'])} | {item['n_changed']}/{item['n_comparable']} |"
            )
    return "\n".join(lines)


def _sign_sentence(flip: Mapping[str, Any]) -> str:
    mean = float(flip["mean"])
    p_value = float(flip["p"])
    pattern = f"{flip['n_negative']} 个为负，{flip['n_positive']} 个为正，{flip['n_zero']} 个为零"
    if mean < 0:
        direction = "平均差为负，处理更接近 Twin-2K"
    elif mean > 0:
        direction = "平均差为正，处理更远离 Twin-2K"
    else:
        direction = "平均差为零"
    if p_value < 0.05 and mean != 0:
        decision = f"双侧符号翻转 p={_p_text(flip)}，低于 0.05"
    else:
        decision = f"双侧符号翻转 p={_p_text(flip)}，不低于 0.05"
    return f"{pattern}。{direction}。{decision}。"


def _noise_line(label: str, values: Sequence[float], calibration: float) -> str:
    absolute = [abs(float(value)) for value in values]
    scale = abs(float(calibration))
    within = sum(value <= scale + 1e-12 for value in absolute)
    return (
        f"{label}的校准绝对差是 {_fmt(scale)}。"
        f"六个 seed 差值的绝对值从 {_fmt(min(absolute))} 到 {_fmt(max(absolute))}，"
        f"其中 {within} 个不大于这次校准。"
    )


def _p_text(flip: Mapping[str, Any]) -> str:
    total = 1 << int(flip["n"])
    hits = int(round(float(flip["p"]) * total))
    return f"{float(flip['p']):.4f}（{hits}/{total}）"


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
