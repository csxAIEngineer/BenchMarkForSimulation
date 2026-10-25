"""Paired contrast of temperature 0 with and without one recent event.

Reads answers that were already saved. Does not call a model. The difference
is the recent arm minus the arm without an event. Choice questions use two
distances against Twin-2K wave 1–3 shares: total variation, and the mean
squared gap across options. Both means are unweighted across choice items.

One seed and one draw. The report states the size of each gap. It does not
run a cross-seed test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from popbench.dao.io import read_jsonl
from popbench.dao.survey import CATEGORY_LABELS
from popbench.evaluate.metrics import mean_squared_error
from popbench.evaluate.survey import score_full
from popbench.simulate.recent import (
    EDUCATION_RECENT_DRAW,
)

CONTROL_RUN = Path("runs/compare/atus-categories-economy-n50-seed0-t0")
TREATMENT_RUN = Path("runs/compare/atus-categories-economy-n50-seed0-t0-recent")
REPORT_DIR = Path("reports/t0-economy-recent")
DEFAULT_FAMILIES: tuple[str, ...] = ("economic", "experiments", "pricing")
FOCUS_FAMILIES: tuple[str, ...] = ("economic", "pricing")
MODEL = "deepseek-v4-pro"
# The saved economy run used three stored events, so the modulus was 3.
# Later draws name the same rule as free_flash_count.
ACCEPTED_DRAWS = (
    EDUCATION_RECENT_DRAW,
    'sha256("education-categories-recent:" + person_id) % 3',
)


def score_t0_economy_recent(
    control_root: Path,
    treatment_root: Path,
    *,
    model: str = MODEL,
    families: Sequence[str] = DEFAULT_FAMILIES,
    temperature: float = 0.0,
    seed: int = 0,
    n_people: int | None = None,
) -> dict[str, Any]:
    """Score saved category runs. The recent arm minus the plain arm."""
    control_root = Path(control_root)
    treatment_root = Path(treatment_root)
    if control_root.resolve() == treatment_root.resolve():
        raise ValueError("control and treatment directories are the same")
    selected = tuple(families)
    if not selected:
        raise ValueError("need at least one category")
    unknown = [name for name in selected if name not in CATEGORY_LABELS]
    if unknown:
        raise ValueError("unknown paper category: " + ", ".join(unknown))
    _require_recent_file(control_root, recent=False)
    payload = _require_recent_file(treatment_root, recent=True)
    control_ids = _persona_ids(control_root)
    treatment_ids = _persona_ids(treatment_root)
    if control_ids != treatment_ids:
        raise ValueError("control and treatment person ids differ")
    if n_people is not None and len(control_ids) != n_people:
        raise ValueError(f"expected {n_people} people, found {len(control_ids)}")
    kind_of = _recent_kinds(payload, control_ids)
    labels = tuple(str(kind) for kind in payload["kinds"])
    family_rows = []
    for family in selected:
        control_items, control_baseline = _load_family(control_root, family)
        treatment_items, treatment_baseline = _load_family(treatment_root, family)
        _require_same_items(family, control_items, treatment_items)
        _require_same_baseline(family, control_baseline, treatment_baseline)
        item_ids = [str(item["item_id"]) for item in control_items]
        _check_arm(
            control_root,
            model,
            family,
            item_ids=item_ids,
            persona_ids=control_ids,
            temperature=temperature,
            seed=seed,
        )
        _check_arm(
            treatment_root,
            model,
            family,
            item_ids=item_ids,
            persona_ids=control_ids,
            temperature=temperature,
            seed=seed,
        )
        control_map = _value_map(_responses(control_root, model, family), item_ids)
        treatment_map = _value_map(_responses(treatment_root, model, family), item_ids)
        _require_same_people(family, control_map, treatment_map, control_ids)
        family_rows.append(
            _family_row(
                family,
                control_items,
                control_baseline,
                control_map,
                treatment_map,
                control_ids,
                kind_of,
                labels,
            )
        )
    counts = {label: sum(kind == label for kind in kind_of.values()) for label in labels}
    return {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "n_people": len(control_ids),
        "control": str(control_root),
        "treatment": str(treatment_root),
        "contrast": "recent minus plain",
        "recent": {
            "draw": payload["draw"],
            "kinds": list(labels),
            "counts": counts,
        },
        "families": family_rows,
    }


def render_t0_economy_recent(report: Mapping[str, Any]) -> str:
    """Chinese report. Differences are the recent arm minus the plain arm."""
    counts = report["recent"]["counts"]
    mix = "、".join(f"{kind} {counts[kind]} 人" for kind in report["recent"]["kinds"])
    labels = "、".join(row["label"] for row in report["families"])
    lines = [
        "# 温度 0 下 Recent 对定价与经济偏好的影响",
        "",
        f"- 模型：`{report['model']}`。温度 {_fmt_temperature(report['temperature'])}，写入每次请求。该模型仍先推理。",
        f"- 人：ATUS 成年人 {report['n_people']} 人，seed {report['seed']}，按 `TUFINLWGT` 抽取。两组是同一批人。",
        f"- 题：{labels}。每一类单独一段对话。",
        "- 指标：选择题的平均 TVD 和平均 MSE。两者都对照 Twin-2K 前三波的选项份额，题目不加权。",
        "- MSE 是一道题各选项份额差的平方再平均。数字题不进入这两个平均。",
        "- 差：有事件减无事件。差为负，表示有事件那一组更接近 Twin-2K。",
        f"- 事件：{mix}。抽取 `{report['recent']['draw']}`。年龄段只改事发地点。",
        "- 这是一个种子上的一次抽取。下面给出差的大小，不做跨种子的显著性检验。",
        "",
        f"- 无事件：`{report['control']}`",
        f"- 有事件：`{report['treatment']}`",
        "",
    ]
    if any(row["family"] == "experiments" for row in report["families"]):
        lines.append("经济偏好和定价是这一次要看的两类。行为经济学实验也答了，所以一并列出。")
    else:
        lines.append("经济偏好和定价是这一次要看的两类。")
    lines.append("")
    for family in FOCUS_FAMILIES:
        row = next((item for item in report["families"] if item["family"] == family), None)
        if row is None:
            continue
        lines.append(_focus_line(row))
    lines.append("")
    lines.append("按事件种类，有事件减无事件：")
    lines.append("")
    for row in report["families"]:
        if row["family"] not in FOCUS_FAMILIES:
            continue
        tvd_parts = []
        mse_parts = []
        for kind_row in row["by_kind"]:
            tvd_parts.append(f"{kind_row['kind']} {_fmt(kind_row['distances']['mean_tvd_diff'])}")
            mse_parts.append(f"{kind_row['kind']} {_fmt(kind_row['distances']['mean_mse_diff'])}")
        lines.append(f"- {row['label']} TVD：" + "，".join(tvd_parts) + "。")
        lines.append(f"- {row['label']} MSE：" + "，".join(mse_parts) + "。")
    lines.append("")
    lines.extend(["## TVD 与 MSE", "", _distance_table(report["families"]), ""])
    lines.extend(["## 事件之间的差", "", _kind_diff_table(report), ""])
    lines.extend(["## 按事件种类", ""])
    lines.append("每一类只用抽到该事件的人。对照是这些人自己的无事件答案。")
    lines.append("")
    for kind in report["recent"]["kinds"]:
        rows = []
        for family in report["families"]:
            kind_row = next(item for item in family["by_kind"] if item["kind"] == kind)
            rows.append(
                {
                    **kind_row,
                    "label": family["label"],
                    "n_items": family["n_items"],
                    "n_choice": family["n_choice"],
                    "n_numeric": family["n_numeric"],
                }
            )
        n_kind = counts[kind]
        lines.extend([f"### {kind}（{n_kind} 人）", "", _distance_table(rows), ""])
    lines.extend(
        [
            "## 无效回答",
            "",
        ]
    )
    for family in report["families"]:
        control = family["distances"]["control"]
        treatment = family["distances"]["treatment"]
        lines.append(
            f"- {family['label']}：无事件 {control['n_invalid']} / {control['n_answers']}，"
            f"有事件 {treatment['n_invalid']} / {treatment['n_answers']}。"
        )
    lines.append("")
    return "\n".join(lines)


def write_t0_economy_recent(report: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Write summary.md and summary.json. Does not call a model."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "summary.json"
    markdown_path = target / "summary.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_t0_economy_recent(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _family_row(
    family: str,
    items: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    control_map: Mapping[str, Mapping[str, float | None]],
    treatment_map: Mapping[str, Mapping[str, float | None]],
    persona_ids: Sequence[str],
    kind_of: Mapping[str, str],
    labels: Sequence[str],
) -> dict[str, Any]:
    n_choice = sum(item["kind"] == "choice" for item in items)
    n_numeric = sum(item["kind"] == "numeric" for item in items)
    distances, paired = _contrast(items, baseline, control_map, treatment_map, persona_ids)
    by_kind = []
    for label in labels:
        group = [person_id for person_id in persona_ids if kind_of[person_id] == label]
        kind_distances, kind_paired = _contrast(items, baseline, control_map, treatment_map, group)
        by_kind.append(
            {
                "kind": label,
                "n_people": len(group),
                "distances": kind_distances,
                "paired": kind_paired,
            }
        )
    return {
        "family": family,
        "label": CATEGORY_LABELS[family],
        "n_items": len(items),
        "n_choice": n_choice,
        "n_numeric": n_numeric,
        "distances": distances,
        "paired": paired,
        "by_kind": by_kind,
    }


def _contrast(
    items: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    control_map: Mapping[str, Mapping[str, float | None]],
    treatment_map: Mapping[str, Mapping[str, float | None]],
    person_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    people = list(person_ids)
    control_rows = _responses_from_map(control_map, people)
    treatment_rows = _responses_from_map(treatment_map, people)
    control_distance = _distance(control_rows, items, baseline)
    treatment_distance = _distance(treatment_rows, items, baseline)
    distances = {
        "control": control_distance,
        "treatment": treatment_distance,
        "mean_tvd_diff": _subtract(treatment_distance["mean_tvd"], control_distance["mean_tvd"]),
        "mean_mse_diff": _subtract(treatment_distance["mean_mse"], control_distance["mean_mse"]),
        "mean_mae_diff": _subtract(treatment_distance["mean_mae"], control_distance["mean_mae"]),
    }
    return distances, _paired(items, control_map, treatment_map, people)


def _distance(
    responses: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not responses:
        return {
            "mean_tvd": None,
            "mean_mse": None,
            "mean_mae": None,
            "n_choice_scored": 0,
            "n_numeric_scored": 0,
            "n_invalid": 0,
            "n_answers": 0,
            "n_people": 0,
        }
    summary = score_full(list(responses), dict(baseline), list(items))
    choice = [row["tvd"] for row in summary["items"] if row.get("tvd") is not None]
    numeric = [row["mae"] for row in summary["items"] if row.get("mae") is not None]
    return {
        "mean_tvd": _mean(choice),
        "mean_mse": _mean(_share_mses(responses, items, baseline)),
        "mean_mae": _mean(numeric),
        "n_choice_scored": len(choice),
        "n_numeric_scored": len(numeric),
        "n_invalid": summary["n_invalid"],
        "n_answers": summary["n_answers"],
        "n_people": summary["n_people"],
    }


def _paired(
    items: Sequence[Mapping[str, Any]],
    control_map: Mapping[str, Mapping[str, float | None]],
    treatment_map: Mapping[str, Mapping[str, float | None]],
    person_ids: Sequence[str],
) -> dict[str, Any]:
    choice_means: list[float] = []
    numeric_means: list[float] = []
    n_changed = 0
    n_comparable = 0
    for item in items:
        item_id = str(item["item_id"])
        diffs: list[float] = []
        for person_id in person_ids:
            left = control_map[person_id][item_id]
            right = treatment_map[person_id][item_id]
            if left is None or right is None:
                continue
            n_comparable += 1
            diff = float(right) - float(left)
            diffs.append(diff)
            if diff != 0:
                n_changed += 1
        item_mean = _mean(diffs)
        if item_mean is None:
            continue
        if item["kind"] == "choice":
            choice_means.append(item_mean)
        elif item["kind"] == "numeric":
            numeric_means.append(item_mean)
    return {
        "change_rate": (n_changed / n_comparable) if n_comparable else None,
        "n_changed": n_changed,
        "n_comparable": n_comparable,
        "choice_mean_diff": _mean(choice_means),
        "n_choice_paired": len(choice_means),
        "numeric_mean_diff": _mean(numeric_means),
        "n_numeric_paired": len(numeric_means),
    }


def _responses_from_map(
    coded: Mapping[str, Mapping[str, float | None]],
    person_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for person_id in person_ids:
        answers = []
        for item_id, value in coded[person_id].items():
            answers.append(
                {
                    "item_id": item_id,
                    "value": value,
                    "ok": value is not None,
                }
            )
        rows.append({"id": person_id, "answers": answers})
    return rows


def _require_recent_file(run_root: Path, *, recent: bool) -> dict[str, Any]:
    path = run_root / "recent.json"
    if not recent:
        if path.exists():
            raise ValueError(f"control arm should not have {path}")
        return {}
    if not path.is_file():
        raise ValueError(f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("draw") not in ACCEPTED_DRAWS:
        raise ValueError(f"{path} draw is {payload.get('draw')!r}")
    kinds = payload.get("kinds")
    if not isinstance(kinds, list) or not kinds:
        raise ValueError(f"{path} kinds are {kinds!r}")
    labels = [str(kind) for kind in kinds]
    if any(not label.strip() for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"{path} kinds are {kinds!r}")
    return payload


def _recent_kinds(payload: Mapping[str, Any], persona_ids: Sequence[str]) -> dict[str, str]:
    kind_of: dict[str, str] = {}
    for row in payload["people"]:
        person_id = str(row["id"])
        if person_id in kind_of:
            raise ValueError(f"recent.json repeats {person_id}")
        kind = str(row["kind"])
        allowed = {str(item) for item in payload["kinds"]}
        if kind not in allowed:
            raise ValueError(f"{person_id} recent kind is {kind!r}")
        kind_of[person_id] = kind
    if set(kind_of) != set(persona_ids):
        raise ValueError("recent.json person ids differ from the answers")
    return kind_of


def _persona_ids(run_root: Path) -> list[str]:
    path = run_root / "personas.jsonl"
    if not path.is_file():
        raise ValueError(f"missing {path}")
    rows = read_jsonl(path)
    ids = [str(row["id"]) for row in rows]
    if not ids:
        raise ValueError(f"{path} is empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} repeats a person id")
    return ids


def _load_family(run_root: Path, family: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folder = run_root / "families" / family
    items_path = folder / "items.json"
    baseline_path = folder / "human_baseline.json"
    if not items_path.is_file():
        raise ValueError(f"missing {items_path}")
    if not baseline_path.is_file():
        raise ValueError(f"missing {baseline_path}")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    item_ids = [str(item["item_id"]) for item in items]
    baseline_ids = [str(row["item_id"]) for row in baseline["items"]]
    if baseline_ids != item_ids:
        raise ValueError(f"{family} baseline items differ from items.json")
    return items, baseline


def _require_same_items(
    family: str,
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> None:
    if [_item_identity(item) for item in control] != [_item_identity(item) for item in treatment]:
        raise ValueError(f"{family} control and treatment items differ")


def _item_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(item["item_id"]),
        str(item["kind"]),
        tuple(str(option) for option in item.get("options") or []),
    )


def _require_same_baseline(family: str, control: Mapping[str, Any], treatment: Mapping[str, Any]) -> None:
    left = {str(row["item_id"]): row for row in control["items"]}
    right = {str(row["item_id"]): row for row in treatment["items"]}
    if list(left) != list(right):
        raise ValueError(f"{family} human baselines list different items")
    for item_id, row in left.items():
        other = right[item_id]
        if row.get("kind") != other.get("kind"):
            raise ValueError(f"{family} {item_id} baseline kind differs")
        if row.get("kind") == "choice":
            shares = [float(value) for value in row["shares"]]
            other_shares = [float(value) for value in other["shares"]]
            if len(shares) != len(other_shares) or any(
                abs(a - b) > 1e-9 for a, b in zip(shares, other_shares, strict=True)
            ):
                raise ValueError(f"{family} {item_id} human shares differ")
        else:
            if row.get("mean") is None or other.get("mean") is None:
                if row.get("mean") != other.get("mean"):
                    raise ValueError(f"{family} {item_id} human mean differs")
            elif abs(float(row["mean"]) - float(other["mean"])) > 1e-6:
                raise ValueError(f"{family} {item_id} human mean differs")


def _check_arm(
    run_root: Path,
    model: str,
    family: str,
    *,
    item_ids: Sequence[str],
    persona_ids: Sequence[str],
    temperature: float,
    seed: int,
) -> None:
    meta_path = run_root / model / family / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != model:
        raise ValueError(f"{meta_path} model is {meta.get('model')!r}")
    if meta.get("family") != family:
        raise ValueError(f"{meta_path} family is {meta.get('family')!r}")
    if meta.get("seed") != seed:
        raise ValueError(f"{meta_path} seed is {meta.get('seed')}")
    if meta.get("n_people") != len(persona_ids):
        raise ValueError(f"{meta_path} n_people is {meta.get('n_people')}")
    if meta.get("n_turns") != len(item_ids):
        raise ValueError(f"{meta_path} n_turns is {meta.get('n_turns')}")
    if meta.get("batch") is True:
        raise ValueError(f"{meta_path} batch is set")
    request = meta.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"{meta_path} has no request")
    got = request.get("temperature")
    if got is None or abs(float(got) - float(temperature)) > 1e-9:
        raise ValueError(f"{meta_path} temperature is {got}")
    extra = request.get("extra_body") or {}
    if extra.get("reasoning_effort"):
        raise ValueError(f"{meta_path} sets reasoning_effort")
    response_path = run_root / model / family / "responses.jsonl"
    if not response_path.is_file():
        raise ValueError(f"missing {response_path}")


def _responses(run_root: Path, model: str, family: str) -> list[dict[str, Any]]:
    return read_jsonl(run_root / model / family / "responses.jsonl")


def _value_map(
    responses: Sequence[Mapping[str, Any]],
    item_ids: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    wanted = [str(item_id) for item_id in item_ids]
    coded: dict[str, dict[str, float | None]] = {}
    for person in responses:
        person_id = str(person["id"])
        if person_id in coded:
            raise ValueError(f"duplicate person id {person_id}")
        values: dict[str, float | None] = {}
        for row in person["answers"]:
            item_id = str(row["item_id"])
            if item_id in values:
                raise ValueError(f"{person_id} repeats {item_id}")
            values[item_id] = _stored_value(person_id, item_id, row)
        if list(values) != wanted:
            raise ValueError(f"{person_id} items differ from the run")
        coded[person_id] = values
    return coded


def _require_same_people(
    family: str,
    control_map: Mapping[str, Any],
    treatment_map: Mapping[str, Any],
    persona_ids: Sequence[str],
) -> None:
    if set(control_map) != set(treatment_map):
        raise ValueError(f"{family} control and treatment person ids differ")
    if set(control_map) != set(persona_ids) or len(control_map) != len(persona_ids):
        raise ValueError(f"{family} answer ids differ from personas.jsonl")


def _stored_value(person_id: str, item_id: str, row: Mapping[str, Any]) -> float | None:
    if not row.get("ok"):
        return None
    value = row.get("value")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{person_id} {item_id} value is {value!r}")
    return float(value)


def _share_mses(
    responses: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> list[float]:
    """Mean squared gap of option shares. Same items that receive a TVD."""
    by_item: dict[str, list[float]] = {str(item["item_id"]): [] for item in items}
    for person in responses:
        answered = {str(row["item_id"]): row for row in person["answers"]}
        for item in items:
            item_id = str(item["item_id"])
            row = answered.get(item_id)
            if row is None or not row.get("ok") or row.get("value") is None:
                continue
            by_item[item_id].append(float(row["value"]))
    gold = {str(row["item_id"]): row for row in baseline["items"]}
    scores: list[float] = []
    for item in items:
        if item.get("kind") != "choice":
            continue
        item_id = str(item["item_id"])
        values = by_item[item_id]
        human = gold[item_id]
        if not values or not human.get("n"):
            continue
        n_options = len(item["options"])
        raw = [sum(value == code for value in values) / len(values) for code in range(1, n_options + 1)]
        simulated = _proportions(raw)
        target = _proportions([float(part) for part in human["shares"]])
        if simulated is None or target is None:
            continue
        scores.append(mean_squared_error(simulated, target))
    return scores


def _proportions(values: Sequence[float]) -> list[float] | None:
    total = float(sum(values))
    if total <= 0:
        return None
    return [float(value) / total for value in values]


def _focus_line(row: Mapping[str, Any]) -> str:
    distances = row["distances"]
    return (
        f"{row['label']}：TVD {_fmt(distances['control']['mean_tvd'])} → "
        f"{_fmt(distances['treatment']['mean_tvd'])}（差 {_fmt(distances['mean_tvd_diff'])}），"
        f"MSE {_fmt(distances['control']['mean_mse'])} → "
        f"{_fmt(distances['treatment']['mean_mse'])}（差 {_fmt(distances['mean_mse_diff'])}）。"
    )


def _distance_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| 类别 | 题数 | 选择 / 数字 | 无事件 TVD | 有事件 TVD | TVD 差 | "
        "无事件 MSE | 有事件 MSE | MSE 差 |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, rule]
    for row in rows:
        distances = row["distances"]
        lines.append(
            "| {label} | {n_items} | {choice} / {numeric} | {tvd_c} | {tvd_t} | {tvd_d} | "
            "{mse_c} | {mse_t} | {mse_d} |".format(
                label=row["label"],
                n_items=row.get("n_items", "—"),
                choice=row.get("n_choice", "—"),
                numeric=row.get("n_numeric", "—"),
                tvd_c=_fmt(distances["control"]["mean_tvd"]),
                tvd_t=_fmt(distances["treatment"]["mean_tvd"]),
                tvd_d=_fmt(distances["mean_tvd_diff"]),
                mse_c=_fmt(distances["control"]["mean_mse"]),
                mse_t=_fmt(distances["treatment"]["mean_mse"]),
                mse_d=_fmt(distances["mean_mse_diff"]),
            )
        )
    return "\n".join(lines)


def _kind_diff_table(report: Mapping[str, Any]) -> str:
    kinds = list(report["recent"]["kinds"])
    header = "| 类别 | 指标 | " + " | ".join(kinds) + " |"
    rule = "| --- | --- | " + " | ".join("---:" for _ in kinds) + " |"
    lines = [header, rule]
    metrics = (
        ("TVD 差", lambda row: _fmt(row["distances"]["mean_tvd_diff"])),
        ("MSE 差", lambda row: _fmt(row["distances"]["mean_mse_diff"])),
    )
    for family in report["families"]:
        by_kind = {row["kind"]: row for row in family["by_kind"]}
        for name, render in metrics:
            cells = [render(by_kind[kind]) for kind in kinds]
            lines.append(f"| {family['label']} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _subtract(treatment: float | None, control: float | None) -> float | None:
    if treatment is None or control is None:
        return None
    return treatment - control


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _fmt_temperature(value: Any) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the temperature-0 recent contrast from saved category answers",
    )
    parser.add_argument("--control", type=Path, default=CONTROL_RUN)
    parser.add_argument("--treatment", type=Path, default=TREATMENT_RUN)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)
    report = score_t0_economy_recent(
        args.control,
        args.treatment,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        n_people=args.n,
    )
    paths = write_t0_economy_recent(report, args.out)
    print(paths["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
