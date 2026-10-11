"""Score one ATUS category run and compare models without pooling the seven categories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from popbench.dao.io import read_jsonl
from popbench.dao.survey import CATEGORY_LABELS, PAPER_FAMILIES
from popbench.evaluate.compare import CompareError
from popbench.evaluate.survey import score_full


def evaluate_category_run(
    run_dir: Path,
    items: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    family: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score saved answers for one model on one paper category."""
    run_dir = Path(run_dir)
    response_path = run_dir / "responses.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(response_path)
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise CompareError(f"missing model name in {meta_path}")
    meta = json.loads(meta_path.read_text())
    model = meta.get("model")
    if not isinstance(model, str) or not model.strip():
        raise CompareError(f"missing model name in {meta_path}")
    if family not in CATEGORY_LABELS:
        raise CompareError(f"unknown paper category {family!r}")
    responses = read_jsonl(response_path)
    summary = score_full(responses, baseline, items)
    choice = [row for row in summary["items"] if row.get("tvd") is not None]
    numeric = [row for row in summary["items"] if row.get("mae") is not None]
    result = {
        "family": family,
        "label": CATEGORY_LABELS[family],
        "model": model.strip(),
        "run_dir": str(run_dir),
        "n_people": summary["n_people"],
        "n_items": summary["n_items"],
        "n_choice": sum(row["kind"] == "choice" for row in summary["items"]),
        "n_numeric": sum(row["kind"] == "numeric" for row in summary["items"]),
        "n_invalid": summary["n_invalid"],
        "n_answers": summary["n_answers"],
        "mean_tvd": _mean([row["tvd"] for row in choice]),
        "mean_mae": _mean([row["mae"] for row in numeric]),
        "seed": seed if seed is not None else meta.get("seed"),
        "request": meta.get("request"),
    }
    (run_dir / "summary.json").write_text(json.dumps(result) + "\n")
    (run_dir / "report.md").write_text(render_category_report(result))
    return result


def write_category_comparison(out_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Write the category-by-model table. Categories stay on separate rows."""
    report = build_category_comparison(summaries)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "compare.md").write_text(render_category_comparison(report))
    report["out_dir"] = str(target)
    return report


def build_category_comparison(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank models inside each category. There is no item-weighted total."""
    if len(summaries) < 2:
        raise CompareError("need at least two model evaluations")
    models = []
    for row in summaries:
        model = row.get("model")
        if not model:
            raise CompareError("each category run needs a model name")
        if model not in models:
            models.append(model)
    if len(models) < 2:
        raise CompareError("need at least two model evaluations")
    present = {row["family"] for row in summaries}
    unknown = sorted(present - set(PAPER_FAMILIES))
    if unknown:
        raise CompareError("unknown paper category: " + ", ".join(unknown))
    order = [family for family in PAPER_FAMILIES if family in present]
    categories = [_category(family, summaries, models) for family in order]
    return {
        "kind": "atus_category_comparison",
        "panel": "atus",
        "baseline": "Twin-2K wave1_3 shares",
        "n_people": _shared(summaries, "n_people"),
        "seed": _shared(summaries, "seed"),
        "models": models,
        "categories": categories,
    }


def render_category_comparison(report: dict[str, Any]) -> str:
    """Markdown table of mean TVD by category. Lower TVD is closer to Twin-2K."""
    lines = [
        "# ATUS 人物卡按论文类别回答 Twin-2K",
        "",
        "- 人：ATUS 2023 成年人。同一人数和种子，七类看到的是同一批人物卡。",
        "- 题：每一类单独一段对话，类别之间不共享已答内容。",
        "- 人类基线：该类在 Twin-2K 前三波的选项份额和数字均值。",
        "- 格子里是选择题平均总变异距离，括号里是该类名次。距离更小更接近 Twin-2K。",
        "",
        "ATUS 受访者和 Twin-2K 受访者不是同一批人，所以这里比的是群体分布。七类分开排名，不按题量合成一个总分。",
        "",
        f"- 人数：{report.get('n_people')}",
        f"- 种子：{report.get('seed')}",
        "",
        "## 选择题平均 TVD",
        "",
        _tvd_table(report),
        "",
    ]
    mae = _mae_table(report)
    if mae:
        lines.extend(["## 数字题平均绝对差", "", mae, ""])
    lines.extend(["## 类内名次", ""])
    for category in report["categories"]:
        ranking = "，".join(
            f"{row['place']}. `{row['model']}`" for row in category["ranked"]
        )
        lines.append(f"- {category['label']}：{ranking}")
    lines.append("")
    return "\n".join(lines)


def render_category_report(summary: dict[str, Any]) -> str:
    """One model, one category."""
    lines = [
        f"# {summary['label']}",
        "",
        f"- 模型：{summary['model']}",
        f"- 人数：{summary['n_people']}",
        f"- 闭卷题：{summary['n_items']}（选择 {summary['n_choice']} / 数字 {summary['n_numeric']}）",
        f"- 选择题平均 TVD：{_fmt(summary['mean_tvd'])}",
        f"- 数字题平均绝对差：{_fmt(summary['mean_mae'])}",
        f"- 无效回答：{summary['n_invalid']} / {summary['n_answers']}",
        "",
        "人物卡是 ATUS 受访者。分数对照的是 Twin-2K 前三波在这些题上的人群分布。",
        "",
    ]
    return "\n".join(lines)


def _category(family: str, summaries: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    rows = [row for row in summaries if row["family"] == family]
    got = [row["model"] for row in rows]
    if len(got) != len(set(got)):
        raise CompareError(f"{family} has more than one run for the same model")
    missing = [model for model in models if model not in got]
    if missing:
        raise CompareError(f"{family} is missing " + ", ".join(missing))
    counts = {row["n_items"] for row in rows}
    if len(counts) != 1:
        raise CompareError(f"{family} runs do not ask the same number of items")
    ranked = sorted(rows, key=_rank_key)
    placed = []
    for place, row in enumerate(ranked, start=1):
        placed.append(
            {
                "place": place,
                "model": row["model"],
                "mean_tvd": row.get("mean_tvd"),
                "mean_mae": row.get("mean_mae"),
                "n_invalid": row.get("n_invalid"),
            }
        )
    return {
        "family": family,
        "label": CATEGORY_LABELS[family],
        "n_items": rows[0]["n_items"],
        "n_choice": rows[0]["n_choice"],
        "n_numeric": rows[0]["n_numeric"],
        "ranked": placed,
    }


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    tvd = row.get("mean_tvd")
    mae = row.get("mean_mae")
    return (
        tvd is None,
        tvd if tvd is not None else 0.0,
        mae is None,
        mae if mae is not None else 0.0,
        str(row.get("model")),
    )


def _shared(summaries: list[dict[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in summaries}
    if len(values) != 1:
        raise CompareError(f"{key} differs across category runs")
    return summaries[0].get(key)


def _tvd_table(report: dict[str, Any]) -> str:
    models = report["models"]
    header = "| 类别 | 题数 | 选择 / 数字 | " + " | ".join(f"`{model}`" for model in models) + " |"
    rule = "| --- | ---: | ---: | " + " | ".join("---:" for _ in models) + " |"
    lines = [header, rule]
    for category in report["categories"]:
        by_model = {row["model"]: row for row in category["ranked"]}
        cells = []
        for model in models:
            row = by_model[model]
            cells.append(f"{_fmt(row['mean_tvd'])}（{row['place']}）")
        lines.append(
            f"| {category['label']} | {category['n_items']} | "
            f"{category['n_choice']} / {category['n_numeric']} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def _mae_table(report: dict[str, Any]) -> str:
    numeric = [category for category in report["categories"] if category["n_numeric"]]
    if not numeric:
        return ""
    models = report["models"]
    header = "| 类别 | " + " | ".join(f"`{model}`" for model in models) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in models) + " |"
    lines = [header, rule]
    for category in numeric:
        by_model = {row["model"]: row for row in category["ranked"]}
        cells = [_fmt(by_model[model]["mean_mae"]) for model in models]
        lines.append(f"| {category['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
