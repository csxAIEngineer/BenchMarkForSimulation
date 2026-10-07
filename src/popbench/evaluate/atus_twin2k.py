"""Score ATUS-persona answers against Twin-2K option shares and rank models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from popbench.dao.interview import OPTIONS, SCALE_ZH
from popbench.dao.io import read_jsonl
from popbench.dao.schema import InterviewRecord, InterviewTurn
from popbench.evaluate.compare import CompareError
from popbench.evaluate.interview import score_simulation
from popbench.simulate.client import model_from_run


def evaluate_atus_twin2k(
    run_dir: Path,
    records: list[InterviewRecord],
    baseline: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score one model's saved answers. Writes summary.json, option shares, and report.md."""
    run_dir = Path(run_dir)
    response_path = run_dir / "responses.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(response_path)
    if not records:
        raise CompareError("the ATUS panel is empty")
    responses = read_jsonl(response_path)
    items = [turn.to_dict() for turn in records[0].turns]
    for item, turn in zip(items, records[0].turns):
        item["reverse"] = turn.reverse
        item["column"] = turn.item_id
    summary = score_simulation(responses, baseline, items)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    by_id = {turn.item_id: turn for turn in records[0].turns}
    for row in summary["items"]:
        turn = by_id[row["item_id"]]
        row["prompt"] = turn.prompt
        row["scale_zh"] = SCALE_ZH.get(row["scale"], row["scale"])
    summary["model"] = model_from_run(meta)
    summary["seed"] = seed if seed is not None else meta.get("seed")
    summary["panel"] = "atus"
    summary["instrument"] = records[0].instrument
    summary["baseline"] = "Twin-2K option shares"
    summary["request"] = meta.get("request")
    shares = _option_shares(responses, records[0].turns)
    (run_dir / "option_shares.json").write_text(json.dumps(shares, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "report.md").write_text(_model_report(summary))
    return summary


def write_atus_twin2k_comparison(
    out_dir: Path,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank models by Twin-2K total variation distance, then mean-code error."""
    if len(summaries) < 2:
        raise CompareError("need at least two model evaluations")
    rows = [_row(summary) for summary in summaries]
    models = [row["model"] for row in rows]
    if any(not model for model in models):
        raise CompareError("each run needs a model name in meta.json")
    if len(set(models)) != len(models):
        raise CompareError("comparison runs must use different models")
    ranked = sorted(rows, key=_rank_key)
    for place, row in enumerate(ranked, start=1):
        row["place"] = place
    report = {
        "kind": "atus_twin2k_comparison",
        "panel": "atus",
        "baseline": "Twin-2K option shares",
        "n_people": summaries[0].get("n_people"),
        "seed": summaries[0].get("seed"),
        "models": [row["model"] for row in ranked],
        "ranked": ranked,
        "per_item": _per_item(rows),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "compare.md").write_text(render_atus_twin2k_comparison(report))
    return report


def render_atus_twin2k_comparison(report: dict[str, Any]) -> str:
    """Markdown ranking. Lower total variation distance is closer to Twin-2K."""
    lines = [
        "# ATUS 人物卡回答 Twin-2K 选择题",
        "",
        "- 人：ATUS 2023 成年人。按年龄段和性别配额，用最终权重抽样，种子固定。",
        "- 人物卡：该受访者的人口学、本人日记分钟，以及按种子扩写的经历。",
        "- 题：大五人格每个维度 1 道正向题，加 1 道绿色消费题，共 6 道五级同意题。",
        "- 人类基线：Twin-2K 全样本在这 6 题上的选项份额。",
        "- 排名：总变异距离更小优先，其次是均分绝对误差更小。",
        "",
        "ATUS 受访者和 Twin-2K 受访者不是同一批人，所以这里比的是群体选项分布。",
        "",
        "## 排名",
        "",
        "| 名次 | 模型 | 平均 TVD | 均分绝对误差 | 无效回答 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["ranked"]:
        lines.append(
            f"| {row['place']} | `{row['model']}` | {_fmt(row['mean_tvd'])} | "
            f"{_fmt(row['mean_abs_code_error'])} | {row['n_invalid']} |"
        )
    lines.extend(
        [
            "",
            "## 逐题总变异距离",
            "",
            "| 题 | 量表 | "
            + " | ".join(f"`{model}`" for model in report["models"])
            + " |",
            "| --- | --- | " + " | ".join("---:" for _ in report["models"]) + " |",
        ]
    )
    for item in report["per_item"]:
        cells = " | ".join(_fmt(item["by_model"].get(model)) for model in report["models"])
        lines.append(f"| `{item['item_id']}` | {item['scale_zh']} | {cells} |")
    lines.extend(["", "## 题目", ""])
    for item in report["per_item"]:
        lines.append(f"- `{item['item_id']}`（{item['scale_zh']}）：{item['prompt']}")
    lines.append("")
    return "\n".join(lines)


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": summary.get("model"),
        "mean_tvd": summary.get("mean_tvd"),
        "mean_abs_code_error": summary.get("mean_abs_code_error"),
        "n_invalid": summary.get("n_invalid"),
        "n_answers": summary.get("n_answers"),
        "items": summary.get("items") or [],
    }


def _rank_key(row: dict[str, Any]) -> tuple[bool, float, bool, float]:
    tvd = row["mean_tvd"]
    error = row["mean_abs_code_error"]
    return (
        tvd is None,
        float(tvd) if tvd is not None else 0.0,
        error is None,
        float(error) if error is not None else 0.0,
    )


def _per_item(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    meta: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, float | None]] = {}
    for row in rows:
        by_model[row["model"]] = {}
        for item in row["items"]:
            item_id = item["item_id"]
            if item_id not in meta:
                order.append(item_id)
                meta[item_id] = {
                    "item_id": item_id,
                    "scale_zh": item.get("scale_zh") or item.get("scale"),
                    "prompt": item.get("prompt") or "",
                }
            by_model[row["model"]][item_id] = item.get("tvd")
    return [{**meta[item_id], "by_model": {model: values.get(item_id) for model, values in by_model.items()}} for item_id in order]


def _option_shares(responses: list[dict[str, Any]], turns: tuple[InterviewTurn, ...]) -> dict[str, Any]:
    items = []
    for turn in turns:
        codes = []
        for person in responses:
            answered = {row["item_id"]: row for row in person.get("answers", [])}
            row = answered.get(turn.item_id)
            if row and row.get("ok"):
                codes.append(int(row["code"]))
        shares = [codes.count(code) / len(codes) if codes else 0.0 for code in range(1, 6)]
        items.append(
            {
                "item_id": turn.item_id,
                "turn": turn.turn,
                "shares": shares,
                "options": list(OPTIONS),
                "n": len(codes),
            }
        )
    return {"items": items}


def _model_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary.get('model')} 回答 Twin-2K 选择题",
        "",
        f"- 人数：{summary.get('n_people')}",
        f"- 平均总变异距离：{_fmt(summary.get('mean_tvd'))}",
        f"- 均分绝对误差：{_fmt(summary.get('mean_abs_code_error'))}",
        f"- 无效回答：{summary.get('n_invalid')} / {summary.get('n_answers')}",
        "",
        "参照是 Twin-2K 全样本在同一题上的选项份额。",
        "",
        "| 题 | 量表 | TVD | 模拟均分 | 人类均分 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in summary["items"]:
        lines.append(
            f"| `{item['item_id']}` | {item.get('scale_zh', item['scale'])} | "
            f"{_fmt(item.get('tvd'))} | {_fmt(item.get('simulated_mean_code'))} | "
            f"{_fmt(item.get('human_mean_code'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
