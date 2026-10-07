"""Rank visitor simulate runs from different models on the same ATUS baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CompareError(RuntimeError):
    """A model comparison could not be assembled."""


def build_visitor_comparison(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank models by distribution correlation, then by share MSE."""
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
    return {
        "kind": "visitor_model_comparison",
        "panel": "visitors",
        "baseline": summaries[0].get("baseline"),
        "n_people": summaries[0].get("n_people"),
        "seed": summaries[0].get("seed"),
        "models": [row["model"] for row in ranked],
        "ranked": ranked,
        "per_item": _per_item(rows),
    }


def write_visitor_comparison(out_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Write `compare.json` and `compare.md` for one shared visitor panel."""
    report = build_visitor_comparison(summaries)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "compare.md").write_text(render_visitor_comparison(report))
    report["out_dir"] = str(target)
    return report


def render_visitor_comparison(report: dict[str, Any]) -> str:
    """Markdown leaderboard plus per-item correlations."""
    lines = [
        "# 模型 Simulate 对比",
        "",
        "同一套访问者数据集、同一套题目、同一套 ATUS 人类基线。",
        "分布相关越高、份额 MSE 越低，模拟答案分布越接近人类作息。",
        "",
        f"- 人数：{report.get('n_people')}",
        f"- 种子：{report.get('seed')}",
        "",
        "| 名次 | 模型 | 分布相关 | 份额 MSE | 均分 MSE | 均分相关 | 幅度覆盖 | 幅度比 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["ranked"]:
        lines.append(
            "| "
            f"{row['place']} | `{row['model']}` | "
            f"{_fmt(row['distribution_correlation'])} | "
            f"{_fmt(row['share_mse'], 4)} | "
            f"{_fmt(row['mean_code_mse'], 4)} | "
            f"{_fmt(row['mean_code_correlation'])} | "
            f"{_fmt(row['response_amplitude_coverage'])} | "
            f"{_fmt(row['response_amplitude_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## 逐题分布相关",
            "",
            _item_table(report),
            "",
            "## 请求设置",
            "",
            "题目和人物卡相同。各模型用该接口能接受的直接作答参数：",
            "",
        ]
    )
    for row in report["ranked"]:
        lines.append(f"- `{row['model']}`：{_policy(row.get('request'))}")
    lines.append("")
    return "\n".join(lines)


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    fit = summary.get("behavioral_fit")
    if not isinstance(fit, dict):
        raise CompareError(f"missing behavioral fit for {summary.get('model')}")
    return {
        "model": summary.get("model"),
        "run_dir": summary.get("run_dir"),
        "distribution_correlation": fit.get("distribution_correlation"),
        "share_mse": fit.get("share_mse"),
        "mean_code_mse": fit.get("mean_code_mse"),
        "mean_code_correlation": fit.get("mean_code_correlation"),
        "response_amplitude_coverage": fit.get("response_amplitude_coverage"),
        "response_amplitude_ratio": fit.get("response_amplitude_ratio"),
        "per_item": fit.get("per_item") or [],
        "request": summary.get("request"),
    }


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    correlation = row["distribution_correlation"]
    share_mse = row["share_mse"]
    return (
        correlation is None,
        -(correlation or 0.0),
        share_mse is None,
        share_mse if share_mse is not None else 0.0,
        str(row["model"]),
    )


def _per_item(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    for row in rows:
        for item in row["per_item"]:
            item_id = str(item["item_id"])
            if item_id not in order:
                order.append(item_id)
    table = []
    for item_id in order:
        cells = []
        for row in rows:
            match = next(
                (item for item in row["per_item"] if item["item_id"] == item_id),
                None,
            )
            cells.append(
                {
                    "model": row["model"],
                    "distribution_correlation": None
                    if match is None
                    else match.get("distribution_correlation"),
                }
            )
        table.append({"item_id": item_id, "models": cells})
    return table


def _item_table(report: dict[str, Any]) -> str:
    models = [row["model"] for row in report["ranked"]]
    header = "| 题 | " + " | ".join(f"`{model}`" for model in models) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in models) + " |"
    lines = [header, rule]
    by_model = {row["model"]: row for row in report["ranked"]}
    for item in report["per_item"]:
        values = []
        for model in models:
            found = next(
                (
                    cell
                    for cell in by_model[model]["per_item"]
                    if cell["item_id"] == item["item_id"]
                ),
                None,
            )
            correlation = None if found is None else found.get("distribution_correlation")
            values.append(_fmt(correlation))
        lines.append("| `" + item["item_id"] + "` | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _policy(request: dict[str, Any] | None) -> str:
    if not isinstance(request, dict):
        return "未记录"
    if request.get("transport") == "cr":
        return "cr_api.CRClient，只传 model 和 messages"
    extra = request.get("extra_body") or {}
    parts = []
    if "thinking" in extra:
        thinking = extra["thinking"]
        kind = thinking.get("type") if isinstance(thinking, dict) else thinking
        parts.append(f"thinking {kind}")
    if "reasoning_effort" in extra:
        parts.append(f"reasoning_effort {extra['reasoning_effort']}")
    temperature = request.get("temperature")
    if temperature is None:
        parts.append("temperature 使用接口默认值")
    else:
        parts.append(f"temperature {temperature}")
    if request.get("max_tokens") is not None:
        parts.append(f"{request.get('token_field') or 'max_tokens'} {request['max_tokens']}")
    return "，".join(parts) if parts else "未记录"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)
