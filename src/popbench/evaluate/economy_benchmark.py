"""Score one model on the fixed twin2K-economy-bechmark slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from popbench.dao.io import read_jsonl
from popbench.evaluate.categories import evaluate_category_run
from popbench.evaluate.survey import score_full

DATASET_LABEL = "twin2K-economy-bechmark"


def evaluate_economy_benchmark(
    root: Path,
    panels: list[dict[str, Any]],
    *,
    slug: str,
    seed: int,
    temperature: float | None,
    recent: Mapping[str, Any] | None = None,
    emphasis: str | None = None,
    splice_recent: bool | None = None,
    human_simulation: bool = False,
) -> dict[str, Any]:
    """Score each saved family run and write one report for the slice."""
    summaries = []
    item_rows: list[dict[str, Any]] = []
    for panel in panels:
        run_dir = Path(root) / slug / panel["family"]
        summary = evaluate_category_run(
            run_dir,
            panel["items"],
            panel["baseline"],
            family=panel["family"],
            seed=seed,
        )
        summaries.append(summary)
        responses = read_jsonl(run_dir / "responses.jsonl")
        scored = score_full(responses, panel["baseline"], panel["items"])
        item_rows.extend(scored["items"])
    report = _report(summaries, item_rows, seed=seed, temperature=temperature)
    if recent is not None:
        report["recent"] = _recent_block(recent)
    stressed = str(emphasis or "").strip()
    if stressed:
        report["emphasis"] = stressed
    if splice_recent is False:
        report["splice_recent"] = False
    if human_simulation:
        report["human_simulation"] = True
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    (target / "compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (target / "compare.md").write_text(render_economy_benchmark(report))
    report["out_dir"] = str(target)
    return report


def render_economy_benchmark(report: dict[str, Any]) -> str:
    """One model against the fixed slice. Lower TVD is closer to Twin-2K."""
    lines = [
        f"# {DATASET_LABEL}",
        "",
        f"- 模型：{report['model']}",
        f"- 温度：{_fmt_temperature(report.get('temperature'))}",
        f"- 种子：{report['seed']}",
    ]
    recent_line = _recent_line(report)
    if recent_line:
        lines.append(recent_line)
    elif report.get("splice_recent") is False:
        lines.append("- recent：不拼接。人物卡保持原样。")
    if report.get("emphasis"):
        lines.append(f"- 强调：{report['emphasis']}")
    if report.get("human_simulation"):
        lines.append(
            "- 类人模拟：skill/humanSimulation（倾向卡 + 调查 framing；"
            "定价 buy_prob→Bernoulli；其余选择题 soft probs→外层抽样）。"
        )
    lines.extend([
        f"- 人数：{report['n_people']}。ATUS 成年人，按年龄段和性别配额抽取。",
        "- 题：固定数据集的 74 个作答列。经济偏好、行为经济学实验、定价各开一段访谈。",
        "- 人类基线：这份数据集里 Twin-2K 前三波的选项份额。",
        "",
        "ATUS 受访者和 Twin-2K 受访者不是同一批人，所以比的是群体分布。",
        "",
        "## 选择题平均 TVD",
        "",
        "| 类别 | 题数 | 平均 TVD | 无效回答 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in report["families"]:
        lines.append(
            f"| {row['label']} | {row['n_items']} | {_fmt(row['mean_tvd'])} | "
            f"{row['n_invalid']} / {row['n_answers']} |"
        )
    lines.append(
        f"| 全部 | {report['n_items']} | {_fmt(report['mean_tvd'])} | "
        f"{report['n_invalid']} / {report['n_answers']} |"
    )
    lines.extend(["", "## 差得最大的选择题", ""])
    for row in report["farthest"]:
        prompt = " ".join(str(row["prompt"]).split())
        if len(prompt) > 110:
            prompt = prompt[:110] + "…"
        lines.append(
            f"- TVD {row['tvd']:.3f} · {row['label']} · {row['item_id']} · "
            f"模拟均值 {row['simulated_mean']:.2f} · 人类均值 {row['human_mean']:.2f} · {prompt}"
        )
    lines.append("")
    return "\n".join(lines)


def _report(
    summaries: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
    *,
    seed: int,
    temperature: float | None,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("economy benchmark evaluation has no families")
    choice = [row for row in item_rows if row.get("tvd") is not None]
    labels = {row["family"]: row["label"] for row in summaries}
    farthest = sorted(choice, key=lambda row: row["tvd"], reverse=True)[:8]
    return {
        "dataset": DATASET_LABEL,
        "model": summaries[0]["model"],
        "seed": seed,
        "temperature": temperature,
        "n_people": summaries[0]["n_people"],
        "n_items": len(item_rows),
        "n_invalid": sum(row["n_invalid"] for row in summaries),
        "n_answers": sum(row["n_answers"] for row in summaries),
        "mean_tvd": _mean([row["tvd"] for row in choice]),
        "families": [
            {
                "family": row["family"],
                "label": row["label"],
                "n_items": row["n_items"],
                "mean_tvd": row["mean_tvd"],
                "n_invalid": row["n_invalid"],
                "n_answers": row["n_answers"],
                "run_dir": row["run_dir"],
            }
            for row in summaries
        ],
        "farthest": [
            {
                "item_id": row["item_id"],
                "family": row["family"],
                "label": labels.get(row["family"], row["family"]),
                "tvd": row["tvd"],
                "simulated_mean": row["simulated_mean"],
                "human_mean": row["human_mean"],
                "prompt": row["prompt"],
            }
            for row in farthest
        ],
    }


def _recent_block(recent: Mapping[str, Any]) -> dict[str, Any]:
    kinds = [str(kind) for kind in recent["kinds"]]
    by_id = recent.get("by_id")
    if not isinstance(by_id, Mapping):
        raise ValueError("recent evaluation needs by_id")
    return {
        "draw": str(recent["draw"]),
        "n_flashes": len(kinds),
        "kinds": kinds,
        "by_id": {str(key): str(value) for key, value in by_id.items()},
    }


def _recent_line(report: Mapping[str, Any]) -> str | None:
    recent = report.get("recent")
    if not recent:
        return None
    count = recent.get("n_flashes")
    if count is None:
        count = len(recent.get("kinds") or [])
    return (
        "- recent：金十免费快讯全量 "
        f"{count} 条。每人在人物卡末尾拼接其中一条，按 `{recent.get('draw')}`。"
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_temperature(value: float | None) -> str:
    if value is None:
        return "未发送"
    return format(float(value), "g")
