"""Score the longer closed-ended survey against Twin-2K response shares."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from popbench.dao.io import read_jsonl
from popbench.dao.survey import ASK_N, survey_dir
from popbench.evaluate.metrics import jensen_shannon, total_variation_distance
from popbench.simulate.client import model_from_run

FAMILY_ZH = {
    "demographics": "人口学",
    "personality": "人格与价值观",
    "cognitive": "认知",
    "economic": "经济偏好",
    "heuristics": "启发式",
    "experiments": "行为实验",
    "pricing": "购买",
}


def score_full(
    responses: list[dict[str, Any]],
    baseline: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    by_item: dict[str, list[Any]] = {item["item_id"]: [] for item in items}
    n_invalid = 0
    n_answers = 0
    for person in responses:
        answered = {row["item_id"]: row for row in person["answers"]}
        for item in items:
            n_answers += 1
            row = answered.get(item["item_id"])
            if row is None or not row.get("ok"):
                n_invalid += 1
                continue
            by_item[item["item_id"]].append(row["value"])
    human = {row["item_id"]: row for row in baseline["items"]}
    item_rows = []
    for item in items:
        values = by_item[item["item_id"]]
        gold = human[item["item_id"]]
        if item["kind"] == "choice" and values and gold["n"]:
            n_options = len(item["options"])
            shares = [values.count(code) / len(values) for code in range(1, n_options + 1)]
            tvd = total_variation_distance(shares, gold["shares"])
            js = jensen_shannon(shares, gold["shares"])
            mean_code = sum(values) / len(values)
            item_rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "choice",
                    "family": item["family"],
                    "block": item["block"],
                    "prompt": item["prompt"],
                    "n": len(values),
                    "human_n": gold["n"],
                    "tvd": tvd,
                    "js": js,
                    "simulated_mean": mean_code,
                    "human_mean": gold["mean_code"],
                }
            )
        elif item["kind"] == "numeric" and values and gold["mean"] is not None:
            simulated_mean = sum(values) / len(values)
            item_rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "numeric",
                    "family": item["family"],
                    "block": item["block"],
                    "prompt": item["prompt"],
                    "n": len(values),
                    "human_n": gold["n"],
                    "mae": abs(simulated_mean - gold["mean"]),
                    "simulated_mean": simulated_mean,
                    "human_mean": gold["mean"],
                }
            )
        else:
            item_rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": item["kind"],
                    "family": item["family"],
                    "block": item["block"],
                    "prompt": item["prompt"],
                    "n": len(values),
                    "human_n": gold["n"],
                }
            )
    return {
        "n_people": len(responses),
        "n_items": len(items),
        "n_answers": n_answers,
        "n_invalid": n_invalid,
        "items": item_rows,
        "families": _family_summary(item_rows),
    }


def evaluate_survey(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    """Score a saved closed-ended run. Items follow the answers that were stored."""
    run_dir = Path(run_dir)
    responses = read_jsonl(run_dir / "responses.jsonl")
    root = survey_dir(data_dir)
    panel = read_jsonl(root / "nemotron_panel.jsonl")
    baseline = json.loads((root / "human_baseline.json").read_text())
    manifest_path = root / "manifest.json"
    n_closed = None
    if manifest_path.is_file():
        n_closed = json.loads(manifest_path.read_text()).get("n_closed")
    items = _items_from_responses(responses, panel)
    summary = score_full(responses, baseline, items)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    summary["model"] = model_from_run(meta)
    if n_closed is not None:
        summary["n_closed"] = n_closed
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    (run_dir / "report.md").write_text(render_full_markdown(summary, {"model": summary["model"]}))
    return summary


def _items_from_responses(
    responses: list[dict[str, Any]],
    panel: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {turn["item_id"]: turn for turn in panel[0]["turns"]}
    items = []
    for index, answer in enumerate(responses[0]["answers"], start=1):
        turn = dict(by_id[answer["item_id"]])
        turn["turn"] = index
        items.append(turn)
    return items


def _family_summary(item_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        grouped[row["family"]].append(row)
    summary = {}
    for family, rows in grouped.items():
        tvds = [row["tvd"] for row in rows if row.get("tvd") is not None]
        jss = [row["js"] for row in rows if row.get("js") is not None]
        maes = [row["mae"] for row in rows if row.get("mae") is not None]
        summary[family] = {
            "n_items": len(rows),
            "n_choice": sum(row["kind"] == "choice" for row in rows),
            "n_numeric": sum(row["kind"] == "numeric" for row in rows),
            "mean_tvd": sum(tvds) / len(tvds) if tvds else None,
            "mean_js": sum(jss) / len(jss) if jss else None,
            "mean_mae": sum(maes) / len(maes) if maes else None,
        }
    return summary


def render_full_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    choice = [row for row in summary["items"] if row.get("tvd") is not None]
    mean_tvd = sum(row["tvd"] for row in choice) / len(choice) if choice else None
    lines = [
        "# Nemotron 回答 Twin-2K 闭口题",
        "",
        f"- 模型：{meta.get('model') or summary.get('model')}",
        f"- 模拟人数：{summary['n_people']}",
        f"- 题目：{summary['n_items']} 道（闭口题共 {summary.get('n_closed', summary['n_items'])} 道，按问卷顺序答到第 {ASK_N} 道，并答完该题所在矩阵）",
        f"- 无效回答：{summary['n_invalid']} / {summary['n_answers']}",
        f"- 选择题平均 TVD：{_fmt(mean_tvd)}",
        "",
        "Nemotron 人物不是 Twin-2K 受访者。选择题比的是选项份额，数字题比的是均值。",
        "",
        "## 分块",
        "",
        "| 分块 | 题数 | 选择 / 数字 | 平均 TVD | 数字题平均绝对差 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family, row in summary["families"].items():
        lines.append(
            "| {name} | {n} | {choice} / {numeric} | {tvd} | {mae} |".format(
                name=FAMILY_ZH.get(family, family),
                n=row["n_items"],
                choice=row["n_choice"],
                numeric=row["n_numeric"],
                tvd=_fmt(row["mean_tvd"]),
                mae=_fmt(row["mean_mae"]),
            )
        )
    lines.extend(["", "## 差得最大的选择题", ""])
    ranked = sorted(choice, key=lambda row: row["tvd"], reverse=True)[:12]
    for row in ranked:
        prompt = " ".join(row["prompt"].split())
        if len(prompt) > 110:
            prompt = prompt[:110] + "…"
        lines.append(
            f"- TVD {row['tvd']:.3f} · {FAMILY_ZH.get(row['family'], row['family'])} · {row['item_id']} · {prompt}"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
