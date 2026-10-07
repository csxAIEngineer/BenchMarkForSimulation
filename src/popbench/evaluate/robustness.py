"""Cross-model robustness audit for simulate runs.

Compares aligned option shares / mean codes across multiple model run
directories and reports cross-model variance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from popbench.evaluate.metrics import cross_model_variance
from popbench.simulate.visitors import VISIT_TURNS


class RobustnessError(RuntimeError):
    """Robustness audit could not load or align runs."""


def audit_cross_model_variance(
    run_dirs: list[Path],
    *,
    out_dir: Path | None = None,
    item_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Audit robustness across simulate runs from different models."""
    if len(run_dirs) < 2:
        raise RobustnessError("need at least two run directories")
    runs = [_load_run(Path(path)) for path in run_dirs]
    visitor_default = item_ids is None
    item_ids = [turn.item_id for turn in VISIT_TURNS] if visitor_default else list(item_ids)

    # Per-item mean codes across models.
    mean_code_series = [_mean_codes(run, item_ids) for run in runs]
    # Flattened share vectors (item-major, then option).
    share_series = [_flat_shares(run, item_ids) for run in runs]

    mean_code_stats = cross_model_variance(mean_code_series)
    share_stats = cross_model_variance(share_series)

    per_item = []
    for index, item_id in enumerate(item_ids):
        values = [series[index] for series in mean_code_series]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        per_item.append(
            {
                "item_id": item_id,
                "models": [
                    {"model": run["model"], "mean_code": values[i]}
                    for i, run in enumerate(runs)
                ],
                "cross_model_variance": var,
                "cross_model_std": var**0.5,
            }
        )

    summary = {
        "kind": "cross_model_robustness",
        "n_runs": len(runs),
        "models": [run["model"] for run in runs],
        "run_dirs": [str(run["run_dir"]) for run in runs],
        "mean_code_variance": mean_code_stats,
        "share_vector_variance": share_stats,
        "per_item": per_item,
        "interpretation": (
            "Lower mean_variance / mean_std means different models produce more "
            "similar simulate behavior on the same ACS/ATUS visitor instrument."
            if visitor_default
            else
            "Lower mean_variance / mean_std means different models produce more "
            "similar answers on the same items."
        ),
    }

    target = Path(out_dir) if out_dir is not None else Path(run_dirs[0]).parent / "robustness"
    target.mkdir(parents=True, exist_ok=True)
    (target / "robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (target / "report.md").write_text(render_robustness_markdown(summary))
    summary["out_dir"] = str(target)
    return summary


def render_robustness_markdown(summary: dict[str, Any]) -> str:
    mean_stats = summary["mean_code_variance"]
    share_stats = summary["share_vector_variance"]
    lines = [
        "# 跨模型稳健性审计",
        "",
        f"- 模型数：{summary['n_runs']}",
        f"- 模型：{', '.join(f'`{m}`' for m in summary['models'])}",
        "",
        "## 汇总",
        "",
        f"- 均分跨模型方差（mean）：{mean_stats['mean_variance']:.4f}",
        f"- 均分跨模型标准差（mean）：{mean_stats['mean_std']:.4f}",
        f"- 均分跨模型标准差（max）：{mean_stats['max_std']:.4f}",
        f"- 份额向量跨模型方差（mean）：{share_stats['mean_variance']:.4f}",
        f"- 份额向量跨模型标准差（mean）：{share_stats['mean_std']:.4f}",
        "",
        summary["interpretation"],
        "",
        "## 逐题均分方差",
        "",
        "| 题 | 跨模型方差 | 跨模型标准差 |",
        "| --- | ---: | ---: |",
    ]
    for row in summary["per_item"]:
        lines.append(
            f"| `{row['item_id']}` | {row['cross_model_variance']:.4f} | "
            f"{row['cross_model_std']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_run(run_dir: Path) -> dict[str, Any]:
    shares_path = run_dir / "option_shares.json"
    if not shares_path.is_file():
        raise RobustnessError(f"missing option_shares.json in {run_dir}")
    shares = json.loads(shares_path.read_text())
    model = "unknown"
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        model = str(meta.get("model") or model)
    return {"run_dir": run_dir, "model": model, "shares": shares}


def _mean_codes(run: dict[str, Any], item_ids: list[str]) -> list[float]:
    by_id = {item["item_id"]: item for item in run["shares"]["items"]}
    means = []
    for item_id in item_ids:
        if item_id not in by_id:
            raise RobustnessError(f"{run['run_dir']} missing item {item_id}")
        shares = [float(x) for x in by_id[item_id]["shares"]]
        means.append(sum(code * weight for code, weight in enumerate(shares, start=1)))
    return means


def _flat_shares(run: dict[str, Any], item_ids: list[str]) -> list[float]:
    by_id = {item["item_id"]: item for item in run["shares"]["items"]}
    flat: list[float] = []
    for item_id in item_ids:
        if item_id not in by_id:
            raise RobustnessError(f"{run['run_dir']} missing item {item_id}")
        flat.extend(float(x) for x in by_id[item_id]["shares"])
    return flat
