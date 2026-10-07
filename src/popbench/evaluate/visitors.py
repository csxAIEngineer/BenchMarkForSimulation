"""Behavioral-fit evaluation for ACS/ATUS visitor visit runs.

Compares simulated Likert option shares and ordinal codes with an ATUS-derived
human baseline using distribution correlation, MSE, and response-amplitude
coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from popbench.dao.visitors import ATUS_MAJOR, atus_dir, visitors_built_dir
from popbench.evaluate.metrics import (
    distribution_correlation,
    mean_squared_error,
    pearson,
    response_amplitude_coverage,
    response_amplitude_ratio,
)
from popbench.simulate.visitors import VISIT_INSTRUMENT, VISIT_OPTIONS, VISIT_TURNS

# Visit item → ATUS major activity label used for the human baseline.
ITEM_ATUS = {
    "visit_personal_care": "personal_care",
    "visit_work": "work",
    "visit_household": "household",
    "visit_care": "care_household",
    "visit_eating": "eating",
    "visit_leisure": "leisure",
    "visit_sports": "sports",
    "visit_travel": "traveling",
}


class VisitorEvalError(RuntimeError):
    """Visitor evaluation could not load a run or ATUS baseline."""


def evaluate_visitors(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    """Score one visitor simulate run against the ATUS ordinal baseline."""
    run_dir = Path(run_dir)
    shares_path = run_dir / "option_shares.json"
    responses_path = run_dir / "responses.jsonl"
    if not shares_path.is_file():
        raise VisitorEvalError(f"missing {shares_path}; run simulate --panel visitors first")
    if not responses_path.is_file():
        raise VisitorEvalError(f"missing {responses_path}")

    simulated = json.loads(shares_path.read_text())
    responses = [
        json.loads(line)
        for line in responses_path.read_text().splitlines()
        if line.strip()
    ]
    baseline = load_or_build_atus_baseline(data_dir)
    fit = behavioral_fit(simulated, responses, baseline)

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())

    summary = {
        "kind": "visitor_behavioral_fit",
        "instrument": VISIT_INSTRUMENT,
        "model": meta.get("model"),
        "run_dir": str(run_dir),
        "n_people": meta.get("n_people", len(responses)),
        "seed": meta.get("seed"),
        "request": meta.get("request"),
        "baseline": {
            "source": baseline["source"],
            "n_respondents": baseline["n_respondents"],
        },
        "behavioral_fit": fit,
        "per_item": fit["per_item"],
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = render_visitor_eval_markdown(summary)
    # Keep simulate narrative, append fit section.
    existing = (run_dir / "report.md").read_text() if (run_dir / "report.md").is_file() else ""
    if "## 行为拟合" in existing or "## Behavioral fit" in existing:
        (run_dir / "report.md").write_text(report)
    else:
        (run_dir / "report.md").write_text(existing.rstrip() + "\n\n" + report)
    return summary


def load_or_build_atus_baseline(data_dir: Path) -> dict[str, Any]:
    """ATUS respondents mapped onto the five visit Likert bins."""
    cache = visitors_built_dir(data_dir) / "atus_visit_baseline.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    path = atus_dir(data_dir) / "sum" / "atussum_2023.dat"
    if not path.is_file():
        raise VisitorEvalError(
            f"ATUS summary missing: {path}. Run `popbench fetch --visitors-only`."
        )
    frame = pd.read_csv(path)
    adults = frame.loc[frame["TEAGE"] >= 18].copy()
    minutes = _respondent_minutes(adults)
    items = []
    for turn in VISIT_TURNS:
        label = ITEM_ATUS[turn.item_id]
        values = minutes[label]
        codes = _minutes_to_codes(values)
        shares = _code_shares(codes, n_options=len(VISIT_OPTIONS))
        items.append(
            {
                "item_id": turn.item_id,
                "turn": turn.turn,
                "atus_label": label,
                "shares": shares,
                "codes": codes,
                "mean_code": float(sum(codes) / len(codes)) if codes else None,
                "min_code": int(min(codes)) if codes else None,
                "max_code": int(max(codes)) if codes else None,
            }
        )
    baseline = {
        "source": "ATUS 2023 activity summary → visit Likert bins (quintiles)",
        "n_respondents": int(len(adults)),
        "n_options": len(VISIT_OPTIONS),
        "items": items,
    }
    visitors_built_dir(data_dir).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


def behavioral_fit(
    simulated_shares: dict[str, Any],
    responses: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compute distribution correlation, MSE, and amplitude coverage."""
    baseline_by_id = {item["item_id"]: item for item in baseline["items"]}
    sim_by_id = {item["item_id"]: item for item in simulated_shares["items"]}
    sim_codes_by_item = _codes_from_responses(responses)

    flat_sim: list[float] = []
    flat_human: list[float] = []
    per_item = []
    mse_share_values: list[float] = []
    corr_item_values: list[float] = []
    coverage_values: list[float] = []
    ratio_values: list[float] = []
    mean_code_sim: list[float] = []
    mean_code_human: list[float] = []

    for turn in VISIT_TURNS:
        item_id = turn.item_id
        human = baseline_by_id[item_id]
        sim = sim_by_id[item_id]
        human_shares = [float(x) for x in human["shares"]]
        sim_shares = [float(x) for x in sim["shares"]]
        flat_sim.extend(sim_shares)
        flat_human.extend(human_shares)
        share_mse = mean_squared_error(sim_shares, human_shares)
        mse_share_values.append(share_mse)
        item_corr = distribution_correlation(sim_shares, human_shares)
        if item_corr is not None:
            corr_item_values.append(item_corr)

        sim_codes = [float(c) for c in sim_codes_by_item.get(item_id, [])]
        human_codes = [float(c) for c in human["codes"]]
        coverage = (
            response_amplitude_coverage(sim_codes, human_codes)
            if sim_codes and human_codes
            else None
        )
        ratio = (
            response_amplitude_ratio(sim_codes, human_codes)
            if sim_codes and human_codes
            else None
        )
        if coverage is not None:
            coverage_values.append(coverage)
        if ratio is not None:
            ratio_values.append(ratio)

        sim_mean = (
            sum(sim_codes) / len(sim_codes)
            if sim_codes
            else sum(i * w for i, w in enumerate(sim_shares, start=1))
        )
        human_mean = float(human["mean_code"]) if human["mean_code"] is not None else None
        if human_mean is not None:
            mean_code_sim.append(sim_mean)
            mean_code_human.append(human_mean)

        per_item.append(
            {
                "item_id": item_id,
                "turn": turn.turn,
                "distribution_correlation": item_corr,
                "share_mse": share_mse,
                "response_amplitude_coverage": coverage,
                "response_amplitude_ratio": ratio,
                "simulated_mean_code": sim_mean,
                "human_mean_code": human_mean,
                "mean_code_mse": None
                if human_mean is None
                else (sim_mean - human_mean) ** 2,
            }
        )

    overall_corr = distribution_correlation(flat_sim, flat_human)
    mean_code_corr = (
        None if len(mean_code_sim) < 2 else pearson(mean_code_sim, mean_code_human)
    )
    return {
        "distribution_correlation": overall_corr,
        "mean_item_distribution_correlation": (
            sum(corr_item_values) / len(corr_item_values) if corr_item_values else None
        ),
        "share_mse": sum(mse_share_values) / len(mse_share_values) if mse_share_values else None,
        "mean_code_mse": (
            mean_squared_error(mean_code_sim, mean_code_human) if mean_code_sim else None
        ),
        "mean_code_correlation": mean_code_corr,
        "response_amplitude_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        ),
        "response_amplitude_ratio": (
            sum(ratio_values) / len(ratio_values) if ratio_values else None
        ),
        "per_item": per_item,
    }


def render_visitor_eval_markdown(summary: dict[str, Any]) -> str:
    fit = summary["behavioral_fit"]

    def _fmt(value: float | None, digits: int = 3) -> str:
        if value is None:
            return "—"
        return f"{value:.{digits}f}"

    lines = [
        "## 行为拟合（对照 ATUS）",
        "",
    ]
    if summary.get("model"):
        lines.append(f"- 模型：`{summary['model']}`")
    lines.extend(
        [
        f"- 基线：{summary['baseline']['source']}（N={summary['baseline']['n_respondents']}）",
        ]
    )
    lines.extend(
        [
        f"- 分布相关系数（flatten shares）：{_fmt(fit.get('distribution_correlation'))}",
        f"- 逐题分布相关均值：{_fmt(fit.get('mean_item_distribution_correlation'))}",
        f"- 选项份额 MSE：{_fmt(fit.get('share_mse'), 4)}",
        f"- 均分 MSE：{_fmt(fit.get('mean_code_mse'), 4)}",
        f"- 均分相关：{_fmt(fit.get('mean_code_correlation'))}",
        f"- 响应幅度覆盖率（离散支持）：{_fmt(fit.get('response_amplitude_coverage'))}",
        f"- 响应幅度比（模拟跨度/人类跨度）：{_fmt(fit.get('response_amplitude_ratio'))}",
        "",
        "| 题 | 分布相关 | 份额 MSE | 幅度覆盖 | 幅度比 | 模拟均分 | ATUS 均分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fit["per_item"]:
        lines.append(
            "| "
            f"`{row['item_id']}` | "
            f"{_fmt(row['distribution_correlation'])} | "
            f"{_fmt(row['share_mse'], 4)} | "
            f"{_fmt(row['response_amplitude_coverage'])} | "
            f"{_fmt(row['response_amplitude_ratio'])} | "
            f"{_fmt(row['simulated_mean_code'])} | "
            f"{_fmt(row['human_mean_code'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _respondent_minutes(frame: pd.DataFrame) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    activity_cols = [
        column
        for column in frame.columns
        if str(column).startswith("t")
        and str(column)[1:].isdigit()
        and len(str(column)) >= 3
    ]
    for code, label in ATUS_MAJOR.items():
        cols = [column for column in activity_cols if str(column)[1:3] == code]
        if cols:
            out[label] = frame[cols].sum(axis=1).astype(float).tolist()
        else:
            out[label] = [0.0] * len(frame)
    return out


def _minutes_to_codes(values: list[float]) -> list[int]:
    """Map minutes onto 1–5 using empirical quintile edges (inclusive upper)."""
    if not values:
        return []
    series = sorted(values)
    n = len(series)

    def _quantile(q: float) -> float:
        if n == 1:
            return series[0]
        pos = (n - 1) * q
        low = int(pos)
        high = min(low + 1, n - 1)
        frac = pos - low
        return series[low] * (1 - frac) + series[high] * frac

    edges = [_quantile(q) for q in (0.2, 0.4, 0.6, 0.8)]
    codes = []
    for value in values:
        code = 1
        for edge in edges:
            if value > edge:
                code += 1
            else:
                break
        codes.append(min(code, 5))
    return codes


def _code_shares(codes: list[int], *, n_options: int) -> list[float]:
    if not codes:
        return [0.0] * n_options
    return [sum(1 for code in codes if code == option) / len(codes) for option in range(1, n_options + 1)]


def _codes_from_responses(responses: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_item: dict[str, list[int]] = {turn.item_id: [] for turn in VISIT_TURNS}
    for row in responses:
        for answer in row.get("answers") or []:
            if answer.get("ok") and answer.get("code") is not None:
                item_id = str(answer["item_id"])
                if item_id in by_item:
                    by_item[item_id].append(int(answer["code"]))
    return by_item
