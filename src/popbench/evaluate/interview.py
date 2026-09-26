"""Score twin-2k-50 answers against the human option shares and scale means."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from popbench.dao.interview import SCALE_ZH, interview_dir, keyed_score, load_panel, panel_path
from popbench.dao.io import read_jsonl
from popbench.dao.schema import InterviewRecord
from popbench.evaluate.metrics import (
    closeness_score,
    cronbach_alpha,
    jensen_shannon,
    kolmogorov_smirnov,
    opinion_alignment,
    ordinal_accuracy,
    pearson,
    population_std,
    quadratic_weighted_kappa,
    total_variation_distance,
    wasserstein_1d,
)

_RANDOM_SHARES = [0.2, 0.2, 0.2, 0.2, 0.2]
_RANDOM_CODE_MEAN = 3.0
_RANDOM_CODE_VARIANCE = 2.0
_POINT_LABELS = (
    ("exact_match", "逐题命中率"),
    ("ordinal_accuracy", "序数准确率"),
    ("kappa", "二次加权 Kappa"),
    ("item_correlation", "逐题相关"),
    ("profile_correlation", "个人轮廓相关"),
    ("trait_correlation", "特质分相关"),
    ("tvd", "总变异距离"),
    ("js", "JS 散度"),
    ("wasserstein", "1-Wasserstein"),
    ("alignment", "OpinionQA 对齐"),
    ("ks", "KS 距离"),
    ("code_mae", "选项均值差"),
    ("scale_mae", "量表均值差"),
    ("variance_ratio", "方差比"),
    ("alpha", "Cronbach α"),
)
from popbench.simulate.client import model_from_run

_PANEL_TEXT = {
    "nemotron": "Nemotron-Personas-USA，18 岁以上，按年龄段和性别配额",
    "twin2k": "Twin-2K 受访者。人物卡只有人口学，这 50 题的答案不进提示",
}


def score_simulation(
    responses: list[dict[str, Any]],
    baseline: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare simulated option shares and scale means with the human baseline."""
    by_item: dict[str, list[int]] = {item["item_id"]: [] for item in items}
    person_scales: dict[str, list[float]] = {scale: [] for scale in SCALE_ZH}
    keyed_rows: dict[str, list[list[float]]] = {scale: [] for scale in SCALE_ZH}
    n_invalid = 0
    n_answers = 0
    for person in responses:
        scores: dict[str, list[int]] = {scale: [] for scale in SCALE_ZH}
        answered = {row["item_id"]: row for row in person["answers"]}
        for item in items:
            n_answers += 1
            row = answered.get(item["item_id"])
            if row is None or not row.get("ok"):
                n_invalid += 1
                continue
            code = int(row["code"])
            by_item[item["item_id"]].append(code)
            scores[item["scale"]].append(keyed_score(code, item["reverse"]))
        for scale, values in scores.items():
            if values:
                person_scales[scale].append(sum(values) / len(values))
                keyed_rows[scale].append([float(value) for value in values])

    item_stats = []
    tvds = []
    jss = []
    wassersteins = []
    alignments = []
    ks_values = []
    code_mae = []
    point_tvd = []
    point_js = []
    point_wasserstein = []
    point_alignment = []
    point_ks = []
    point_code_mae = []
    human_by_id = {row["item_id"]: row for row in baseline["items"]}
    for item in items:
        human = human_by_id[item["item_id"]]
        codes = by_item[item["item_id"]]
        shares = [codes.count(code) / len(codes) for code in range(1, 6)] if codes else [0.0] * 5
        tvd = total_variation_distance(shares, human["shares"]) if codes else None
        js = jensen_shannon(shares, human["shares"]) if codes else None
        wasserstein = wasserstein_1d(shares, human["shares"]) if codes else None
        alignment = opinion_alignment(shares, human["shares"]) if codes else None
        ks = kolmogorov_smirnov(shares, human["shares"]) if codes else None
        mean_code = sum(codes) / len(codes) if codes else None
        if tvd is not None:
            tvds.append(tvd)
            jss.append(js)
            wassersteins.append(wasserstein)
            alignments.append(alignment)
            ks_values.append(ks)
            human_shares = human["shares"]
            point_tvd.append(
                closeness_score(
                    tvd,
                    perfect=0,
                    random_floor=total_variation_distance(_RANDOM_SHARES, human_shares),
                )
            )
            point_js.append(
                closeness_score(
                    js,
                    perfect=0,
                    random_floor=jensen_shannon(_RANDOM_SHARES, human_shares),
                )
            )
            point_wasserstein.append(
                closeness_score(
                    wasserstein,
                    perfect=0,
                    random_floor=wasserstein_1d(_RANDOM_SHARES, human_shares),
                )
            )
            point_alignment.append(
                closeness_score(
                    alignment,
                    perfect=1,
                    random_floor=opinion_alignment(_RANDOM_SHARES, human_shares),
                )
            )
            point_ks.append(
                closeness_score(
                    ks,
                    perfect=0,
                    random_floor=kolmogorov_smirnov(_RANDOM_SHARES, human_shares),
                )
            )
        if mean_code is not None and human["mean_code"] is not None:
            absolute_code_error = abs(mean_code - human["mean_code"])
            code_mae.append(absolute_code_error)
            point_code_mae.append(
                closeness_score(
                    absolute_code_error,
                    perfect=0,
                    random_floor=abs(_RANDOM_CODE_MEAN - human["mean_code"]),
                )
            )
        item_stats.append(
            {
                "item_id": item["item_id"],
                "turn": item["turn"],
                "scale": item["scale"],
                "n": len(codes),
                "tvd": tvd,
                "js": js,
                "wasserstein": wasserstein,
                "alignment": alignment,
                "ks": ks,
                "simulated_mean_code": mean_code,
                "human_mean_code": human["mean_code"],
            }
        )
    scales = {}
    for scale, values in person_scales.items():
        human_row = baseline["scales"][scale]
        human_mean = human_row["mean"]
        human_std = human_row.get("std")
        simulated_mean = sum(values) / len(values) if values else None
        simulated_std = population_std(values)
        variance_ratio = None
        if simulated_std is not None and human_std not in (None, 0):
            variance_ratio = (simulated_std / human_std) ** 2
        scales[scale] = {
            "n": len(values),
            "simulated_mean": simulated_mean,
            "human_mean": human_mean,
            "absolute_error": None if simulated_mean is None else abs(simulated_mean - human_mean),
            "simulated_std": simulated_std,
            "human_std": human_std,
            "variance_ratio": variance_ratio,
            "simulated_alpha": cronbach_alpha(keyed_rows[scale]),
        }
    return {
        "n_people": len(responses),
        "n_answers": n_answers,
        "n_invalid": n_invalid,
        "mean_tvd": _mean(tvds),
        "mean_js": _mean(jss),
        "mean_wasserstein": _mean(wassersteins),
        "mean_alignment": _mean(alignments),
        "mean_ks": _mean(ks_values),
        "mean_abs_code_error": _mean(code_mae),
        "points": {
            "tvd": _mean(point_tvd),
            "js": _mean(point_js),
            "wasserstein": _mean(point_wasserstein),
            "alignment": _mean(point_alignment),
            "ks": _mean(point_ks),
            "code_mae": _mean(point_code_mae),
        },
        "items": item_stats,
        "scales": scales,
    }


def paired_closeness(
    records: list[InterviewRecord],
    responses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Person-level closeness. None when the panel has no gold answers."""
    if not records or all(turn.gold_code is None for turn in records[0].turns):
        return None
    by_id = {row["id"]: row for row in responses}
    predicted: list[float] = []
    gold: list[float] = []
    by_item: dict[str, tuple[list[float], list[float]]] = {}
    profile_rs: list[float] = []
    trait_sim: dict[str, list[float]] = {scale: [] for scale in SCALE_ZH}
    trait_gold: dict[str, list[float]] = {scale: [] for scale in SCALE_ZH}
    for record in records:
        response = by_id.get(record.id)
        answered = {} if response is None else {row["item_id"]: row for row in response["answers"]}
        sim_items: list[float] = []
        gold_items: list[float] = []
        sim_traits: dict[str, list[int]] = {scale: [] for scale in SCALE_ZH}
        gold_traits: dict[str, list[int]] = {scale: [] for scale in SCALE_ZH}
        for turn in record.turns:
            if turn.gold_code is None:
                continue
            row = answered.get(turn.item_id)
            if row is None or not row.get("ok"):
                continue
            code = float(row["code"])
            gold_code = float(turn.gold_code)
            predicted.append(code)
            gold.append(gold_code)
            bucket = by_item.setdefault(turn.item_id, ([], []))
            bucket[0].append(code)
            bucket[1].append(gold_code)
            sim_items.append(float(keyed_score(int(code), turn.reverse)))
            gold_items.append(float(keyed_score(int(gold_code), turn.reverse)))
            sim_traits[turn.scale].append(keyed_score(int(code), turn.reverse))
            gold_traits[turn.scale].append(keyed_score(int(gold_code), turn.reverse))
        if len(sim_items) >= 2:
            profile = pearson(sim_items, gold_items)
            if profile is not None:
                profile_rs.append(profile)
        for scale in SCALE_ZH:
            if sim_traits[scale] and gold_traits[scale]:
                trait_sim[scale].append(sum(sim_traits[scale]) / len(sim_traits[scale]))
                trait_gold[scale].append(sum(gold_traits[scale]) / len(gold_traits[scale]))
    if not predicted:
        return None
    item_rs = []
    for sim_codes, gold_codes in by_item.values():
        if len(sim_codes) < 2:
            continue
        correlation = pearson(sim_codes, gold_codes)
        if correlation is not None:
            item_rs.append(correlation)
    trait_rs = []
    for scale in SCALE_ZH:
        if len(trait_sim[scale]) < 2:
            continue
        correlation = pearson(trait_sim[scale], trait_gold[scale])
        if correlation is not None:
            trait_rs.append(correlation)
    return {
        "exact_match": sum(left == right for left, right in zip(predicted, gold)) / len(predicted),
        "ordinal_accuracy": ordinal_accuracy(predicted, gold, span=4),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            [int(value) for value in predicted],
            [int(value) for value in gold],
            categories=5,
        ),
        "item_correlation": _mean(item_rs),
        "profile_correlation": _mean(profile_rs),
        "trait_correlation": _mean(trait_rs),
        "n_pairs": len(predicted),
        "random_ordinal_accuracy": _random_ordinal_accuracy(gold),
    }


def individual_accuracy(
    records: list[InterviewRecord],
    responses: list[dict[str, Any]],
) -> float | None:
    """Exact-match rate against gold codes. None when the panel has no gold."""
    if not records or all(turn.gold_code is None for turn in records[0].turns):
        return None
    by_id = {row["id"]: row for row in responses}
    hits = 0
    total = 0
    for record in records:
        response = by_id.get(record.id)
        answered = {} if response is None else {row["item_id"]: row for row in response["answers"]}
        for turn in record.turns:
            if turn.gold_code is None:
                continue
            total += 1
            row = answered.get(turn.item_id)
            if row and row.get("ok") and row.get("code") == turn.gold_code:
                hits += 1
    if total == 0:
        return None
    return hits / total


def evaluate_interview(data_dir: Path, run_dir: Path, *, panel: str | None = None) -> dict[str, Any]:
    """Score a saved twin-2k-50 run and write summary.json plus report.md."""
    run_dir = Path(run_dir)
    response_path = run_dir / "responses.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(response_path)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    panel_name = panel or meta.get("panel") or "nemotron"
    records = load_panel(panel_path(data_dir, panel_name))
    responses = read_jsonl(response_path)
    baseline = json.loads((interview_dir(data_dir) / "human_baseline.json").read_text())
    items = [turn.to_dict() for turn in records[0].turns]
    summary = score_simulation(responses, baseline, items)
    paired = paired_closeness(records, responses)
    if paired is not None:
        summary["individual_accuracy"] = paired["exact_match"]
        summary["paired"] = paired
    summary["human_alpha"] = _human_alphas(data_dir, items)
    summary["score_100"] = score_100(summary, items)
    summary["model"] = model_from_run(meta)
    summary["panel"] = panel_name
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "report.md").write_text(
        render_result_markdown(summary, {"model": summary["model"], "panel": panel_name})
    )
    return summary


def render_result_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """One page: sample, persona condition, item distance, scale distance."""
    panel = meta.get("panel", "nemotron")
    lines = [
        "# Twin-2K-50 多轮模拟",
        "",
        "- 卷子：大五人格 44 题 + 绿色消费 6 题，共 50 轮",
        f"- 模拟人数：{summary['n_people']}",
        f"- 人物：{_PANEL_TEXT.get(panel, panel)}",
        f"- 模型：{meta.get('model') or summary.get('model')}",
        "- 人类参照：Twin-2K 全样本在同一套题上的选项份额和量表均分",
        f"- 无效回答：{summary['n_invalid']} / {summary['n_answers']}",
        "",
    ]
    score = summary.get("score_100") or {}
    points = score.get("metrics") or {}
    if score.get("overall") is not None:
        lines.extend(
            [
                f"**总分：{score['overall']:.1f} / 100**。100 分是和真人一致，0 分是在五个选项里均匀随机作答。",
                "",
                "| 部分 | 分数 |",
                "| --- | ---: |",
            ]
        )
        for key, label in (
            ("individual", "个体"),
            ("distribution", "分布"),
            ("scales", "量表"),
        ):
            if score.get(key) is not None:
                lines.append(f"| {label} | {score[key]:.1f} |")
        lines.extend(["", "| 指标 | 分数 |", "| --- | ---: |"])
        for key, label in _POINT_LABELS:
            value = score.get("metrics", {}).get(key)
            if value is not None:
                lines.append(f"| {label} | {value:.1f} |")
        lines.append("")
    paired = summary.get("paired")
    if not isinstance(paired, dict):
        lines.append("这个面板没有逐人标准答案，下面只比较分布和量表。")
    else:
        lines.extend(
            [
                "同一个人的答案可以和真人选择对齐。随机五选一的命中率是 0.200。",
                "Twin-2K 第四波没有重测这 50 题，所以这里报原始相关，不除以其他论文里的重测系数。",
                "",
                "## 个体接近度",
                "",
                _scored("逐题命中率", paired.get("exact_match"), points, "exact_match"),
                _scored(
                    "序数准确率",
                    paired.get("ordinal_accuracy"),
                    points,
                    "ordinal_accuracy",
                    note="1 减去 |预测−真人| / 4，Toubia 等人 2025。",
                ),
                _scored("二次加权 Kappa", paired.get("quadratic_weighted_kappa"), points, "kappa"),
                _scored(
                    "逐题 Pearson 相关",
                    paired.get("item_correlation"),
                    points,
                    "item_correlation",
                    note="各题上模拟与真人的相关再平均，Toubia 等人 2025 mega-study。",
                ),
                _scored(
                    "个人轮廓相关",
                    paired.get("profile_correlation"),
                    points,
                    "profile_correlation",
                    note="每人 50 题向量的 Pearson 再平均。",
                ),
                _scored(
                    "特质分相关",
                    paired.get("trait_correlation"),
                    points,
                    "trait_correlation",
                    note="五个量表分各自的 Pearson 再平均，Park 等人 2024 用它比较大五人格。",
                ),
                "",
            ]
        )
    by_scale = (summary.get("score_100") or {}).get("by_scale") or {}
    lines.extend(
        [
            "",
            "## 量表（1–5，反向题已翻转）",
            "",
            "| 量表 | 人类均值 | 模拟均值 | 绝对差 | 均值分 | 方差比 | 方差分 | 人类 α | 模拟 α | α 分 |",
            "| --- | --- | --- | --- | ---: | --- | ---: | --- | --- | ---: |",
        ]
    )
    human_alpha = summary.get("human_alpha") or {}
    for scale, row in summary["scales"].items():
        scale_points = by_scale.get(scale) or {}
        lines.append(
            "| {name} | {human:.2f} | {sim} | {err} | {mean_score} | {var} | {var_score} | {human_alpha} | {sim_alpha} | {alpha_score} |".format(
                name=SCALE_ZH[scale],
                human=row["human_mean"],
                sim="—" if row["simulated_mean"] is None else f"{row['simulated_mean']:.2f}",
                err="—" if row["absolute_error"] is None else f"{row['absolute_error']:.2f}",
                mean_score=_fmt(scale_points.get("scale_mae")),
                var=_fmt(row.get("variance_ratio")),
                var_score=_fmt(scale_points.get("variance_ratio")),
                human_alpha=_fmt(human_alpha.get(scale)),
                sim_alpha=_fmt(row.get("simulated_alpha")),
                alpha_score=_fmt(scale_points.get("alpha")),
            )
        )
    lines.extend(
        [
            "",
            "方差比是模拟量表分方差除以人类方差。小于 1 表示模拟比真人更集中，Toubia 等人 2025 mega-study 里的数字孪生就是这样。",
            "Cronbach α 衡量同一量表里各题是否一致。",
            "",
            "## 逐题分布",
            "",
            _scored(
                "平均总变异距离（TVD）",
                summary["mean_tvd"],
                points,
                "tvd",
                note="0 是份额相同，1 是没有重叠。",
            ),
            _scored("平均 JS 散度（bit）", summary["mean_js"], points, "js"),
            _scored(
                "平均 1-Wasserstein",
                summary.get("mean_wasserstein"),
                points,
                "wasserstein",
                note="相邻档的距离是 1，Santurkar 等人 2023。",
            ),
            _scored(
                "OpinionQA 对齐分",
                summary.get("mean_alignment"),
                points,
                "alignment",
                note="1 减去 Wasserstein / 4，1 是完全重合。",
            ),
            _scored("平均 KS 距离", summary.get("mean_ks"), points, "ks"),
            _scored("选项编码的平均绝对差", summary["mean_abs_code_error"], points, "code_mae"),
            "",
        ]
    )
    return "\n".join(lines)


def score_100(summary: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse the paper metrics onto one 0–100 score.

    100 matches the human answers. 0 matches uniform random choices on the
    five-point scale.
    """
    metrics: dict[str, float] = {}
    distribution = summary.get("points") or {}
    for key in ("tvd", "js", "wasserstein", "alignment", "ks", "code_mae"):
        value = distribution.get(key)
        if value is not None:
            metrics[key] = value
    paired = summary.get("paired")
    if isinstance(paired, dict):
        metrics["exact_match"] = closeness_score(
            paired["exact_match"], perfect=1, random_floor=0.2
        )
        if paired.get("ordinal_accuracy") is not None and paired.get("random_ordinal_accuracy") is not None:
            metrics["ordinal_accuracy"] = closeness_score(
                paired["ordinal_accuracy"],
                perfect=1,
                random_floor=paired["random_ordinal_accuracy"],
            )
        if paired.get("quadratic_weighted_kappa") is not None:
            metrics["kappa"] = closeness_score(
                paired["quadratic_weighted_kappa"], perfect=1, random_floor=0
            )
        for key in ("item_correlation", "profile_correlation", "trait_correlation"):
            if paired.get(key) is not None:
                metrics[key] = closeness_score(paired[key], perfect=1, random_floor=0)
    scale_counts: dict[str, int] = {scale: 0 for scale in SCALE_ZH}
    for item in items:
        scale_counts[item["scale"]] = scale_counts.get(item["scale"], 0) + 1
    human_alpha = summary.get("human_alpha") or {}
    mean_points = []
    variance_points = []
    alpha_points = []
    by_scale: dict[str, dict[str, float]] = {}
    for scale, row in summary["scales"].items():
        scale_points: dict[str, float] = {}
        if row.get("absolute_error") is not None:
            scale_points["scale_mae"] = closeness_score(
                row["absolute_error"],
                perfect=0,
                random_floor=abs(_RANDOM_CODE_MEAN - row["human_mean"]),
            )
            mean_points.append(scale_points["scale_mae"])
        human_std = row.get("human_std")
        if row.get("variance_ratio") is not None and human_std not in (None, 0) and scale_counts.get(scale):
            random_ratio = (_RANDOM_CODE_VARIANCE / scale_counts[scale]) / (human_std**2)
            scale_points["variance_ratio"] = closeness_score(
                row["variance_ratio"], perfect=1, random_floor=random_ratio
            )
            variance_points.append(scale_points["variance_ratio"])
        if row.get("simulated_alpha") is not None and human_alpha.get(scale) not in (None, 0):
            scale_points["alpha"] = closeness_score(
                row["simulated_alpha"], perfect=human_alpha[scale], random_floor=0
            )
            alpha_points.append(scale_points["alpha"])
        if scale_points:
            by_scale[scale] = scale_points
    if mean_points:
        metrics["scale_mae"] = _mean(mean_points)
    if variance_points:
        metrics["variance_ratio"] = _mean(variance_points)
    if alpha_points:
        metrics["alpha"] = _mean(alpha_points)
    individual_keys = (
        "exact_match",
        "ordinal_accuracy",
        "kappa",
        "item_correlation",
        "profile_correlation",
        "trait_correlation",
    )
    distribution_keys = ("tvd", "js", "wasserstein", "alignment", "ks", "code_mae")
    scale_keys = ("scale_mae", "variance_ratio", "alpha")
    families = {
        "individual": _mean([metrics[key] for key in individual_keys if key in metrics]),
        "distribution": _mean([metrics[key] for key in distribution_keys if key in metrics]),
        "scales": _mean([metrics[key] for key in scale_keys if key in metrics]),
    }
    overall = _mean([value for value in families.values() if value is not None])
    return {"overall": overall, **families, "metrics": metrics, "by_scale": by_scale}


def _random_ordinal_accuracy(gold: list[float]) -> float:
    total = 0.0
    for value in gold:
        deviation = sum(abs(option - value) for option in range(1, 6)) / 5
        total += 1 - deviation / 4
    return total / len(gold)


def _human_alphas(data_dir: Path, items: list[dict[str, Any]]) -> dict[str, float | None]:
    """Cronbach's alpha of the full Twin-2K sample on each scale."""
    from popbench.dao.interview import _code_frame
    from popbench.dao.twin2k import load_twin2k

    prepared = [
        {"column": item.get("column", item["item_id"]), "item_id": item["item_id"]}
        for item in items
    ]
    codes = _code_frame(load_twin2k(data_dir, download=False).wave1_3, prepared)
    grouped: dict[str, list[str]] = {scale: [] for scale in SCALE_ZH}
    reverse = {item["item_id"]: item["reverse"] for item in items}
    for item in items:
        grouped[item["scale"]].append(item["item_id"])
    alphas: dict[str, float | None] = {}
    for scale, item_ids in grouped.items():
        matrix: list[list[float]] = []
        for row in codes.itertuples(index=False):
            values = dict(zip(codes.columns, row))
            scored: list[float] = []
            complete = True
            for item_id in item_ids:
                code = values[item_id]
                if math.isnan(float(code)):
                    complete = False
                    break
                scored.append(float(keyed_score(int(code), reverse[item_id])))
            if complete:
                matrix.append(scored)
        alphas[scale] = cronbach_alpha(matrix)
    return alphas


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _scored(
    label: str,
    value: float | None,
    points: dict[str, float],
    key: str,
    *,
    note: str = "",
) -> str:
    score = points.get(key)
    score_text = "" if score is None else f"（{score:.1f} / 100）"
    suffix = f"。{note}" if note else ""
    return f"- {label}：{_fmt(value)}{score_text}{suffix}"
