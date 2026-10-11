"""Test whether age-banded ATUS persona answers match Twin-2K personality.

The human ages are Twin-2K QID13, whose four options are the same bands as
ATUS. Distribution tests treat those respondents' option shares as the
population. Age tests ask whether the four bands differ, and whether the
simulated age gradient points the same way as the human one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from popbench.dao.atus_populations import LEVEL_TITLES, chi2_sf
from popbench.dao.atus_twin2k import AGE_BAND_ORDER
from popbench.dao.interview import BFI_REVERSE, BFI_SCALES, SCALE_ZH, keyed_score
from popbench.dao.io import read_jsonl
from popbench.evaluate.metrics import total_variation_distance, wasserstein_1d

ALPHA = 0.05
# Monte Carlo resolution has to survive a Holm correction across ~275 items.
DEFAULT_REPS = 9999
# Permutations for the Big Five means and the age-gradient correlation.
DEFAULT_MEAN_REPS = 4999

# Twin-2K QID13 option order.
AGE_CODE_TO_BAND = {1: "18-29", 2: "30-49", 3: "50-64", 4: "65+"}

INSTRUMENT_ZH = {
    "Big Five": "大五人格",
    "Need for cognition": "认知需求",
    "Agentic vs. communal values": "能动与共生价值观",
    "Consumer minimalism": "消费极简",
    "Empathy": "共情",
    "Green values": "绿色价值观",
    "Social desirability": "社会赞许",
    "Conscientiousness": "尽责性量表",
    "Anxiety": "焦虑",
    "Individualism vs. collectivism": "个体与集体主义",
    "Regulatory focus": "调节焦点",
    "Tightwads vs. spendthrift": "吝啬与挥霍",
    "Depression": "抑郁",
    "Need for uniqueness": "独特性需求",
    "Self-monitoring": "自我监控",
    "Self-concept clarity": "自我概念清晰度",
    "Need for closure": "认知闭合需求",
    "Maximization": "最大化",
}


def holm(p_values: list[float]) -> list[float]:
    """Holm adjusted p-values, in the original order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    size = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (size - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def load_human_personality(data_dir: Path, items: list[dict[str, Any]]) -> pd.DataFrame:
    """Wave 1–3 personality codes with the QID13 age band on each respondent."""
    from popbench.dao.interview import InterviewError
    from popbench.dao.survey import _choice_codes, category_split_dir

    root = category_split_dir(data_dir)
    demo = pd.read_csv(root / "demographics" / "wave1_3_response.csv", usecols=["pid", "QID13"])
    age = pd.to_numeric(demo["QID13"], errors="coerce").map(AGE_CODE_TO_BAND)
    demo = demo.assign(age_band=age)
    wave = pd.read_csv(root / "personality" / "wave1_3_response.csv")
    merged = wave.merge(demo[["pid", "age_band"]], on="pid", how="inner")
    merged = merged.loc[merged["age_band"].notna()].copy()
    if merged.empty:
        raise InterviewError("personality responses did not match demographic ages")
    columns: dict[str, Any] = {
        "pid": merged["pid"].to_numpy(),
        "age_band": merged["age_band"].to_numpy(),
    }
    for item in items:
        if item.get("kind") != "choice":
            continue
        columns[item["item_id"]] = _choice_codes(merged, item).to_numpy()
    return pd.DataFrame(columns)


def build_personality_age_report(
    responses: list[dict[str, Any]],
    personas: list[dict[str, Any]],
    items: list[dict[str, Any]],
    human: pd.DataFrame,
    *,
    model: str,
    seed: int,
    n_per_age_band: int | None,
    reps: int = DEFAULT_REPS,
    mean_reps: int = DEFAULT_MEAN_REPS,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Score one model's personality answers against Twin-2K, overall and by age."""
    if reps < 99:
        raise ValueError("reps must be at least 99")
    if mean_reps < 99:
        raise ValueError("mean_reps must be at least 99")
    generator = np.random.default_rng(seed) if rng is None else rng
    people = _simulated_people(responses, personas)
    choice_items = [item for item in items if item.get("kind") == "choice"]
    rows = [
        _score_item(item, people, human, generator, reps)
        for item in choice_items
    ]
    _holm_column(rows, "p_gof", "p_gof_holm")
    _holm_column(rows, "p_mean", "p_mean_holm")
    _holm_column(rows, "p_age_human", "p_age_human_holm")
    _holm_column(rows, "p_age_sim", "p_age_sim_holm")
    for band in AGE_BAND_ORDER:
        _holm_column(rows, f"p_gof_{band}", f"p_gof_holm_{band}")
    bfi = _bfi_section(people, human, generator, mean_reps)
    report = {
        "model": model,
        "seed": seed,
        "n_per_age_band": n_per_age_band,
        "alpha": ALPHA,
        "reps": reps,
        "mean_reps": mean_reps,
        "n_people": len(people),
        "n_human": int(human["pid"].nunique()) if "pid" in human.columns else int(len(human)),
        "n_invalid": _invalid_answers(responses),
        "n_answers": _answer_slots(responses),
        "composition": _composition(personas, human),
        "overall": _overall(rows),
        "bands": [_band_summary(rows, human, people, band) for band in AGE_BAND_ORDER],
        "instruments": _instruments(rows),
        "age_effects": _age_effects(rows, generator, mean_reps),
        "bfi": bfi,
        "farthest": _extremes(rows, reverse=True),
        "closest": _extremes(rows, reverse=False),
        "items": [_compact_item(row) for row in rows],
    }
    report["verdict"] = _verdict(report)
    report["api"] = "CR API"
    return report


def write_personality_age_evaluation(
    run_dir: Path,
    personas: list[dict[str, Any]],
    items: list[dict[str, Any]],
    data_dir: Path,
    *,
    model: str,
    seed: int,
    n_per_age_band: int | None,
) -> dict[str, Any]:
    """Load saved answers and Twin-2K, then write significance.json and significance.md."""
    run_dir = Path(run_dir)
    responses = read_jsonl(run_dir / "responses.jsonl")
    human = load_human_personality(data_dir, items)
    report = build_personality_age_report(
        responses,
        personas,
        items,
        human,
        model=model,
        seed=seed,
        n_per_age_band=n_per_age_band,
    )
    report["api"] = _api_label(run_dir)
    (run_dir / "significance.json").write_text(json.dumps(report) + "\n")
    text = render_personality_age_report(report)
    (run_dir / "significance.md").write_text(text)
    return report


def render_personality_age_report(report: dict[str, Any]) -> str:
    """Chinese markdown for one model on the personality category."""
    overall = report["overall"]
    lines = [
        "# 按年龄段抽样的人物卡回答人格题",
        "",
        f"- 模型：`{report['model']}`，经 {report.get('api') or 'CR API'}。",
        "- 人物卡：ATUS 2023 成年人。每个年龄段按 `TUFINLWGT` 不放回抽"
        f" {report['n_per_age_band']} 人，种子 {report['seed']}。性别不再分层。",
        "- 题目：Twin-2K 人格类全部闭卷选择题。同一量表里还没答过的题合成一次请求，选项只写一次，不要求解释。",
        "- 人类基线：Twin-2K 前三波，年龄用 QID13，四档与 ATUS 的年龄段相同。",
        f"- 模拟人数：{report['n_people']}。人类人数：{report['n_human']}。",
        f"- 无效回答：{report['n_invalid']} / {report['n_answers']}。",
        "",
        "ATUS 受访者和 Twin-2K 受访者不是同一批人。检验问的是：模型拿着人物卡作出的选项分布，能否看作从 Twin-2K 的选项份额里抽出来的。",
        "",
        f"分布检验是多项分布拟合的蒙特卡洛 p 值（{report['reps']} 次）。人类份额为 0 而模拟却选了该选项时，改用期望不低于 0.5 的卡方尾概率。年龄组差异用 Kruskal-Wallis，并列名次做了校正。大五人格均分和年龄梯度用置换检验（{report['mean_reps']} 次）。多项比较用 Holm，α = {report['alpha']}。",
        "",
        "## 样本",
        "",
        _composition_table(report["composition"]),
        "",
        "## 结论",
        "",
    ]
    for sentence in report["verdict"]:
        lines.append(sentence)
        lines.append("")
    lines.extend(
        [
            "## 整体拟合",
            "",
            f"- 选择题：{overall['n_tested']}",
            f"- 平均 TVD：{_fmt(overall['mean_tvd'])}（中位数 {_fmt(overall['median_tvd'])}）",
            f"- 平均 Wasserstein：{_fmt(overall['mean_wasserstein'])}",
            f"- 均分绝对差：{_fmt(overall['mean_abs_code_error'])}",
            f"- 分布 Holm 显著：{overall['n_gof_holm']} / {overall['n_tested']}",
            f"- 均分 Holm 显著：{overall['n_mean_holm']} / {overall['n_tested']}",
            "",
            "TVD 为 0 表示选项份额相同，为 1 表示完全不重叠。均分显著说的是位置不同；分布显著还包括形状不同、均分却接近的情况。",
            "",
            "## 分年龄段拟合",
            "",
            "每一档的人类基线只含 Twin-2K 里该年龄段的人。Holm 在该档的题目内校正。",
            "",
            _band_table(report["bands"]),
            "",
            "## 分量表",
            "",
            _instrument_table(report["instruments"]),
            "",
            "## 年龄差异",
            "",
            _age_paragraph(report["age_effects"]),
            "",
            "## 大五人格",
            "",
            "均分是反向计分后的 1–5。差值为模拟减人类。p 是均分差值的双侧置换概率。年龄 p 是四个年龄段均分的 Kruskal-Wallis。",
            "",
            _bfi_table(report["bfi"]),
            "",
            _bfi_band_table(report["bfi"]),
            "",
            "## 离 Twin-2K 最远的题",
            "",
            *_item_lines(report["farthest"]),
            "",
            "## 离 Twin-2K 最近的题",
            "",
            *_item_lines(report["closest"]),
            "",
        ]
    )
    return "\n".join(lines)


def _api_label(run_dir: Path) -> str:
    """Name the endpoint stored with the run. The key is not read."""
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.is_file():
        return "CR API"
    meta = json.loads(meta_path.read_text())
    request = meta.get("request") or {}
    transport = request.get("transport")
    base = str(meta.get("base_url") or "")
    if transport == "deepseek" or "api.deepseek.com" in base:
        return "DeepSeek 官方 API"
    if transport == "cr" or "creative-reasoning.com" in base:
        return "CR API"
    return base or "API"


def kruskal_p(groups: list[np.ndarray]) -> float | None:
    """Kruskal-Wallis p-value with the tie correction. Large samples use chi-square."""
    arrays = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if len(arrays) < 2:
        return None
    values = np.concatenate(arrays)
    if values.size < len(arrays) + 1:
        return None
    if np.all(values == values[0]):
        return 1.0
    ranks = _average_ranks(values)
    n = int(values.size)
    offset = 0
    statistic = 0.0
    for group in arrays:
        size = len(group)
        mean_rank = float(ranks[offset : offset + size].mean())
        offset += size
        statistic += size * (mean_rank - (n + 1) / 2.0) ** 2
    statistic *= 12.0 / (n * (n + 1))
    _, counts = np.unique(values, return_counts=True)
    tie = float(np.sum(counts.astype(np.int64) ** 3 - counts.astype(np.int64)))
    denom = 1.0 - tie / (n**3 - n)
    if denom <= 0:
        return 1.0
    statistic /= denom
    if statistic < 0:
        statistic = 0.0
    return float(chi2_sf(statistic, len(arrays) - 1))


def welch_z_p(simulated: np.ndarray, human: np.ndarray) -> float | None:
    """Two-sided normal approximation to Welch's test for a difference of means."""
    left = np.asarray(simulated, dtype=float)
    right = np.asarray(human, dtype=float)
    if left.size < 2 or right.size < 2:
        return None
    variance_left = float(left.var(ddof=1))
    variance_right = float(right.var(ddof=1))
    se2 = variance_left / left.size + variance_right / right.size
    diff = float(left.mean() - right.mean())
    if se2 <= 0:
        return 1.0 if diff == 0 else 0.0
    z = abs(diff) / math.sqrt(se2)
    return float(math.erfc(z / math.sqrt(2.0)))


def _score_item(
    item: dict[str, Any],
    people: list[dict[str, Any]],
    human: pd.DataFrame,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    n_options = len(item["options"])
    human_codes, valid = _column_codes(human[item["item_id"]].to_numpy(), n_options)
    human_bands = human["age_band"].to_numpy()[valid]
    sim_codes: list[int] = []
    sim_by_band: dict[str, list[int]] = {band: [] for band in AGE_BAND_ORDER}
    for person in people:
        code = _as_code(person["answers"].get(item["item_id"]), n_options)
        if code is None:
            continue
        sim_codes.append(code)
        band = person["age_band"]
        if band in sim_by_band:
            sim_by_band[band].append(code)
    sim = np.asarray(sim_codes, dtype=int)
    fit = _gof(sim, human_codes, n_options, rng, reps)
    row: dict[str, Any] = {
        "item_id": item["item_id"],
        "instrument": item.get("instrument") or "",
        "prompt": _short(item.get("prompt") or ""),
        "n_options": n_options,
        "n_sim": int(sim.size),
        "n_human": int(human_codes.size),
        "tvd": fit["tvd"],
        "wasserstein": fit["wasserstein"],
        "sim_mean": fit["sim_mean"],
        "human_mean": fit["human_mean"],
        "p_gof": fit["p"],
        "p_mean": welch_z_p(sim.astype(float), human_codes.astype(float)) if sim.size else None,
    }
    human_groups: list[np.ndarray] = []
    sim_groups: list[np.ndarray] = []
    for band in AGE_BAND_ORDER:
        human_band = human_codes[human_bands == band]
        sim_band = np.asarray(sim_by_band[band], dtype=int)
        band_fit = _gof(sim_band, human_band, n_options, rng, reps)
        row[f"tvd_{band}"] = band_fit["tvd"]
        row[f"p_gof_{band}"] = band_fit["p"]
        row[f"n_sim_{band}"] = int(sim_band.size)
        row[f"n_human_{band}"] = int(human_band.size)
        row[f"sim_mean_{band}"] = band_fit["sim_mean"]
        row[f"human_mean_{band}"] = band_fit["human_mean"]
        if human_band.size:
            human_groups.append(human_band.astype(float))
        if sim_band.size:
            sim_groups.append(sim_band.astype(float))
    row["p_age_human"] = kruskal_p(human_groups) if len(human_groups) == len(AGE_BAND_ORDER) else None
    row["p_age_sim"] = kruskal_p(sim_groups) if len(sim_groups) == len(AGE_BAND_ORDER) else None
    return row


def _gof(
    simulated: np.ndarray,
    human: np.ndarray,
    n_options: int,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    if simulated.size == 0 or human.size == 0:
        return {"tvd": None, "wasserstein": None, "sim_mean": None, "human_mean": None, "p": None}
    sim_counts = np.bincount(simulated, minlength=n_options + 1)[1:].astype(float)
    human_counts = np.bincount(human, minlength=n_options + 1)[1:].astype(float)
    shares = human_counts / human_counts.sum()
    sim_shares = sim_counts / sim_counts.sum()
    return {
        "tvd": float(total_variation_distance(sim_shares.tolist(), shares.tolist())),
        "wasserstein": float(wasserstein_1d(sim_shares.tolist(), shares.tolist())),
        "sim_mean": float(simulated.mean()),
        "human_mean": float(human.mean()),
        "p": _multinomial_p(sim_counts, shares, rng, reps),
    }


def _multinomial_p(
    counts: np.ndarray,
    shares: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> float:
    n = int(counts.sum())
    probs = np.clip(np.asarray(shares, dtype=float), 0.0, None)
    total = float(probs.sum())
    if n < 1 or total <= 0:
        return 1.0
    probs = probs / total
    if float(counts[probs == 0].sum()) > 0:
        expected = np.maximum(probs * n, 0.5)
        stat = float(np.sum((counts - expected) ** 2 / expected))
        return float(chi2_sf(stat, max(1, len(counts) - 1)))
    expected = probs * n
    safe = np.maximum(expected, 1e-15)
    stat = float(np.sum((counts - expected) ** 2 / safe))
    draws = rng.multinomial(n, probs, size=reps)
    draw_stats = np.sum((draws - expected) ** 2 / safe, axis=1)
    exceed = int(np.sum(draw_stats >= stat - 1e-8))
    p_mc = (exceed + 1) / (reps + 1)
    if exceed == 0:
        df = int(np.sum(probs > 0) - 1)
        if df >= 1:
            return float(min(p_mc, chi2_sf(stat, df)))
    return float(p_mc)


def _bfi_section(
    people: list[dict[str, Any]],
    human: pd.DataFrame,
    rng: np.random.Generator,
    reps: int,
) -> list[dict[str, Any]]:
    sim_scores = [_bfi_scores(person["answers"]) for person in people]
    human_scores = [_bfi_scores(_row_answers(row)) for row in human.itertuples(index=False)]
    human_bands = human["age_band"].tolist()
    sim_bands = [person["age_band"] for person in people]
    section = []
    for scale in BFI_SCALES:
        sim_pairs = [
            (band, row[scale])
            for band, row in zip(sim_bands, sim_scores)
            if scale in row
        ]
        human_pairs = [
            (band, row[scale])
            for band, row in zip(human_bands, human_scores)
            if scale in row
        ]
        sim = np.asarray([value for _band, value in sim_pairs], dtype=float)
        gold = np.asarray([value for _band, value in human_pairs], dtype=float)
        compared = _mean_diff(sim, gold, rng, reps)
        by_band = {}
        sim_groups = []
        human_groups = []
        for band in AGE_BAND_ORDER:
            sim_band = np.asarray([value for name, value in sim_pairs if name == band], dtype=float)
            human_band = np.asarray([value for name, value in human_pairs if name == band], dtype=float)
            by_band[band] = {
                "sim_mean": None if sim_band.size == 0 else float(sim_band.mean()),
                "human_mean": None if human_band.size == 0 else float(human_band.mean()),
                "n_sim": int(sim_band.size),
                "n_human": int(human_band.size),
            }
            if sim_band.size:
                sim_groups.append(sim_band)
            if human_band.size:
                human_groups.append(human_band)
        section.append(
            {
                "scale": scale,
                "label": SCALE_ZH.get(scale, scale),
                **compared,
                "p_age_sim": kruskal_p(sim_groups) if len(sim_groups) == len(AGE_BAND_ORDER) else None,
                "p_age_human": kruskal_p(human_groups) if len(human_groups) == len(AGE_BAND_ORDER) else None,
                "by_band": by_band,
            }
        )
    return section


def _mean_diff(
    simulated: np.ndarray,
    human: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    if simulated.size < 2 or human.size < 2:
        return {
            "sim_mean": None if simulated.size == 0 else float(simulated.mean()),
            "human_mean": None if human.size == 0 else float(human.mean()),
            "diff": None,
            "d": None,
            "p": None,
            "n_sim": int(simulated.size),
            "n_human": int(human.size),
        }
    diff = float(simulated.mean() - human.mean())
    pooled = np.concatenate([simulated, human])
    n_sim = int(simulated.size)
    exceed = 0
    observed = abs(diff)
    for _ in range(reps):
        rng.shuffle(pooled)
        trial = abs(float(pooled[:n_sim].mean() - pooled[n_sim:].mean()))
        if trial + 1e-12 >= observed:
            exceed += 1
    return {
        "sim_mean": float(simulated.mean()),
        "human_mean": float(human.mean()),
        "diff": diff,
        "d": _cohen_d(simulated, human),
        "p": (exceed + 1) / (reps + 1),
        "n_sim": n_sim,
        "n_human": int(human.size),
    }


def _age_effects(
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    human_p = [row["p_age_human"] for row in rows if row["p_age_human"] is not None]
    sim_p = [row["p_age_sim"] for row in rows if row["p_age_sim"] is not None]
    human_holm = [row["p_age_human_holm"] for row in rows if row.get("p_age_human_holm") is not None]
    sim_holm = [row["p_age_sim_holm"] for row in rows if row.get("p_age_sim_holm") is not None]
    both = [
        row
        for row in rows
        if row.get("p_age_human_holm") is not None and row.get("p_age_sim_holm") is not None
    ]
    young = "18-29"
    old = "65+"
    paired = []
    for row in rows:
        human_young = row.get(f"human_mean_{young}")
        human_old = row.get(f"human_mean_{old}")
        sim_young = row.get(f"sim_mean_{young}")
        sim_old = row.get(f"sim_mean_{old}")
        if None in (human_young, human_old, sim_young, sim_old):
            continue
        if min(row[f"n_sim_{young}"], row[f"n_sim_{old}"], row[f"n_human_{young}"], row[f"n_human_{old}"]) < 5:
            continue
        paired.append((human_old - human_young, sim_old - sim_young))
    if len(paired) >= 5:
        human_delta = np.asarray([pair[0] for pair in paired], dtype=float)
        sim_delta = np.asarray([pair[1] for pair in paired], dtype=float)
        correlation, correlation_p = _correlation_p(human_delta, sim_delta, rng, reps)
        nonzero = (human_delta != 0) & (sim_delta != 0)
        if int(nonzero.sum()) == 0:
            agreement = None
        else:
            agreement = float(np.mean(np.sign(human_delta[nonzero]) == np.sign(sim_delta[nonzero])))
    else:
        correlation, correlation_p, agreement = None, None, None
    return {
        "n_items": len(rows),
        "n_human_raw": sum(p < ALPHA for p in human_p),
        "n_sim_raw": sum(p < ALPHA for p in sim_p),
        "n_human_holm": sum(p < ALPHA for p in human_holm),
        "n_sim_holm": sum(p < ALPHA for p in sim_holm),
        "n_both_holm": sum(
            row["p_age_human_holm"] < ALPHA and row["p_age_sim_holm"] < ALPHA for row in both
        ),
        "n_gradient": len(paired),
        "gradient_r": correlation,
        "gradient_p": correlation_p,
        "sign_agreement": agreement,
    }


def _correlation_p(
    human_delta: np.ndarray,
    sim_delta: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> tuple[float | None, float | None]:
    observed = _pearson(human_delta, sim_delta)
    if observed is None:
        return None, None
    exceed = 0
    shuffled = np.array(sim_delta, dtype=float, copy=True)
    for _ in range(reps):
        rng.shuffle(shuffled)
        trial = _pearson(human_delta, shuffled)
        if trial is not None and abs(trial) + 1e-12 >= abs(observed):
            exceed += 1
    return float(observed), (exceed + 1) / (reps + 1)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0:
        return None
    return float(np.dot(x, y) / denom)


def _overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tested = [row for row in rows if row["tvd"] is not None]
    return {
        "n_tested": len(tested),
        "mean_tvd": _mean([row["tvd"] for row in tested]),
        "median_tvd": _median([row["tvd"] for row in tested]),
        "mean_wasserstein": _mean([row["wasserstein"] for row in tested if row["wasserstein"] is not None]),
        "mean_abs_code_error": _mean(
            [abs(row["sim_mean"] - row["human_mean"]) for row in tested if row["sim_mean"] is not None]
        ),
        "n_gof_holm": sum(row.get("p_gof_holm") is not None and row["p_gof_holm"] < ALPHA for row in tested),
        "n_mean_holm": sum(row.get("p_mean_holm") is not None and row["p_mean_holm"] < ALPHA for row in tested),
    }


def _band_summary(
    rows: list[dict[str, Any]],
    human: pd.DataFrame,
    people: list[dict[str, Any]],
    band: str,
) -> dict[str, Any]:
    tested = [row for row in rows if row.get(f"tvd_{band}") is not None]
    return {
        "band": band,
        "label": LEVEL_TITLES.get(band, band),
        "n_sim": sum(person["age_band"] == band for person in people),
        "n_human": int((human["age_band"] == band).sum()),
        "mean_tvd": _mean([row[f"tvd_{band}"] for row in tested]),
        "n_gof_holm": sum(
            row.get(f"p_gof_holm_{band}") is not None and row[f"p_gof_holm_{band}"] < ALPHA
            for row in tested
        ),
        "n_tested": len(tested),
    }


def _instruments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names: list[str] = []
    for row in rows:
        if row["instrument"] not in names:
            names.append(row["instrument"])
    summary = []
    for name in names:
        group = [row for row in rows if row["instrument"] == name and row["tvd"] is not None]
        summary.append(
            {
                "instrument": name,
                "label": INSTRUMENT_ZH.get(name, name),
                "n_items": len(group),
                "mean_tvd": _mean([row["tvd"] for row in group]),
                "n_gof_holm": sum(row.get("p_gof_holm") is not None and row["p_gof_holm"] < ALPHA for row in group),
            }
        )
    return summary


def _composition(personas: list[dict[str, Any]], human: pd.DataFrame) -> dict[str, Any]:
    bands = []
    for band in AGE_BAND_ORDER:
        people = [
            person
            for person in personas
            if (person.get("demographics") or {}).get("age_band") == band
        ]
        sexes = [(person.get("demographics") or {}).get("sex") for person in people]
        bands.append(
            {
                "band": band,
                "label": LEVEL_TITLES.get(band, band),
                "n_sim": len(people),
                "n_female": sum(sex == "Female" for sex in sexes),
                "n_male": sum(sex == "Male" for sex in sexes),
                "n_human": int((human["age_band"] == band).sum()),
            }
        )
    return {"bands": bands}


def _compact_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "instrument": row["instrument"],
        "tvd": row["tvd"],
        "p_gof_holm": row.get("p_gof_holm"),
        "sim_mean": row["sim_mean"],
        "human_mean": row["human_mean"],
    }


def _extremes(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    ranked = [row for row in rows if row["tvd"] is not None]
    ranked.sort(key=lambda row: row["tvd"], reverse=reverse)
    picked = []
    for row in ranked[:8]:
        picked.append(
            {
                "item_id": row["item_id"],
                "instrument": row["instrument"],
                "label": INSTRUMENT_ZH.get(row["instrument"], row["instrument"]),
                "prompt": row["prompt"],
                "tvd": row["tvd"],
                "p_gof_holm": row.get("p_gof_holm"),
                "sim_mean": row["sim_mean"],
                "human_mean": row["human_mean"],
            }
        )
    return picked


def _verdict(report: dict[str, Any]) -> list[str]:
    overall = report["overall"]
    tested = overall["n_tested"] or 1
    significant = overall["n_gof_holm"]
    lines = [
        (
            f"整体上，{overall['n_tested']} 道题里有 {significant} 道在 Holm 校正后拒绝"
            f"“模拟选项来自 Twin-2K 份额”。平均 TVD 为 {_fmt(overall['mean_tvd'])}。"
        )
    ]
    if significant == 0:
        lines.append("校正后没有题目达到 0.05。以这批模拟样本，不能拒绝它和 Twin-2K 来自同一选项分布。")
    elif significant / tested >= 0.5:
        lines.append(
            f"过半题目的差异具有统计学意义。以 {report['n_people']} 名模拟受访者，"
            "选项分布与 Twin-2K 不同。"
        )
    else:
        lines.append("一部分题目的差异具有统计学意义，其余题目在校正后与 Twin-2K 相容。")
    band_bits = []
    for band in report["bands"]:
        band_bits.append(
            f"{band['label']} TVD {_fmt(band['mean_tvd'])}，显著 {band['n_gof_holm']}/{band['n_tested']}"
        )
    lines.append("分年龄段对照该档人类：" + "；".join(band_bits) + "。")
    age = report["age_effects"]
    lines.append(
        "年龄段之间的 Kruskal-Wallis："
        f"人类 Holm 显著 {age['n_human_holm']} 道，模拟 Holm 显著 {age['n_sim_holm']} 道，"
        f"两边都显著的有 {age['n_both_holm']} 道。"
        f"未校正时人类 {age['n_human_raw']} 道、模拟 {age['n_sim_raw']} 道"
        f"（275 道题在 0.05 下大约会有 {0.05 * age['n_items']:.0f} 道假阳性）。"
    )
    if age["gradient_r"] is None:
        lines.append("年龄梯度相关没有算出来。")
    else:
        lines.append(
            f"65 岁及以上减 18–29 岁的均分差，模拟与人类的相关为 {_fmt(age['gradient_r'])}"
            f"（{age['n_gradient']} 道题，置换 p = {_fmt_p(age['gradient_p'])}）。"
            f"方向一致的题目占 {_fmt_pct(age['sign_agreement'])}。"
        )
        if age["gradient_p"] is not None and age["gradient_p"] < ALPHA and age["gradient_r"] > 0:
            lines.append("年龄方向与 Twin-2K 相同，这个相关具有统计学意义。")
        elif age["gradient_p"] is not None and age["gradient_p"] < ALPHA and age["gradient_r"] < 0:
            lines.append("年龄方向与 Twin-2K 相反，这个相关具有统计学意义。")
        else:
            lines.append("年龄梯度的相关没有达到 0.05，不能说模拟复现了人类随年龄变化的方向。")
    return lines


def _holm_column(rows: list[dict[str, Any]], source: str, target: str) -> None:
    indexed = [(index, row[source]) for index, row in enumerate(rows) if row.get(source) is not None]
    if not indexed:
        return
    adjusted = holm([p_value for _index, p_value in indexed])
    for (index, _p_value), p_adjusted in zip(indexed, adjusted):
        rows[index][target] = p_adjusted


def _simulated_people(
    responses: list[dict[str, Any]],
    personas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bands = {
        person["id"]: (person.get("demographics") or {}).get("age_band")
        for person in personas
    }
    people = []
    for response in responses:
        answers = {}
        for row in response.get("answers") or []:
            if row.get("ok") and row.get("value") is not None:
                answers[row["item_id"]] = row["value"]
        people.append(
            {
                "id": response["id"],
                "age_band": bands.get(response["id"]),
                "answers": answers,
            }
        )
    return people


def _bfi_scores(answers: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for scale, numbers in BFI_SCALES.items():
        keyed: list[int] = []
        for number in numbers:
            code = _as_code(answers.get(f"QID25_{number}"), 5)
            if code is None:
                keyed = []
                break
            keyed.append(keyed_score(code, number in BFI_REVERSE))
        if len(keyed) == len(numbers):
            scores[scale] = sum(keyed) / len(keyed)
    return scores


def _row_answers(row: Any) -> dict[str, Any]:
    raw = row._asdict() if hasattr(row, "_asdict") else dict(row)
    return {str(key): value for key, value in raw.items()}


def _column_codes(values: np.ndarray, n_options: int) -> tuple[np.ndarray, np.ndarray]:
    """Valid option codes, and a mask that keeps them aligned with the source rows."""
    codes: list[int] = []
    keep: list[bool] = []
    for value in values:
        code = _as_code(value, n_options)
        keep.append(code is not None)
        if code is not None:
            codes.append(code)
    return np.asarray(codes, dtype=int), np.asarray(keep, dtype=bool)


def _as_code(value: Any, n_options: int) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    code = int(number)
    if 1 <= code <= n_options:
        return code
    return None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    n = int(values.size)
    while start < n:
        stop = start + 1
        while stop < n and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _cohen_d(simulated: np.ndarray, human: np.ndarray) -> float:
    variance_left = float(simulated.var(ddof=1))
    variance_right = float(human.var(ddof=1))
    pooled = (
        (simulated.size - 1) * variance_left + (human.size - 1) * variance_right
    ) / (simulated.size + human.size - 2)
    if pooled <= 0:
        return 0.0
    return float((simulated.mean() - human.mean()) / math.sqrt(pooled))


def _invalid_answers(responses: list[dict[str, Any]]) -> int:
    total = 0
    for response in responses:
        for row in response.get("answers") or []:
            if not row.get("ok"):
                total += 1
    return total


def _answer_slots(responses: list[dict[str, Any]]) -> int:
    return sum(len(response.get("answers") or []) for response in responses)


def _composition_table(composition: dict[str, Any]) -> str:
    lines = [
        "| 年龄段 | 模拟人数 | 女 | 男 | Twin-2K 人数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for band in composition["bands"]:
        lines.append(
            f"| {band['label']} | {band['n_sim']} | {band['n_female']} | {band['n_male']} | {band['n_human']} |"
        )
    return "\n".join(lines)


def _band_table(bands: list[dict[str, Any]]) -> str:
    lines = [
        "| 年龄段 | 模拟 / 人类 | 平均 TVD | 分布 Holm 显著 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for band in bands:
        lines.append(
            f"| {band['label']} | {band['n_sim']} / {band['n_human']} | {_fmt(band['mean_tvd'])} | "
            f"{band['n_gof_holm']} / {band['n_tested']} |"
        )
    return "\n".join(lines)


def _instrument_table(instruments: list[dict[str, Any]]) -> str:
    lines = [
        "| 量表 | 题数 | 平均 TVD | 分布 Holm 显著 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in instruments:
        lines.append(
            f"| {row['label']} | {row['n_items']} | {_fmt(row['mean_tvd'])} | "
            f"{row['n_gof_holm']} / {row['n_items']} |"
        )
    return "\n".join(lines)


def _bfi_table(scales: list[dict[str, Any]]) -> str:
    lines = [
        "| 维度 | 人类 | 模拟 | 差值 | Cohen's d | 均分 p | 人类年龄 p | 模拟年龄 p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scales:
        lines.append(
            f"| {row['label']} | {_fmt(row['human_mean'])} | {_fmt(row['sim_mean'])} | "
            f"{_fmt_signed(row['diff'])} | {_fmt_signed(row['d'])} | {_fmt_p(row['p'])} | "
            f"{_fmt_p(row['p_age_human'])} | {_fmt_p(row['p_age_sim'])} |"
        )
    return "\n".join(lines)


def _bfi_band_table(scales: list[dict[str, Any]]) -> str:
    header = "| 维度 | " + " | ".join(LEVEL_TITLES[band] for band in AGE_BAND_ORDER) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in AGE_BAND_ORDER) + " |"
    lines = [
        "各年龄段均分，模拟在前，人类在后。",
        "",
        header,
        rule,
    ]
    for row in scales:
        cells = []
        for band in AGE_BAND_ORDER:
            cell = row["by_band"][band]
            cells.append(f"{_fmt(cell['sim_mean'])} / {_fmt(cell['human_mean'])}")
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _age_paragraph(age: dict[str, Any]) -> str:
    return (
        "题级检验看的是选项代码在四个年龄段之间是否相同。"
        f"梯度用的是每题「65 岁及以上均分 − 18–29 岁均分」，再求模拟与人类这两列差的相关。"
        f"相关 {_fmt(age['gradient_r'])}，置换 p = {_fmt_p(age['gradient_p'])}，"
        f"非零差的方向一致率 {_fmt_pct(age['sign_agreement'])}。"
    )


def _item_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        lines.append(
            f"- TVD {_fmt(row['tvd'])}，Holm p = {_fmt_p(row['p_gof_holm'])}，"
            f"模拟均分 {_fmt(row['sim_mean'])}，人类 {_fmt(row['human_mean'])}。"
            f"{row['label']} · {row['item_id']} · {row['prompt']}"
        )
    return lines or ["- 没有可比较的选择题。"]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def _short(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > 110:
        return flat[:110] + "…"
    return flat


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_signed(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}"


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"
