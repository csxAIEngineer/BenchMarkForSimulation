"""Turn the Twin-2K personality battery into a 50-turn interview.

The instrument is the 44-item Big Five Inventory plus six green-consumption
items. Gold answers stay the human option text. Nemotron records use the same
questions and have no gold answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from popbench.metrics import jensen_shannon, total_variation_distance

OPTIONS = (
    "Disagree strongly",
    "Disagree a little",
    "Neither agree nor disagree",
    "Agree a little",
    "Agree strongly",
)

# Item numbers are 1-based positions inside QID25, matching the BFI-44 key.
BFI_SCALES: dict[str, tuple[int, ...]] = {
    "extraversion": (1, 6, 11, 16, 21, 26, 31, 36),
    "agreeableness": (2, 7, 12, 17, 22, 27, 32, 37, 42),
    "conscientiousness": (3, 8, 13, 18, 23, 28, 33, 38, 43),
    "neuroticism": (4, 9, 14, 19, 24, 29, 34, 39),
    "openness": (5, 10, 15, 20, 25, 30, 35, 40, 41, 44),
}
BFI_REVERSE = frozenset({2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43})
SCALE_ZH = {
    "extraversion": "外向性",
    "agreeableness": "宜人性",
    "conscientiousness": "尽责性",
    "neuroticism": "神经质",
    "openness": "开放性",
    "green_values": "绿色消费",
}
DEMOGRAPHICS = (
    ("QID11", "Region"),
    ("QID12", "Sex assigned at birth"),
    ("QID13", "Age"),
    ("QID14", "Education"),
    ("QID15", "Race or origin"),
    ("QID17", "Marital status"),
    ("QID20", "Party"),
    ("QID21", "Family income"),
    ("QID22", "Political views"),
    ("QID24", "Employment"),
)
AGE_BANDS = (
    (18, 29, "18-29"),
    (30, 49, "30-49"),
    (50, 64, "50-64"),
    (65, 200, "65+"),
)
NARRATIVE_FIELDS = (
    "persona",
    "professional_persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
)


class InterviewError(RuntimeError):
    """The interview export could not be built from the local cache."""


def keyed_score(code: int, reverse: bool) -> int:
    """Map a 1–5 agreement code onto the scale direction."""
    if code < 1 or code > 5:
        raise ValueError(f"agreement code must be 1–5, got {code}")
    return 6 - code if reverse else code


def scale_means(item_scores: dict[str, list[int]]) -> dict[str, float]:
    """Mean of keyed item scores for each scale present in `item_scores`."""
    means = {}
    for scale, scores in item_scores.items():
        if scores:
            means[scale] = sum(scores) / len(scores)
    return means


def allocate_quotas(weights: dict[str, int], n: int) -> dict[str, int]:
    """Spread `n` seats across keys in proportion to positive weights."""
    usable = {key: weight for key, weight in weights.items() if weight > 0}
    total = sum(usable.values())
    if n < 1 or total < n:
        raise ValueError("not enough weighted rows to fill the quota")
    raw = {key: n * weight / total for key, weight in usable.items()}
    base = {key: int(value) for key, value in raw.items()}
    leftover = n - sum(base.values())
    ranked = sorted(usable, key=lambda key: (raw[key] - base[key], usable[key]), reverse=True)
    for key in ranked:
        if leftover == 0:
            break
        if base[key] < usable[key]:
            base[key] += 1
            leftover -= 1
    if leftover:
        raise ValueError("quota allocation could not fill every seat")
    return {key: count for key, count in base.items() if count}


def interview_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "interview" / "v0"


def build_instrument(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The 50 closed-ended turns, in interview order."""
    bfi = _question(catalog, "QID25")
    green = _question(catalog, "QID235")
    items: list[dict[str, Any]] = []
    items.extend(_matrix_items(bfi, stem="I see myself as someone who", family="bfi44"))
    items.extend(_matrix_items(green, stem=None, family="green6"))
    if len(items) != 50:
        raise InterviewError(f"expected 50 interview items, found {len(items)}")
    for turn, item in enumerate(items, start=1):
        item["turn"] = turn
    return items


def build_interview(data_dir: Path, n: int = 50, seed: int = 0) -> Path:
    """Write the 50-person Twin-2K interview, one Nemotron panel, and examples."""
    from popbench.twin2k import load_twin2k

    if n < 1:
        raise InterviewError("n must be positive")
    data = load_twin2k(data_dir, download=False)
    items = build_instrument(data.catalog)
    out = interview_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline, complete_pids = human_baseline(data.wave1_3, items)
    twin_records = twin2k_records(
        data.wave1_3,
        data.wave1_3_labels,
        items,
        complete_pids,
        n=n,
        seed=seed,
    )
    nemotron_records = sample_nemotron_panel(data_dir, items, n=n, seed=seed)
    _write_jsonl(out / "twin2k_n50.jsonl", twin_records)
    _write_jsonl(out / "nemotron_n50.jsonl", nemotron_records)
    (out / "human_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (out / "example.md").write_text(_example_markdown(twin_records[0], nemotron_records[0], baseline))
    manifest = {
        "instrument": "bfi44_plus_green6",
        "n_turns": 50,
        "region": "US",
        "language": "en",
        "seed": seed,
        "n_people": n,
        "twin2k_complete_cases": len(complete_pids),
        "twin2k_example_pid": twin_records[0]["persona"]["id"],
        "nemotron_example_id": nemotron_records[0]["persona"]["id"],
        "comparison": [
            "Per-item option shares against the full Twin-2K sample",
            "Mean scale scores on the five Big Five dimensions and green consumption",
        ],
        "files": [
            "twin2k_n50.jsonl",
            "nemotron_n50.jsonl",
            "human_baseline.json",
            "example.md",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def human_baseline(
    numeric: pd.DataFrame,
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Option shares and scale means on every Twin-2K respondent."""
    codes = _code_frame(numeric, items)
    complete = codes.notna().all(axis=1)
    item_rows = []
    for item in items:
        column = codes[item["item_id"]]
        observed = column.dropna().astype(int)
        shares = []
        for code, _option in enumerate(OPTIONS, start=1):
            shares.append(float((observed == code).mean()) if len(observed) else 0.0)
        item_rows.append(
            {
                "item_id": item["item_id"],
                "turn": item["turn"],
                "scale": item["scale"],
                "n": int(len(observed)),
                "shares": shares,
                "mean_code": float(observed.mean()) if len(observed) else None,
            }
        )
    scale_rows = _population_scales(codes)
    baseline = {
        "n_respondents": int(len(numeric)),
        "n_complete": int(complete.sum()),
        "options": list(OPTIONS),
        "items": item_rows,
        "scales": scale_rows,
    }
    pids = numeric.loc[complete, "pid"].astype(str).tolist()
    return baseline, pids


def twin2k_records(
    numeric: pd.DataFrame,
    labels: pd.DataFrame,
    items: list[dict[str, Any]],
    complete_pids: list[str],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(complete_pids) < n:
        raise InterviewError(
            f"only {len(complete_pids)} Twin-2K respondents answered all 50 items"
        )
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(np.array(complete_pids), size=n, replace=False).tolist())
    by_pid = numeric.set_index(numeric["pid"].astype(str), drop=False)
    label_by_pid = labels.set_index(labels["pid"].astype(str), drop=False)
    records = []
    for pid in chosen:
        row = by_pid.loc[pid]
        label_row = label_by_pid.loc[pid]
        turns = []
        scores: dict[str, list[int]] = {scale: [] for scale in SCALE_ZH}
        for item in items:
            code = int(row[item["column"]])
            keyed = keyed_score(code, item["reverse"])
            scores[item["scale"]].append(keyed)
            label_text = label_row[item["column"]]
            turns.append(
                {
                    "turn": item["turn"],
                    "item_id": item["item_id"],
                    "scale": item["scale"],
                    "prompt": item["prompt"],
                    "options": list(OPTIONS),
                    "gold_answer": OPTIONS[code - 1],
                    "gold_code": code,
                    "label_text": None if pd.isna(label_text) else str(label_text),
                    "reverse": item["reverse"],
                }
            )
        records.append(
            {
                "id": f"twin2k:{pid}",
                "region": "US",
                "language": "en",
                "source": "LLM-Digital-Twin/Twin-2K-500",
                "instrument": "bfi44_plus_green6",
                "persona": {
                    "id": pid,
                    "condition": "demographics_only",
                    "text": _twin_persona(label_row),
                },
                "turns": turns,
                "scale_scores": {scale: sum(values) / len(values) for scale, values in scores.items()},
            }
        )
    return records


def sample_nemotron_panel(
    data_dir: Path,
    items: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Adults, quota-sampled on age band and sex. State follows the shard."""
    import pyarrow.parquet as pq

    from popbench.personas import nemotron_shard_name

    path = Path(data_dir) / "nemotron_usa" / nemotron_shard_name(0)
    if not path.is_file():
        raise InterviewError(f"Nemotron shard is missing: {path}")
    columns = [
        "uuid",
        "age",
        "sex",
        "state",
        "education_level",
        "occupation",
        "marital_status",
        *NARRATIVE_FIELDS,
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    adults = frame.loc[frame["age"] >= 18].copy()
    adults["age_band"] = adults["age"].map(_age_band)
    adults = adults.loc[adults["age_band"].notna() & adults["sex"].isin(["Female", "Male"])]
    adults["stratum"] = adults["age_band"] + "|" + adults["sex"]
    weights = adults["stratum"].value_counts().to_dict()
    quotas = allocate_quotas(weights, n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum, count in quotas.items():
        pool = adults.index[adults["stratum"] == stratum].to_numpy()
        picked.extend(rng.choice(pool, size=count, replace=False).tolist())
    panel = adults.loc[picked].sort_values(["stratum", "uuid"])
    records = []
    for row in panel.itertuples(index=False):
        persona_id = str(row.uuid)
        turns = [
            {
                "turn": item["turn"],
                "item_id": item["item_id"],
                "scale": item["scale"],
                "prompt": item["prompt"],
                "options": list(OPTIONS),
                "gold_answer": None,
                "gold_code": None,
                "reverse": item["reverse"],
            }
            for item in items
        ]
        records.append(
            {
                "id": f"nemotron:{persona_id}",
                "region": "US",
                "language": "en",
                "source": "nvidia/Nemotron-Personas-USA",
                "instrument": "bfi44_plus_green6",
                "persona": {
                    "id": persona_id,
                    "condition": "nemotron_usa",
                    "age": int(row.age),
                    "age_band": row.age_band,
                    "sex": row.sex,
                    "state": row.state,
                    "education_level": row.education_level,
                    "occupation": row.occupation,
                    "marital_status": row.marital_status,
                    "text": _nemotron_persona(row),
                },
                "turns": turns,
            }
        )
    return records


def turns_user_message(turn: dict[str, Any]) -> str:
    """One interview turn. Gold answers are not included."""
    lines = [
        f"Question {turn['turn']} of 50.",
        turn["prompt"],
        "",
        "Options:",
    ]
    for index, option in enumerate(turn["options"], start=1):
        lines.append(f"{index}. {option}")
    lines.extend(
        [
            "",
            'Reply with JSON only: {"answer": "<one option, exactly as written>", '
            '"rationale": "<one short sentence>"}',
        ]
    )
    return "\n".join(lines)


def system_message(persona_text: str) -> str:
    return (
        "You are taking a survey as the person described below. "
        "Answer as that person would, and stay consistent across questions. "
        "Choose exactly one of the listed options.\n\n"
        f"{persona_text}"
    )


def score_simulation(
    responses: list[dict[str, Any]],
    baseline: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare simulated option shares and scale means with the human baseline."""
    by_item: dict[str, list[int]] = {item["item_id"]: [] for item in items}
    person_scales: dict[str, list[float]] = {scale: [] for scale in SCALE_ZH}
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

    item_stats = []
    tvds = []
    jss = []
    code_mae = []
    human_by_id = {row["item_id"]: row for row in baseline["items"]}
    for item in items:
        human = human_by_id[item["item_id"]]
        codes = by_item[item["item_id"]]
        shares = [codes.count(code) / len(codes) for code in range(1, 6)] if codes else [0.0] * 5
        tvd = total_variation_distance(shares, human["shares"]) if codes else None
        js = jensen_shannon(shares, human["shares"]) if codes else None
        mean_code = sum(codes) / len(codes) if codes else None
        if tvd is not None:
            tvds.append(tvd)
            jss.append(js)
        if mean_code is not None and human["mean_code"] is not None:
            code_mae.append(abs(mean_code - human["mean_code"]))
        item_stats.append(
            {
                "item_id": item["item_id"],
                "turn": item["turn"],
                "scale": item["scale"],
                "n": len(codes),
                "tvd": tvd,
                "js": js,
                "simulated_mean_code": mean_code,
                "human_mean_code": human["mean_code"],
            }
        )
    scales = {}
    for scale, values in person_scales.items():
        human_mean = baseline["scales"][scale]["mean"]
        simulated_mean = sum(values) / len(values) if values else None
        scales[scale] = {
            "n": len(values),
            "simulated_mean": simulated_mean,
            "human_mean": human_mean,
            "absolute_error": None
            if simulated_mean is None
            else abs(simulated_mean - human_mean),
        }
    return {
        "n_people": len(responses),
        "n_answers": n_answers,
        "n_invalid": n_invalid,
        "mean_tvd": sum(tvds) / len(tvds) if tvds else None,
        "mean_js": sum(jss) / len(jss) if jss else None,
        "mean_abs_code_error": sum(code_mae) / len(code_mae) if code_mae else None,
        "items": item_stats,
        "scales": scales,
    }


def render_result_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """One page: sample, persona condition, item distance, scale distance."""
    lines = [
        "# Nemotron 多轮访谈 vs Twin-2K 人类分布",
        "",
        f"- 卷子：大五人格 44 题 + 绿色消费 6 题，共 50 轮",
        f"- 模拟人数：{summary['n_people']}",
        f"- 人物：Nemotron-Personas-USA，18 岁以上，按年龄段和性别配额",
        f"- 模型：{meta.get('model', 'unknown')}",
        f"- 人类参照：Twin-2K 全样本在同一套题上的选项份额和量表均分",
        f"- 无效回答：{summary['n_invalid']} / {summary['n_answers']}",
        "",
        "Nemotron 人物不是 Twin-2K 受访者，所以这里比的是分布，不是同一个人的逐题命中。",
        "",
        "## 量表均分（1–5，反向题已翻转）",
        "",
        "| 量表 | 人类均值 | 模拟均值 | 绝对差 |",
        "| --- | --- | --- | --- |",
    ]
    for scale, row in summary["scales"].items():
        lines.append(
            "| {name} | {human:.2f} | {sim} | {err} |".format(
                name=SCALE_ZH[scale],
                human=row["human_mean"],
                sim="—" if row["simulated_mean"] is None else f"{row['simulated_mean']:.2f}",
                err="—" if row["absolute_error"] is None else f"{row['absolute_error']:.2f}",
            )
        )
    lines.extend(
        [
            "",
            "## 逐题分布",
            "",
            f"- 50 题平均总变异距离（TVD）：{_fmt(summary['mean_tvd'])}",
            f"- 50 题平均 JS 散度（bit）：{_fmt(summary['mean_js'])}",
            f"- 选项编码的平均绝对差：{_fmt(summary['mean_abs_code_error'])}",
            "",
            "TVD 为 0 表示选项份额相同，为 1 表示没有重叠。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _question(catalog: list[dict[str, Any]], question_id: str) -> dict[str, Any]:
    matches = [entry for entry in catalog if entry.get("QuestionID") == question_id]
    if len(matches) != 1:
        raise InterviewError(f"expected one catalog entry for {question_id}")
    return matches[0]


def _matrix_items(entry: dict[str, Any], stem: str | None, family: str) -> list[dict[str, Any]]:
    rows = entry.get("Rows") or []
    columns = entry.get("csv_columns") or []
    if len(rows) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} rows and csv columns differ")
    if list(entry.get("Columns") or []) != list(OPTIONS):
        raise InterviewError(f"{entry['QuestionID']} does not use the 5-point agreement scale")
    items = []
    for number, (statement, column) in enumerate(zip(rows, columns), start=1):
        if family == "bfi44":
            scale = _bfi_scale(number)
            reverse = number in BFI_REVERSE
            prompt = f"{stem} {statement[0].lower()}{statement[1:]}."
        else:
            scale = "green_values"
            reverse = False
            prompt = statement if statement.endswith(".") else statement + "."
        items.append(
            {
                "item_id": column,
                "column": column,
                "scale": scale,
                "reverse": reverse,
                "prompt": prompt,
                "statement": statement,
            }
        )
    return items


def _bfi_scale(number: int) -> str:
    for scale, numbers in BFI_SCALES.items():
        if number in numbers:
            return scale
    raise InterviewError(f"BFI item {number} is not on a scale")


def _code_frame(numeric: pd.DataFrame, items: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [item["column"] for item in items]
    frame = numeric[columns].apply(pd.to_numeric, errors="coerce")
    frame.columns = [item["item_id"] for item in items]
    valid = frame.isna() | frame.isin([1, 2, 3, 4, 5])
    if not bool(valid.all().all()):
        raise InterviewError("interview columns contain codes outside 1–5")
    return frame


def _population_scales(codes: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    totals: dict[str, list[float]] = {scale: [] for scale in SCALE_ZH}
    for row in codes.itertuples(index=False):
        values = dict(zip(codes.columns, row))
        bucket: dict[str, list[int]] = {scale: [] for scale in SCALE_ZH}
        for item_id, code in values.items():
            if pd.isna(code):
                continue
            scale, reverse = _item_key(item_id)
            bucket[scale].append(keyed_score(int(code), reverse))
        for scale, scores in bucket.items():
            if scores:
                totals[scale].append(sum(scores) / len(scores))
    summary = {}
    for scale, values in totals.items():
        series = pd.Series(values, dtype="float64")
        summary[scale] = {
            "n": int(series.size),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }
    return summary


def _item_key(item_id: str) -> tuple[str, bool]:
    if item_id.startswith("QID235_"):
        return "green_values", False
    number = int(item_id.split("_", 1)[1])
    return _bfi_scale(number), number in BFI_REVERSE


def _twin_persona(label_row: pd.Series) -> str:
    lines = ["This person is a U.S. adult in the Twin-2K sample."]
    for column, label in DEMOGRAPHICS:
        value = label_row[column] if column in label_row.index else None
        shown = "unknown" if value is None or pd.isna(value) else str(value)
        lines.append(f"{label}: {shown}")
    return "\n".join(lines)


def _nemotron_persona(row: Any) -> str:
    lines = [
        f"Age: {int(row.age)}",
        f"Sex: {row.sex}",
        f"State: {row.state}",
        f"Education: {row.education_level}",
        f"Occupation: {row.occupation}",
        f"Marital status: {row.marital_status}",
        "",
    ]
    for field in NARRATIVE_FIELDS:
        text = getattr(row, field)
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
            lines.append("")
    return "\n".join(lines).strip()


def _age_band(age: int) -> str | None:
    for low, high, name in AGE_BANDS:
        if low <= int(age) <= high:
            return name
    return None


def _example_markdown(
    twin: dict[str, Any],
    nemotron: dict[str, Any],
    baseline: dict[str, Any],
) -> str:
    lines = [
        "# 多轮访谈样例",
        "",
        "卷子共 50 题：大五人格 44 题（QID25）加上绿色消费 6 题（QID235）。",
        "个人主义 16 题和花钱习惯 1 题没有放进来，这样访谈停在 50 轮。",
        "",
        "比较有两层，都对着 Twin-2K 全样本，而不是对着这一个受访者：",
        "",
        "- 逐题：模拟人群在五个同意档上的份额，和人类份额的总变异距离。",
        "- 量表：外向性、宜人性、尽责性、神经质、开放性、绿色消费的 1–5 分均值。反向题先翻转再平均。",
        "",
        "多轮指的是模型一题一题答，并且看得到自己前面的选择。标准答案是真人当时选的那一档。",
        "",
        "## 人类全样本的量表均分",
        "",
    ]
    for scale, row in baseline["scales"].items():
        lines.append(
            f"- {SCALE_ZH[scale]}：{row['mean']:.2f}（标准差 {row['std']:.2f}，n={row['n']}）"
        )
    lines.extend(
        [
            "",
            f"50 题全部作答的 Twin-2K 受访者：{baseline['n_complete']} / {baseline['n_respondents']}。",
            "",
            f"## Twin-2K 样例受访者 {twin['persona']['id']}",
            "",
            "人物卡只有人口学，不含这 50 题的答案。答案写在每一轮下面。",
            "",
            "```",
            twin["persona"]["text"],
            "```",
            "",
            "这个人的量表分：",
            "",
        ]
    )
    for scale, value in twin["scale_scores"].items():
        lines.append(f"- {SCALE_ZH[scale]}：{value:.2f}")
    lines.append("")
    for turn in twin["turns"]:
        lines.append(f"### 第 {turn['turn']} 轮 · {turn['item_id']} · {SCALE_ZH[turn['scale']]}")
        lines.append("")
        lines.append(turn["prompt"])
        lines.append("")
        lines.append(f"标准答案：{turn['gold_answer']}")
        lines.append("")
    person = nemotron["persona"]
    lines.extend(
        [
            f"## Nemotron 样例人物 {person['id']}",
            "",
            "同一套 50 题。这条记录没有标准答案，用来给模型扮演。",
            f"年龄段 {person['age_band']}，性别 {person['sex']}，州 {person['state']}。",
            "",
            "```",
            person["text"],
            "```",
            "",
        ]
    )
    for turn in nemotron["turns"]:
        lines.append(f"### 第 {turn['turn']} 轮 · {turn['item_id']}")
        lines.append("")
        lines.append(turn["prompt"])
        lines.append("")
    lines.append("模型回答要落在这五档里：")
    lines.append("")
    for option in OPTIONS:
        lines.append(f"- {option}")
    lines.append("")
    return "\n".join(lines)


def run_nemotron_interview(data_dir: Path, run_dir: Path, workers: int = 4) -> Path:
    """Ask the configured model to answer the Nemotron panel, then score it."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from popbench.simulate import chat, model_settings_from_env, parse_choice, response_cache_key

    try:
        settings = model_settings_from_env()
    except RuntimeError as exc:
        raise InterviewError(str(exc)) from exc
    panel_path = interview_dir(data_dir) / "nemotron_n50.jsonl"
    baseline_path = interview_dir(data_dir) / "human_baseline.json"
    if not panel_path.is_file() or not baseline_path.is_file():
        raise InterviewError("interview files are missing; build them first")
    records = [json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
    baseline = json.loads(baseline_path.read_text())
    items = records[0]["turns"]
    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    finished = 0
    lock = threading.Lock()

    def _one(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal finished
        result = _answer_record(settings, record, cache_dir, chat, parse_choice, response_cache_key, time)
        with lock:
            finished += 1
            print(f"answered {finished}/{len(records)} {record['id']}", flush=True)
        return result

    workers = max(1, min(workers, len(records)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_one, records))
    response_path = run_dir / "responses.jsonl"
    _write_jsonl(response_path, responses)
    summary = score_simulation(responses, baseline, items)
    summary["model"] = settings.model
    summary["base_url"] = settings.base_url
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "report.md").write_text(
        render_result_markdown(summary, {"model": settings.model})
    )
    return run_dir


def _answer_record(settings, record, cache_dir, chat, parse_choice, response_cache_key, time) -> dict:
    messages = [{"role": "system", "content": system_message(record["persona"]["text"])}]
    answers = []
    for turn in record["turns"]:
        user = turns_user_message(turn)
        cache_path = cache_dir / f"{response_cache_key(settings.model, record['id'], turn['item_id'])}.json"
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text())["raw"]
        else:
            raw = _chat_with_retry(settings, messages + [{"role": "user", "content": user}], chat, time)
            cache_path.write_text(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
        try:
            choice = parse_choice(raw, tuple(turn["options"]))
            answers.append(
                {
                    "item_id": turn["item_id"],
                    "answer": choice.answer,
                    "code": turn["options"].index(choice.answer) + 1,
                    "rationale": choice.rationale,
                    "raw": raw,
                    "ok": True,
                }
            )
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": raw})
        except ValueError as exc:
            answers.append(
                {
                    "item_id": turn["item_id"],
                    "answer": None,
                    "code": None,
                    "raw": raw,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return {"id": record["id"], "answers": answers}


def _chat_with_retry(settings, messages, chat, time) -> str:
    import httpx

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return chat(settings, messages)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code not in {429, 500, 502, 503, 504}:
                raise InterviewError(f"model request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(2**attempt)
    raise InterviewError(f"model request failed after retries: {last_error}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

