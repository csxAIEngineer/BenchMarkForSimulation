"""Build the twin-2k-50 interview: one record per person, fifty turns each.

The instrument is the 44-item Big Five Inventory plus six green-consumption
items. Gold answers stay on the Twin-2K records. Nemotron records use the same
questions and have no gold answer. This module does not call a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from popbench.dao.io import read_jsonl, write_jsonl
from popbench.dao.personas import sample_usa_adults
from popbench.dao.schema import TWIN2K50_INSTRUMENT, TWIN2K50_TURNS, InterviewRecord

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


def interview_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "interview" / "v0"


def build_instrument(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The 50 closed-ended turns, in interview order."""
    bfi = _question(catalog, "QID25")
    green = _question(catalog, "QID235")
    items: list[dict[str, Any]] = []
    items.extend(_matrix_items(bfi, stem="I see myself as someone who", family="bfi44"))
    items.extend(_matrix_items(green, stem=None, family="green6"))
    if len(items) != TWIN2K50_TURNS:
        raise InterviewError(f"expected {TWIN2K50_TURNS} interview items, found {len(items)}")
    for turn, item in enumerate(items, start=1):
        item["turn"] = turn
    return items


def panel_path(data_dir: Path, panel: str) -> Path:
    """Path of a built panel. `panel` is `twin2k` or `nemotron`."""
    root = interview_dir(data_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        name = (manifest.get("panels") or {}).get(panel)
        if name:
            return root / name
    fallback = {"twin2k": "twin2k_n50.jsonl", "nemotron": "nemotron_n50.jsonl"}
    if panel not in fallback:
        raise InterviewError("panel must be twin2k or nemotron")
    return root / fallback[panel]


def load_panel(path: Path) -> list[InterviewRecord]:
    """Read a JSONL panel into interview records."""
    try:
        rows = read_jsonl(path)
    except FileNotFoundError as exc:
        raise InterviewError(f"interview panel is missing: {path}. Run `popbench build`.") from exc
    return [InterviewRecord.from_dict(row) for row in rows]


def build_interview(data_dir: Path, n: int = 50, seed: int = 0) -> Path:
    """Write Twin-2K and Nemotron interview records, the human baseline, and an example."""
    from popbench.dao.twin2k import load_twin2k

    if n < 1:
        raise InterviewError("n must be positive")
    data = load_twin2k(data_dir, download=False)
    items = build_instrument(data.catalog)
    out = interview_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline, complete_pids = human_baseline(data.wave1_3, items)
    twin_records = _as_records(
        twin2k_records(
            data.wave1_3,
            data.wave1_3_labels,
            items,
            complete_pids,
            n=n,
            seed=seed,
        )
    )
    nemotron_records = _as_records(sample_nemotron_panel(data_dir, items, n=n, seed=seed))
    twin_name = f"twin2k_n{n}.jsonl"
    nemotron_name = f"nemotron_n{n}.jsonl"
    write_jsonl(out / twin_name, [record.to_dict() for record in twin_records])
    write_jsonl(out / nemotron_name, [record.to_dict() for record in nemotron_records])
    (out / "human_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (out / "example.md").write_text(
        _example_markdown(twin_records[0].to_dict(), nemotron_records[0].to_dict(), baseline)
    )
    manifest = {
        "instrument": TWIN2K50_INSTRUMENT,
        "n_turns": TWIN2K50_TURNS,
        "region": "US",
        "language": "en",
        "seed": seed,
        "n_people": n,
        "twin2k_complete_cases": len(complete_pids),
        "twin2k_example_pid": twin_records[0].persona.id,
        "nemotron_example_id": nemotron_records[0].persona.id,
        "comparison": [
            "Per-item option shares against the full Twin-2K sample",
            "Mean scale scores on the five Big Five dimensions and green consumption",
        ],
        "panels": {"twin2k": twin_name, "nemotron": nemotron_name},
        "files": [
            twin_name,
            nemotron_name,
            "human_baseline.json",
            "example.md",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def _as_records(rows: list[dict[str, Any]]) -> list[InterviewRecord]:
    return [InterviewRecord.from_dict(row) for row in rows]


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
    from popbench.dao.personas import PersonaError

    try:
        adults = sample_usa_adults(data_dir, n=n, seed=seed)
    except PersonaError as exc:
        raise InterviewError(str(exc)) from exc
    records = []
    for person in adults:
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
                "id": f"nemotron:{person.id}",
                "region": "US",
                "language": "en",
                "source": "nvidia/Nemotron-Personas-USA",
                "instrument": TWIN2K50_INSTRUMENT,
                "persona": {
                    "id": person.id,
                    "condition": "nemotron_usa",
                    "age": person.age,
                    "age_band": person.age_band,
                    "sex": person.sex,
                    "state": person.state,
                    "education_level": person.education_level,
                    "occupation": person.occupation,
                    "marital_status": person.marital_status,
                    "text": person.text,
                },
                "turns": turns,
            }
        )
    return records


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


