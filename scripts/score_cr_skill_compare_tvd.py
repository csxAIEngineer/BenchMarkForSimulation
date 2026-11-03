#!/usr/bin/env python3
"""Re-score cr-skill-compare runs with accuracy + mean per-question TVD.

Reads existing responses.jsonl only. Does not call any model API.

TVD follows popbench.evaluate.metrics.total_variation_distance:
    TVD(p, q) = 0.5 * sum_i |p_i - q_i|
after normalizing non-negative share vectors to sum to 1.

For twin2k_500 / bdm_lettuce: for each question_id, build empirical answer
share vectors from human gold.answer and model parsed.answer over the union of
observed labels among rows with both sides present; mean_tvd is the unweighted
mean of those per-question TVDs.

For twin2k_mega_study: gold.answer is null; extract ordered Answers from
gold.payload (SelectedByPosition / SelectedText / Values) and align to model
Q1..Qn (by index when lengths match, otherwise LCS on question types). Each
aligned sub-question becomes study:QK for population shares.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from popbench.evaluate.metrics import total_variation_distance

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cr-skill-compare-income50"
METHODS = ("atus_expanded", "atus_expanded_skill")
DATASETS = ("twin2k_500", "bdm_lettuce", "twin2k_mega_study")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def shares_from_labels(labels: list[str], vocabulary: list[str]) -> list[float]:
    counts = Counter(labels)
    n = len(labels)
    if n == 0:
        raise ValueError("empty labels")
    return [counts[label] / n for label in vocabulary]


def mean_tvd_by_item(
    pairs_by_item: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    item_rows: list[dict[str, Any]] = []
    tvds: list[float] = []
    for item_id in sorted(pairs_by_item):
        pairs = pairs_by_item[item_id]
        human = [h for h, _ in pairs]
        model = [m for _, m in pairs]
        vocab = sorted(set(human) | set(model))
        if not vocab:
            continue
        tvd = total_variation_distance(
            shares_from_labels(model, vocab),
            shares_from_labels(human, vocab),
        )
        tvds.append(tvd)
        item_rows.append(
            {
                "item_id": item_id,
                "n": len(pairs),
                "n_labels": len(vocab),
                "tvd": tvd,
            }
        )
    return {
        "mean_tvd": (sum(tvds) / len(tvds)) if tvds else None,
        "n_items_tvd": len(tvds),
        "n_pairs": sum(row["n"] for row in item_rows),
        "items": item_rows,
    }


def accuracy_from_pairs(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"n_scored": 0, "accuracy": None, "hits": 0}
    hits = sum(h == m for h, m in pairs)
    n = len(pairs)
    return {"n_scored": n, "accuracy": hits / n, "hits": hits}


def score_simple(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs_by_item: dict[str, list[tuple[str, str]]] = defaultdict(list)
    flat: list[tuple[str, str]] = []
    parse_errors = 0
    missing = 0
    for row in rows:
        if row.get("parse_error"):
            parse_errors += 1
        gold = row.get("gold") or {}
        human = norm_label(gold.get("answer"))
        pred = norm_label((row.get("parsed") or {}).get("answer"))
        if human is None or pred is None:
            missing += 1
            continue
        item = str(row.get("question_id") or "")
        pairs_by_item[item].append((human, pred))
        flat.append((human, pred))
    tvd = mean_tvd_by_item(pairs_by_item)
    acc = accuracy_from_pairs(flat)
    return {
        **acc,
        "missing_or_unscored": missing,
        "parse_errors": parse_errors,
        "n_rows": len(rows),
        "mean_tvd": tvd["mean_tvd"],
        "n_items_tvd": tvd["n_items_tvd"],
        "n_pairs_tvd": tvd["n_pairs"],
        "tvd_items": tvd["items"],
    }


def _canon_answer(ans: dict[str, Any]) -> str | None:
    if not isinstance(ans, dict):
        return None
    if "SelectedByPosition" in ans and ans["SelectedByPosition"] is not None:
        pos = ans["SelectedByPosition"]
        if isinstance(pos, list):
            return "#" + "#".join(str(int(x) if str(x).lstrip("-").isdigit() else x) for x in pos)
        return f"#{pos}"
    if "SelectedText" in ans and ans["SelectedText"] is not None:
        text = ans["SelectedText"]
        if isinstance(text, list):
            return "|" + "|".join(str(x).strip().casefold() for x in text) + "|"
        return str(text).strip().casefold()
    if "Values" in ans and ans["Values"] is not None:
        vals = ans["Values"]
        if isinstance(vals, list):
            return "~" + "~".join(str(x).strip() for x in vals)
        return f"~{vals}"
    return None


def _human_type(question: dict[str, Any]) -> str:
    raw = str(question.get("QuestionType") or "").strip().strip('"').upper()
    if raw in {"MC", "DB", "SAVR", "MAVR"}:
        return "choice"
    if raw in {"Matrix", "MATRIX"} or raw.startswith("MATRIX"):
        return "matrix"
    if raw in {"Slider", "SLIDER"} or "SLIDER" in raw:
        return "slider"
    if raw in {"TE", "TEXT"}:
        return "text"
    answers = question.get("Answers") or {}
    if isinstance(answers, dict):
        if answers.get("Values") is not None:
            return "slider"
        pos = answers.get("SelectedByPosition")
        if isinstance(pos, list):
            return "matrix"
        if pos is not None or answers.get("SelectedText") is not None:
            return "choice"
    return "other"


def _model_type(cell: dict[str, Any]) -> str:
    raw = str(cell.get("Question Type") or cell.get("QuestionType") or "").strip().casefold()
    if "matrix" in raw:
        return "matrix"
    if "slider" in raw:
        return "slider"
    if "text" in raw:
        return "text"
    if "choice" in raw or "mc" == raw:
        return "choice"
    answers = cell.get("Answers") if isinstance(cell.get("Answers"), dict) else cell
    if isinstance(answers, dict):
        if answers.get("Values") is not None:
            return "slider"
        if isinstance(answers.get("SelectedByPosition"), list) or isinstance(
            answers.get("SelectedText"), list
        ):
            return "matrix"
    return "choice"


def _extract_human_answers(payload_text: str) -> list[tuple[str, str]]:
    payload = json.loads(payload_text)
    questions: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            answers = obj.get("Answers")
            if isinstance(answers, dict) and any(
                key in answers for key in ("SelectedByPosition", "SelectedText", "Values")
            ):
                questions.append(obj)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(payload)
    out: list[tuple[str, str]] = []
    for question in questions:
        label = _canon_answer(question.get("Answers") or {})
        if label is None:
            continue
        out.append((_human_type(question), label))
    return out


def _extract_model_answers(parsed_answer: Any) -> list[tuple[str, str, str]]:
    if isinstance(parsed_answer, str):
        try:
            parsed_answer = json.loads(parsed_answer)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed_answer, dict):
        return []
    keys = sorted(
        (key for key in parsed_answer if re.fullmatch(r"Q\d+", str(key))),
        key=lambda key: int(str(key)[1:]),
    )
    out: list[tuple[str, str, str]] = []
    for key in keys:
        cell = parsed_answer[key]
        if not isinstance(cell, dict):
            continue
        answers = cell.get("Answers") if isinstance(cell.get("Answers"), dict) else cell
        label = _canon_answer(answers if isinstance(answers, dict) else {})
        if label is None:
            continue
        out.append((str(key), _model_type(cell), label))
    return out


def _align(
    human: list[tuple[str, str]],
    model: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Return list of (item_key, human_label, model_label)."""
    if not human or not model:
        return []
    if len(human) == len(model):
        return [
            (model[index][0], human[index][1], model[index][2])
            for index in range(len(human))
        ]
    # LCS alignment on types
    n_h, n_m = len(human), len(model)
    dp = [[0] * (n_m + 1) for _ in range(n_h + 1)]
    for i in range(n_h - 1, -1, -1):
        for j in range(n_m - 1, -1, -1):
            if human[i][0] == model[j][1]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    aligned: list[tuple[str, str, str]] = []
    i = j = 0
    while i < n_h and j < n_m:
        if human[i][0] == model[j][1]:
            aligned.append((model[j][0], human[i][1], model[j][2]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return aligned


def score_mega(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs_by_item: dict[str, list[tuple[str, str]]] = defaultdict(list)
    flat: list[tuple[str, str]] = []
    parse_errors = 0
    missing = 0
    aligned_cases = 0
    length_mismatch_cases = 0
    no_align_cases = 0
    for row in rows:
        if row.get("parse_error"):
            parse_errors += 1
        gold = row.get("gold") or {}
        payload = gold.get("payload")
        parsed_answer = (row.get("parsed") or {}).get("answer")
        if not payload or parsed_answer is None:
            missing += 1
            continue
        try:
            human = _extract_human_answers(str(payload))
        except Exception:
            missing += 1
            continue
        model = _extract_model_answers(parsed_answer)
        if not human or not model:
            missing += 1
            continue
        if len(human) != len(model):
            length_mismatch_cases += 1
        aligned = _align(human, model)
        if not aligned:
            no_align_cases += 1
            missing += 1
            continue
        aligned_cases += 1
        study = str(row.get("question_id") or "").split(":", 1)[0]
        for qkey, human_label, model_label in aligned:
            h = norm_label(human_label)
            m = norm_label(model_label)
            if h is None or m is None:
                continue
            item = f"{study}:{qkey}"
            pairs_by_item[item].append((h, m))
            flat.append((h, m))
    tvd = mean_tvd_by_item(pairs_by_item)
    acc = accuracy_from_pairs(flat)
    return {
        **acc,
        "missing_or_unscored": missing,
        "parse_errors": parse_errors,
        "n_rows": len(rows),
        "mean_tvd": tvd["mean_tvd"],
        "n_items_tvd": tvd["n_items_tvd"],
        "n_pairs_tvd": tvd["n_pairs"],
        "aligned_cases": aligned_cases,
        "length_mismatch_cases": length_mismatch_cases,
        "no_align_cases": no_align_cases,
        "tvd_items": tvd["items"],
        "gold_source": "gold.payload Answers (SelectedByPosition/SelectedText/Values)",
        "alignment": "index zip when lengths match else LCS on question types",
    }


def score_run(dataset: str, method: str) -> dict[str, Any]:
    path = RUN / dataset / method / "responses.jsonl"
    rows = read_jsonl(path)
    if dataset == "twin2k_mega_study":
        metrics = score_mega(rows)
    else:
        metrics = score_simple(rows)
    return {
        "dataset": dataset,
        "method": method,
        "path": str(path),
        **metrics,
    }


def main() -> None:
    details: list[dict[str, Any]] = []
    compare: list[dict[str, Any]] = []
    for dataset in DATASETS:
        by_method: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            metrics = score_run(dataset, method)
            # keep large item tables in detail file only
            slim = {k: v for k, v in metrics.items() if k != "tvd_items"}
            details.append(metrics)
            by_method[method] = slim
            summary_path = RUN / dataset / method / "run_summary_tvd.json"
            summary_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n")
        base = by_method["atus_expanded"]
        skill = by_method["atus_expanded_skill"]
        def delta(a, b):
            if a is None or b is None:
                return None
            return round(b - a, 6)

        compare.append(
            {
                "dataset": dataset,
                "atus_expanded_accuracy": base.get("accuracy"),
                "atus_expanded_skill_accuracy": skill.get("accuracy"),
                "delta_accuracy_skill_minus_base": delta(base.get("accuracy"), skill.get("accuracy")),
                "atus_expanded_mean_tvd": base.get("mean_tvd"),
                "atus_expanded_skill_mean_tvd": skill.get("mean_tvd"),
                "delta_tvd_skill_minus_base": delta(base.get("mean_tvd"), skill.get("mean_tvd")),
                "n_scored_base": base.get("n_scored"),
                "n_scored_skill": skill.get("n_scored"),
                "n_items_tvd_base": base.get("n_items_tvd"),
                "n_items_tvd_skill": skill.get("n_items_tvd"),
                "note": (
                    "lower TVD = closer to human answer shares; "
                    "mean over questions with both human and model labels"
                ),
            }
        )

    formula = {
        "metric": "total_variation_distance",
        "formula": "0.5 * sum_i |p_i - q_i| on normalized share vectors",
        "aggregation": (
            "per-question (or per Mega study:Qi) empirical answer distributions "
            "across subjects; report unweighted mean_tvd"
        ),
        "source": "popbench.evaluate.metrics.total_variation_distance (restored from main)",
        "no_api_calls": True,
    }
    out = {
        "formula": formula,
        "compare": compare,
        "methods": list(METHODS),
        "run_dir": str(RUN),
    }
    (RUN / "compare_vs_human_tvd.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    # Extend existing accuracy compare without deleting fields.
    legacy_path = RUN / "compare_vs_human.json"
    legacy = json.loads(legacy_path.read_text()) if legacy_path.is_file() else []
    legacy_by_dataset = {row["dataset"]: row for row in legacy}
    extended = []
    for row in compare:
        old = dict(legacy_by_dataset.get(row["dataset"]) or {})
        old.update(
            {
                "dataset": row["dataset"],
                "atus_expanded_accuracy": row["atus_expanded_accuracy"],
                "atus_expanded_skill_accuracy": row["atus_expanded_skill_accuracy"],
                "delta_skill_minus_base": row["delta_accuracy_skill_minus_base"],
                "n_scored_base": row["n_scored_base"],
                "n_scored_skill": row["n_scored_skill"],
                "atus_expanded_mean_tvd": row["atus_expanded_mean_tvd"],
                "atus_expanded_skill_mean_tvd": row["atus_expanded_skill_mean_tvd"],
                "delta_tvd_skill_minus_base": row["delta_tvd_skill_minus_base"],
                "n_items_tvd_base": row["n_items_tvd_base"],
                "n_items_tvd_skill": row["n_items_tvd_skill"],
            }
        )
        extended.append(old)
    legacy_path.write_text(json.dumps(extended, ensure_ascii=False, indent=2) + "\n")

    # Compact detail without dumping every item into stdout later
    detail_out = []
    for row in details:
        detail_out.append({k: v for k, v in row.items() if k != "tvd_items"})
    (RUN / "tvd_score_details.json").write_text(
        json.dumps({"formula": formula, "runs": detail_out}, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
