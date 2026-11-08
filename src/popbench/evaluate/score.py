"""Compare local simulation responses with held-out human outcomes.

Reads files that prepro and simulate already wrote. Does not call a model.
The report is written only where the caller asks, under a gitignored directory.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from popbench.evaluate.metrics import (
    accuracy,
    distribution_top1_accuracy,
    fisher_z_mean_correlation,
    glass_delta,
    individual_accuracy,
    ordinal_mae,
    pearson,
    sd_ratio,
    total_variation_distance,
)
from popbench.evaluate.person import score_pricing_persons
from popbench.prepro.spec import BDM_OUTCOME_COLUMNS

WTP_COLUMNS = tuple(column for column in BDM_OUTCOME_COLUMNS if column != "conclusion_purchased_product")
WTP_SPAN = 10.0


def score_run(root: Path, config: Mapping[str, object]) -> dict[str, object]:
    """Join responses to outcomes and write one JSON report."""
    responses_root = _path(root, config["responses_dir"])
    outcomes_dir = _path(root, config["outcomes_dir"])
    output = _path(root, config["output"])
    model = str(config["model"])
    datasets = _string_list(config.get("datasets"), "datasets")
    report: dict[str, object] = {"datasets": {}, "model": model, "provider": "cr"}
    datasets_out: dict[str, object] = {}
    for dataset in datasets:
        responses = _read_jsonl(responses_root / model / dataset / "responses.jsonl")
        outcomes = _read_jsonl(outcomes_dir / dataset / "outcomes.jsonl")
        datasets_out[dataset] = score_dataset(dataset, responses, outcomes)
    report["datasets"] = datasets_out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score_dataset(
    dataset: str,
    responses: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    human = _human_index(outcome_rows)
    paired: dict[str, list[tuple[str, str]]] = defaultdict(list)
    numeric: dict[str, list[tuple[float, float]]] = defaultdict(list)
    parse_errors = 0
    scored = 0
    for row in responses:
        question_id = str(row.get("question_id") or "")
        subject_id = str(row.get("subject_id") or "")
        if row.get("parse_error"):
            parse_errors += 1
            continue
        key = (subject_id, question_id)
        if key not in human:
            raise ValueError(f"{dataset} response {subject_id} {question_id} has no held-out answer")
        predicted = str(row.get("answer") or "").strip()
        held = human[key]
        if question_id in WTP_COLUMNS:
            left = _number(predicted)
            right = _number(held)
            if left is None or right is None:
                parse_errors += 1
                continue
            numeric[question_id].append((right, left))
        else:
            paired[question_id].append((held, predicted))
        scored += 1
    categorical = mean_tvd_by_item(paired)
    result: dict[str, object] = {
        "categorical": categorical,
        "n_parse_errors": parse_errors,
        "n_scored": scored,
    }
    if dataset == "bdm_lettuce":
        result["wtp"] = _score_wtp(numeric)
        purchase = {key: value for key, value in paired.items() if key == "conclusion_purchased_product"}
        result["purchase"] = mean_tvd_by_item(purchase)
        flat_h = [human_answer for pairs in purchase.values() for human_answer, _model in pairs]
        flat_m = [model_answer for pairs in purchase.values() for _human, model_answer in pairs]
        result["purchase_accuracy"] = accuracy(flat_m, flat_h) if flat_h else None
    pricing = score_pricing_persons(responses, outcome_rows)
    if pricing is not None:
        result["pricing_person"] = pricing
    return result


def mean_tvd_by_item(pairs_by_item: Mapping[str, Sequence[tuple[str, str]]]) -> dict[str, Any]:
    """Per-item TVD and top-1. Each pair is ``(human, model)``."""
    tvds: list[float] = []
    top1s: list[float] = []
    for item_id in sorted(pairs_by_item):
        pairs = list(pairs_by_item[item_id])
        human = [left for left, _right in pairs]
        model = [right for _left, right in pairs]
        vocab = sorted(set(human) | set(model))
        if not vocab:
            continue
        model_shares = _shares(model, vocab)
        human_shares = _shares(human, vocab)
        tvds.append(total_variation_distance(model_shares, human_shares))
        top1s.append(distribution_top1_accuracy(model_shares, human_shares))
    return {
        "mean_top1_acc": (sum(top1s) / len(top1s)) if top1s else None,
        "mean_tvd": (sum(tvds) / len(tvds)) if tvds else None,
        "n_items_top1": len(top1s),
        "n_items_tvd": len(tvds),
    }


def _score_wtp(codes_by_item: Mapping[str, Sequence[tuple[float, float]]]) -> dict[str, Any]:
    indiv: list[float] = []
    maes: list[float] = []
    correlations: list[float] = []
    glasses: list[float] = []
    ratios: list[float] = []
    for item_id in sorted(codes_by_item):
        pairs = list(codes_by_item[item_id])
        if not pairs:
            continue
        human = [left for left, _right in pairs]
        model = [right for _left, right in pairs]
        indiv.append(individual_accuracy(model, human, span=WTP_SPAN))
        maes.append(ordinal_mae(model, human))
        if len(pairs) >= 2:
            value = pearson(model, human)
            if value is not None:
                correlations.append(value)
        delta = glass_delta(model, human)
        if delta is not None:
            glasses.append(delta)
        ratio = sd_ratio(model, human)
        if ratio is not None:
            ratios.append(ratio)
    return {
        "fisher_z_corr": fisher_z_mean_correlation(correlations) if correlations else None,
        "glass_delta": (sum(glasses) / len(glasses)) if glasses else None,
        "indiv_acc": (sum(indiv) / len(indiv)) if indiv else None,
        "mae": (sum(maes) / len(maes)) if maes else None,
        "n_items": len(indiv),
        "sd_ratio": (sum(ratios) / len(ratios)) if ratios else None,
    }


def _shares(labels: Sequence[str], vocab: Sequence[str]) -> list[float]:
    counts = Counter(labels)
    return [float(counts[label]) for label in vocab]


def _human_index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for row in rows:
        subject_id = str(row.get("subject_id") or "")
        items = row.get("items")
        if not isinstance(items, list):
            raise ValueError(f"outcomes for {subject_id} have no items")
        for item in items:
            if not isinstance(item, dict):
                continue
            found[(subject_id, str(item.get("question_id") or ""))] = str(item.get("answer") or "").strip()
    return found


def _number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"file is missing: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [str(item) for item in value]


def _path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path
