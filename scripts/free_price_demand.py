#!/usr/bin/env python3
"""Compare human and model purchase rates at the Twin2K free-price tier.

Does not call a model. Reads local outcomes, the locked sample, and
``responses.jsonl`` files. Writes aggregate JSON, CSV, and a PNG.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.evaluate.demand import (  # noqa: E402
    adjacent_upward_jumps,
    bootstrap_free_gap,
    bootstrap_free_jump,
    build_offers,
    classify_free_endpoint,
    equal_product_curve,
    free_endpoint,
    free_to_first_positive,
    offer_weighted_curve,
    product_curves,
)
from popbench.evaluate.stimulus import parse_stimulus  # noqa: E402
from popbench.prepro.spec import TWIN_REPO_ID, TWIN_REVISION  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _stimuli(outcome_rows: list[dict[str, object]], sample_ids: set[str]) -> tuple[dict[tuple[str, str], dict[str, object]], int]:
    found: dict[tuple[str, str], dict[str, object]] = {}
    unparsed = 0
    for row in outcome_rows:
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in sample_ids:
            continue
        items = row.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id") or "")
            parsed = parse_stimulus(str(item.get("question_text") or ""))
            if parsed is None:
                unparsed += 1
                continue
            found[(subject_id, question_id)] = parsed
    return found, unparsed


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, human_curve: list[dict[str, object]], model_curves: dict[str, list[dict[str, object]]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for the demand chart. pip install matplotlib") from exc
    names = list(model_curves)
    columns = 3
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows), sharex=True, sharey=True)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    human_x = [float(point["tier_fraction"]) for point in human_curve]
    human_y = [float(point["purchase_rate"]) for point in human_curve]
    for axis, name in zip(flat, names):
        curve = model_curves[name]
        axis.plot(human_x, human_y, color="black", linewidth=2, label="human")
        axis.plot(
            [float(point["tier_fraction"]) for point in curve],
            [float(point["purchase_rate"]) for point in curve],
            color="#1f77b4",
            linewidth=1.6,
            label=name,
        )
        axis.axvline(0, color="#c0392b", linestyle="--", linewidth=1, label="free tier")
        axis.set_title(name, fontsize=9)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Fraction of product maximum price")
        axis.set_ylabel("Purchase rate")
    for axis in flat[len(names) :]:
        axis.axis("off")
    if flat:
        flat[0].legend(fontsize=7, loc="best")
    figure.suptitle("Twin2K n=93 product-equal-weight purchase rate. Zero is the free tier.", fontsize=11)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Twin2K free-price demand comparison. No model calls.")
    parser.add_argument("config", nargs="?", default=str(ROOT / "config" / "free_price.json"))
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("datasets") != ["twin2k_500"]:
        print("error: datasets must be exactly [\"twin2k_500\"]", file=sys.stderr)
        return 2
    root = ROOT
    outcomes = _load_jsonl(root / str(config["outcomes"]))
    sample = json.loads((root / str(config["sample"])).read_text(encoding="utf-8"))
    sample_ids = [str(subject_id) for subject_id in sample["subject_ids"]]
    stimuli, unparsed = _stimuli(outcomes, set(sample_ids))
    responses_dir = root / str(config["responses_dir"])
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates = int(config.get("bootstrap_replicates") or 1000)
    seed = int(config.get("bootstrap_seed") or 42)
    models = config["models"]
    if not isinstance(models, list) or not models:
        print("error: models must be a non-empty list", file=sys.stderr)
        return 2

    free_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    violation_rows: list[dict[str, object]] = []
    model_reports = []
    human_curve: list[dict[str, object]] | None = None
    plotted: dict[str, list[dict[str, object]]] = {}

    for index, spec in enumerate(models):
        model = str(spec["id"])
        temperature = spec.get("temperature")
        response_path = responses_dir / model / "twin2k_500" / "responses.jsonl"
        if not response_path.is_file():
            print(f"error: missing responses for {model}: {response_path}", file=sys.stderr)
            return 1
        responses = _load_jsonl(response_path)
        offers = build_offers(outcomes, responses, sample_ids, stimuli)
        free = free_endpoint(offers)
        human_products = product_curves(offers, "human")
        model_products = product_curves(offers, "model")
        human_equal = equal_product_curve(human_products)
        model_equal = equal_product_curve(model_products)
        human_jump = free_to_first_positive(human_products)
        model_jump = free_to_first_positive(model_products)
        human_boot = bootstrap_free_jump(offers, "human", seed=seed, replicates=replicates)
        model_boot = bootstrap_free_jump(offers, "model", seed=seed + 1 + index, replicates=replicates)
        gap_boot = bootstrap_free_gap(offers, seed=seed + 100 + index, replicates=replicates)
        finding = classify_free_endpoint(
            model_jump["rate_increase_to_first_positive"],
            model_boot["ci_low"],
            model_boot["ci_high"],
        )
        human_finding = classify_free_endpoint(
            human_jump["rate_increase_to_first_positive"],
            human_boot["ci_low"],
            human_boot["ci_high"],
        )
        report = {
            "finding_free_endpoint": finding,
            "free": free,
            "human_finding_free_endpoint": human_finding,
            "human_free_jump": {**human_jump, "bootstrap": human_boot},
            "model": model,
            "model_adjacent_upward_jumps": adjacent_upward_jumps(model_equal),
            "model_free_jump": {**model_jump, "bootstrap": model_boot},
            "offer_weighted_curve": offer_weighted_curve(offers, "model"),
            "paired_gap_bootstrap": gap_boot,
            "product_equal_weight_curve": model_equal,
            "temperature": temperature,
            "unparsed_stimuli_in_sample": unparsed,
        }
        model_reports.append(report)
        free_rows.append(
            {
                "assigned_free_offers": free["assigned_offers"],
                "both_mapped": free["both_mapped_offers"],
                "finding": finding,
                "human_finding": human_finding,
                "human_free_to_first_positive": human_jump["rate_increase_to_first_positive"],
                "human_jump_ci_high": human_boot["ci_high"],
                "human_jump_ci_low": human_boot["ci_low"],
                "human_mapped_rate": free["human"]["rate_mapped"],
                "human_products_with_free": free["products"],
                "human_rate_lower": free["human"]["rate_lower_unknown_as_no"],
                "human_rate_upper": free["human"]["rate_upper_unknown_as_yes"],
                "human_subjects_with_free": free["subjects"],
                "model": model,
                "model_free_to_first_positive": model_jump["rate_increase_to_first_positive"],
                "model_jump_ci_high": model_boot["ci_high"],
                "model_jump_ci_low": model_boot["ci_low"],
                "model_mapped_rate_on_paired_offers": free["paired_model_rate"],
                "model_rate_lower": free["model"]["rate_lower_unknown_as_no"],
                "model_rate_upper": free["model"]["rate_upper_unknown_as_yes"],
                "model_status_conflict": free["model_status"].get("conflict", 0),
                "model_status_missing": free["model_status"].get("missing", 0),
                "model_status_parse_error": free["model_status"].get("parse_error", 0),
                "model_status_unmapped": free["model_status"].get("unmapped", 0),
                "model_unknown": free["model"]["unknown"],
                "model_yes": free["model"]["yes"],
                "paired_accuracy": free["paired_accuracy"],
                "paired_gap_pp": free["paired_gap_percentage_points"],
                "paired_gap_ci_high": gap_boot["ci_high"],
                "paired_gap_ci_low": gap_boot["ci_low"],
                "paired_human_rate": free["paired_human_rate"],
                "questions_with_a_free_offer": free["questions"],
                "temperature": temperature,
            }
        )
        if human_curve is None:
            human_curve = human_equal
            for point in human_equal:
                curve_rows.append({"series": "human", "temperature": None, "weighting": "product_equal", **point})
        for point in model_equal:
            curve_rows.append({"series": model, "temperature": temperature, "weighting": "product_equal", **point})
        plotted[model] = model_equal
        for jump in adjacent_upward_jumps(model_equal):
            violation_rows.append({"series": model, **jump})
        for jump in adjacent_upward_jumps(human_equal):
            if index == 0:
                violation_rows.append({"series": "human", **jump})

    payload = {
        "bootstrap": {
            "interval": "subject-cluster percentile, 2.5 and 97.5, linear between order statistics",
            "replicates": replicates,
            "seed": seed,
        },
        "caveat": (
            "Cells are between-subject assignments in the locked n=93 sample. "
            "They are not individual counterfactual demand curves, not a causal "
            "free-price effect, and not a replication of the original paper's "
            "models or population. The quoted sentence was not checked against a "
            "paper PDF in this repository."
        ),
        "criterion": (
            "Non-increasing means each adjacent observed tier has a purchase rate "
            "less than or equal to the previous tier. A tie is allowed. An upward "
            "jump is a violation. Absent tiers are omitted rather than treated as "
            "zero demand."
        ),
        "dataset": "twin2k_500",
        "models": model_reports,
        "provenance": {
            "currency": "US dollars with two decimal places in the stimulus sentence",
            "free_price": "Parsed amount is exactly 0.00. That is tier 0 of the product grid.",
            "normalized_tier": (
                "Tier k is round-half-up of k/10 times the product's maximum positive "
                "price in this locked sample. Dollar prices are not pooled across products."
            ),
            "price_field": (
                "No price_shown column exists. The price is the $X.XX slot in "
                "wave4 QuestionText, copied to outcomes.question_text. Unparsed "
                "sentences are not treated as free."
            ),
            "sample_seed": sample.get("seed"),
            "sample_size": sample.get("sample_size"),
            "source": f"Hugging Face {TWIN_REPO_ID} revision {TWIN_REVISION}",
            "unparsed_stimuli_in_sample": unparsed,
        },
        "schema": {
            "free_price_summary.csv": "One row per model. Rates are fractions. paired_gap_pp is model minus human, in percentage points, on offers both sides mapped.",
            "demand_curves.csv": "Product-equal-weight purchase rate by tier. tier_fraction 0 is free.",
            "upward_jumps.csv": "Adjacent tiers on that curve where the rate increases.",
            "free_price_demand.json": "Full per-model blocks, bootstrap intervals, and provenance.",
        },
    }
    (output_dir / "free_price_demand.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "free_price_summary.csv", free_rows)
    _write_csv(output_dir / "demand_curves.csv", curve_rows)
    _write_csv(output_dir / "upward_jumps.csv", violation_rows)
    if human_curve is not None:
        _plot(output_dir / "demand_curves.png", human_curve, plotted)
    print(json.dumps({"models": len(model_reports), "output_dir": str(output_dir), "unparsed": unparsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
