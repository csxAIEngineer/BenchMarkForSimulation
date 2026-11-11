#!/usr/bin/env python3
"""Build Arm A triplets and write metrics, checks, and permutation balance.

Does not call a model. Reads local default baselines and ``__three-option`` responses.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.evaluate.arm_a import (  # noqa: E402
    HUMAN_NEED_REJECTION,
    class_shares,
    content_sha256,
    file_sha256,
    interpolate_p50,
    permutation_balance,
    product_equal_rates,
    product_maxima,
    read_triplets,
    semantic_spot_checks,
    triplets_from_responses,
    write_triplets,
)
from popbench.evaluate.demand import (  # noqa: E402
    bootstrap_free_jump,
    build_offers,
    classify_free_endpoint,
    equal_product_curve,
    free_endpoint,
    free_to_first_positive,
    product_curves,
)
from popbench.evaluate.options import NO_OPTION, YES_OPTION, match_option  # noqa: E402
from popbench.evaluate.score import mean_tvd_by_item, score_run  # noqa: E402
from popbench.evaluate.stimulus import parse_stimulus  # noqa: E402

DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "deepseek-flash",
    "gemini-3.5-flash",
    "deepseek-v4-flash",
    "claude-opus-5-5",
    "MiniMax-M3.1-Flash",
    "gemini-3.5-flash-lite",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _outcomes_index(rows: list[dict]) -> dict[str, list[dict]]:
    return {str(row["subject_id"]): list(row["items"]) for row in rows}


def _metric(block: object) -> float | None:
    if isinstance(block, dict) and "value" in block:
        value = block["value"]
        return None if value is None else float(value)
    if isinstance(block, (int, float)):
        return float(block)
    return None


def prepare_triplets() -> dict:
    outcomes = _load_jsonl(ROOT / "data/prepro/outcomes/twin2k_500/outcomes.jsonl")
    sample = json.loads((ROOT / "outputs/prepro/twin2k_500/sample.json").read_text(encoding="utf-8"))
    sample_ids = {str(subject_id) for subject_id in sample["subject_ids"]}
    indexed = _outcomes_index(outcomes)
    maxima = product_maxima(outcomes, sample_ids)
    hashes = {}
    reference = None
    for model in DEFAULT_MODELS:
        path = ROOT / "outputs/simulate" / model / "twin2k_500/responses.jsonl"
        if not path.is_file():
            hashes[model] = None
            continue
        triplets = triplets_from_responses(_load_jsonl(path), indexed, maxima)
        if len(triplets) != 3720:
            raise SystemExit(f"{model} baseline has {len(triplets)} triplets, expected 3720")
        digest = content_sha256(triplets)
        hashes[model] = digest
        reference = triplets if reference is None else reference
    present = {model: digest for model, digest in hashes.items() if digest}
    if len(set(present.values())) != 1:
        raise SystemExit(f"baseline triplet hashes differ: {present}")
    if reference is None:
        raise SystemExit("no default baseline responses were found")
    destination = ROOT / "outputs/evaluate/triplets.parquet"
    file_hash = write_triplets(destination, reference)
    content = content_sha256(reference)
    orders = []
    from popbench.simulate.three_option import perm_for

    for row in reference:
        orders.append(perm_for(str(row["pid"]), str(row["product_id"])))
    balance = permutation_balance(orders)
    payload = {
        "content_sha256": content,
        "file_sha256": file_hash,
        "models": hashes,
        "n": len(reference),
        "parquet": str(destination.relative_to(ROOT)),
        "permutation_balance": balance,
        "permutation_pass": all(row["pass"] for row in balance),
    }
    (ROOT / "outputs/evaluate/triplets_sha256.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(ROOT / "reports/arm_A_permutation_balance.csv", balance)
    print(json.dumps({"content_sha256": content, "file_sha256": file_hash, "models": list(present), "permutation_pass": payload["permutation_pass"]}, indent=2))
    return payload


def _matched(responses: list[dict], outcomes: list[dict]) -> dict:
    human = {}
    for row in outcomes:
        for item in row["items"]:
            human[(str(row["subject_id"]), str(item["question_id"]))] = str(item["answer"]).strip()
    options = [NO_OPTION, YES_OPTION]
    paired: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in responses:
        if row.get("parse_error"):
            continue
        key = (str(row["subject_id"]), str(row["question_id"]))
        held = human[key]
        predicted = str(row.get("answer") or "").strip()
        paired[key[1]].append((match_option(held, options) or held, match_option(predicted, options) or predicted))
    return mean_tvd_by_item(paired)


def _curve_p50(responses: list[dict], outcomes: list[dict], sample_ids: list[str]) -> dict:
    stimuli = {}
    sample = set(sample_ids)
    for row in outcomes:
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in sample:
            continue
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            parsed = parse_stimulus(str(item.get("question_text") or ""))
            if parsed is not None:
                stimuli[(subject_id, str(item.get("question_id") or ""))] = parsed
    offers = build_offers(outcomes, responses, sample_ids, stimuli)
    curve = equal_product_curve(product_curves(offers, "model"))
    rates = {int(point["tier"]): float(point["purchase_rate"]) for point in curve}
    return interpolate_p50(rates)


def _free_jump(responses: list[dict], outcomes: list[dict], sample_ids: list[str], seed: int) -> dict:
    stimuli = {}
    sample = set(sample_ids)
    for row in outcomes:
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in sample:
            continue
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            parsed = parse_stimulus(str(item.get("question_text") or ""))
            if parsed is not None:
                stimuli[(subject_id, str(item.get("question_id") or ""))] = parsed
    offers = build_offers(outcomes, responses, sample_ids, stimuli)
    jump = free_to_first_positive(product_curves(offers, "model"))
    boot = bootstrap_free_jump(offers, "model", seed=seed, replicates=1000)
    finding = classify_free_endpoint(jump["rate_increase_to_first_positive"], boot["ci_low"], boot["ci_high"])
    free = free_endpoint(offers)
    return {
        "free_assigned": free["assigned_offers"],
        "free_both_mapped": free["both_mapped_offers"],
        "free_endpoint_call": finding,
        "free_jump_ci_high": boot["ci_high"],
        "free_jump_ci_low": boot["ci_low"],
        "free_to_first_positive": jump["rate_increase_to_first_positive"],
    }


def _check_lines(model: str, rows: list[dict], baseline_mtime: float | None, current_mtime: float | None) -> list[str]:
    locked = read_triplets(ROOT / "outputs/evaluate/triplets.parquet")
    locked_hash = content_sha256(locked)
    run_hash = content_sha256(rows)
    shares = class_shares(rows)
    mapped = sum(1 for row in rows if row.get("semantic") in {"buy", "no_need", "too_expensive"})
    parse_rate = mapped / len(rows) if rows else 0.0
    balance = permutation_balance([row.get("option_order") or [] for row in rows])
    spots = semantic_spot_checks(rows, limit=5)
    tier_locked = Counter(int(row["tier"]) for row in locked)
    tier_run = Counter(int(row["tier"]) for row in rows)
    checks = [
        ("1 triplet sha256", run_hash == locked_hash, run_hash),
        ("2 n_rows", len(rows) == 3720, str(len(rows))),
        ("3 parse_rate", parse_rate >= 0.98, f"{parse_rate:.6f}"),
        (
            "4 class shares sum",
            abs(shares["P_buy"] + shares["P_need"] + shares["P_price"] - 1) <= 0.001,
            f"P_buy+P_need+P_price={shares['P_buy'] + shares['P_need'] + shares['P_price']:.6f}",
        ),
        ("5 permutation balance", all(item["pass"] for item in balance), "see arm_A_permutation_balance.csv"),
        ("6 semantic spot check", len(spots) == 5 and all(item["pass"] for item in spots), json.dumps(spots, ensure_ascii=False)),
        (
            "7 baseline mtime",
            baseline_mtime is not None and current_mtime is not None and current_mtime == baseline_mtime,
            f"before={baseline_mtime} after={current_mtime}",
        ),
        ("8 tier counts", tier_locked == tier_run, "match" if tier_locked == tier_run else "differ"),
    ]
    lines = [f"model {model}"]
    failed = False
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failed = True
        lines.append(f"{status} {name}: {detail}")
    lines.append("RESULT " + ("PASS" if not failed else "FAIL"))
    return lines


def score_finished(models: list[str], mtimes: dict[str, float]) -> int:
    outcomes = _load_jsonl(ROOT / "data/prepro/outcomes/twin2k_500/outcomes.jsonl")
    sample = json.loads((ROOT / "outputs/prepro/twin2k_500/sample.json").read_text(encoding="utf-8"))
    sample_ids = [str(subject_id) for subject_id in sample["subject_ids"]]
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    check_blocks = []
    balance_rows = []
    hard_fail = False
    for index, model in enumerate(models):
        path = ROOT / "outputs/simulate" / f"{model}__three-option" / "twin2k_500/responses.jsonl"
        if not path.is_file():
            continue
        rows = _load_jsonl(path)
        baseline_path = ROOT / "outputs/simulate" / model / "twin2k_500/responses.jsonl"
        current_mtime = baseline_path.stat().st_mtime if baseline_path.is_file() else None
        lines = _check_lines(model, rows, mtimes.get(model), current_mtime)
        check_blocks.extend(lines)
        check_blocks.append("")
        if any(line.startswith("FAIL") for line in lines):
            hard_fail = True
        report = score_run(
            ROOT,
            {
                "datasets": ["twin2k_500"],
                "model": f"{model}__three-option",
                "outcomes_dir": "data/prepro/outcomes",
                "output": f"outputs/evaluate/arm_a_{model}.json",
                "responses_dir": "outputs/simulate",
            },
        )
        dataset = report["datasets"]["twin2k_500"]
        person = dataset["pricing_person"]
        matched = _matched(rows, outcomes)
        shares = class_shares(rows)
        mapped = sum(1 for row in rows if row.get("semantic") in {"buy", "no_need", "too_expensive"})
        baseline_rows = _load_jsonl(baseline_path)
        raw_baseline = _curve_p50(baseline_rows, outcomes, sample_ids)
        conditional = interpolate_p50(product_equal_rates(rows, conditional=True))
        collapsed = interpolate_p50(product_equal_rates(rows, conditional=False))
        free = _free_jump(rows, outcomes, sample_ids, seed=42 + index)
        balance = permutation_balance([row.get("option_order") or [] for row in rows])
        for item in balance:
            balance_rows.append({"model": model, **item})
        need = shares["P_need"]
        metric_rows.append(
            {
                "model": model,
                "individual_accuracy_micro": _metric(person["individual_accuracy_micro"]),
                "individual_accuracy_macro": _metric(person["individual_accuracy_macro"]),
                "matched_mean_tvd": matched["mean_tvd"],
                "matched_mean_top1_acc": matched["mean_top1_acc"],
                "cross_person_pearson": _metric(person["cross_person_pearson"]),
                "glass_delta": _metric(person["glass_delta"]),
                "sd_ratio": _metric(person["sd_ratio"]),
                "n_complete_subjects": person["coverage"]["n_complete_subjects"],
                "n_unmapped_model": person["coverage"]["n_unmapped_model"],
                "n_parse_errors": dataset["n_parse_errors"],
                **free,
                "P_buy": shares["P_buy"],
                "P_need": shares["P_need"],
                "P_price": shares["P_price"],
                "P_unmapped": shares["P_unmapped"],
                "R_need": need,
                "R_need_diff": need - float(HUMAN_NEED_REJECTION),
                "parse_rate": mapped / len(rows) if rows else 0.0,
                "raw_p50_baseline": raw_baseline["p50"],
                "raw_p50_baseline_reason": raw_baseline["reason"],
                "raw_p50_baseline_upward": json.dumps(raw_baseline["upward_tiers"]),
                "conditional_p50": conditional["p50"],
                "conditional_p50_reason": conditional["reason"],
                "conditional_p50_upward": json.dumps(conditional["upward_tiers"]),
                "collapsed_p50": collapsed["p50"],
                "collapsed_p50_reason": collapsed["reason"],
                "collapsed_p50_upward": json.dumps(collapsed["upward_tiers"]),
                "class_collapse": min(shares["P_buy"], shares["P_need"], shares["P_price"]) < 0.02,
            }
        )
    _write_csv(reports / "arm_A_metrics.csv", metric_rows)
    (reports / "arm_A_metrics.json").write_text(json.dumps(metric_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports / "arm_A_checks.txt").write_text("\n".join(check_blocks) + "\n", encoding="utf-8")
    if balance_rows:
        _write_csv(reports / "arm_A_permutation_balance.csv", balance_rows)
    print(json.dumps(metric_rows, ensure_ascii=False, indent=2))
    return 1 if hard_fail else 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Arm A triplet lock and metric report. No model calls.")
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--mtimes", default=str(ROOT / "outputs/evaluate/arm_a_baseline_mtimes.json"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_triplets()
        mtimes = {}
        for model in DEFAULT_MODELS:
            path = ROOT / "outputs/simulate" / model / "twin2k_500/responses.jsonl"
            if path.is_file():
                mtimes[model] = path.stat().st_mtime
        Path(args.mtimes).write_text(json.dumps(mtimes, indent=2) + "\n", encoding="utf-8")
        return 0
    mtimes = json.loads(Path(args.mtimes).read_text(encoding="utf-8")) if Path(args.mtimes).is_file() else {}
    models = args.models or DEFAULT_MODELS
    return score_finished(models, mtimes)


if __name__ == "__main__":
    raise SystemExit(main())
