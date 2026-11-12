#!/usr/bin/env python3
"""Arm B checks and E7. Does not call a model."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.evaluate.arm_a import content_sha256, read_triplets  # noqa: E402
from popbench.evaluate.arm_b import (  # noqa: E402
    appendix_association,
    bootstrap_d_buy,
    condition_p50,
    d_buy,
    e7_pass,
    observation_p_buy,
    person_rates,
)
from popbench.evaluate.options import NO_OPTION, YES_OPTION, code_purchase  # noqa: E402
from popbench.simulate.need_inject import NEED_SENTENCES, inject_need, need_state, sentence_lengths_ok  # noqa: E402

LOCKED_SHA = "5ab41b0772a5b26e5802a2236e0e9ae955b6cf2e4b4a72e963a36c53ca853433"
EXPECTED = {"high": 930, "low": 930, "fill": 1860}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _human_codes(outcomes: list[dict]) -> dict[tuple[str, str], int]:
    codebook = {NO_OPTION: 0, YES_OPTION: 1}
    found: dict[tuple[str, str], int] = {}
    for row in outcomes:
        subject_id = str(row.get("subject_id") or "")
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            code = code_purchase(str(item.get("answer") or ""), codebook)
            if code is not None:
                found[(subject_id, str(item.get("question_id") or ""))] = code
    return found


def _outcomes_by_question(outcomes: list[dict]) -> dict[tuple[str, str], str]:
    found = {}
    for row in outcomes:
        subject_id = str(row.get("subject_id") or "")
        for item in row.get("items") or []:
            if isinstance(item, dict):
                found[(subject_id, str(item.get("question_id") or ""))] = str(item.get("question_text") or "")
    return found


def _within(count: int, expected: int) -> bool:
    return abs(count - expected) <= expected * 0.03


def _spot_checks(triplets: list[dict], stems: dict[tuple[str, str], str]) -> list[dict]:
    checks = []
    for row in triplets:
        state = need_state(str(row["pid"]), str(row["product_id"]))
        stem = stems.get((str(row["pid"]), str(row["question_id"])))
        if not stem:
            continue
        prompt = inject_need(stem, state)
        sentence = NEED_SENTENCES[state]
        needle = f"${row['price']}. {sentence} Would you or would you not purchase this product?"
        checks.append(
            {
                "need_state": state,
                "pass": needle in prompt and "Options:\nA) Yes, I would purchase the product\n" in prompt,
                "pid": row["pid"],
                "price": row["price"],
                "product_id": row["product_id"],
            }
        )
        if len(checks) == 5 and {item["need_state"] for item in checks} >= {"high", "low", "fill"}:
            break
    if len(checks) < 5:
        return checks
    # Prefer a mix: walk until we have five that include more than one state if possible.
    mixed = []
    seen = set()
    for row in triplets:
        state = need_state(str(row["pid"]), str(row["product_id"]))
        if state in seen and len(mixed) < 5 and len(seen) < 3:
            continue
        stem = stems.get((str(row["pid"]), str(row["question_id"])))
        if not stem:
            continue
        prompt = inject_need(stem, state)
        sentence = NEED_SENTENCES[state]
        needle = f"${row['price']}. {sentence} Would you or would you not purchase this product?"
        mixed.append(
            {
                "need_state": state,
                "pass": needle in prompt and "A) Yes, I would purchase the product" in prompt,
                "pid": row["pid"],
                "price": row["price"],
                "product_id": row["product_id"],
            }
        )
        seen.add(state)
        if len(mixed) == 5:
            break
    return mixed or checks[:5]


def score(models: list[str], mtimes: dict[str, float]) -> int:
    triplets = read_triplets(ROOT / "outputs/evaluate/triplets.parquet")
    locked_hash = content_sha256(triplets)
    outcomes = _load_jsonl(ROOT / "data/prepro/outcomes/twin2k_500/outcomes.jsonl")
    stems = _outcomes_by_question(outcomes)
    humans = _human_codes(outcomes)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    e7_rows = []
    check_lines = []
    hard_fail = False
    sentences = [
        {
            "injected_text": NEED_SENTENCES[need_state(str(row["pid"]), str(row["product_id"]))],
            "need_state": need_state(str(row["pid"]), str(row["product_id"])),
            "pid": row["pid"],
            "product_id": row["product_id"],
        }
        for row in triplets
    ]
    for model in models:
        path = ROOT / "outputs/simulate" / f"{model}__need-inject" / "twin2k_500" / "responses.jsonl"
        if not path.is_file():
            continue
        rows = _load_jsonl(path)
        run_hash = content_sha256(rows)
        counts = Counter(row.get("need_state") for row in rows)
        mapped = sum(1 for row in rows if row.get("parsed_choice") in {"A", "B"})
        parse_rate = mapped / len(rows) if rows else 0.0
        spots = _spot_checks(triplets, stems)
        baseline = ROOT / "outputs/simulate" / model / "twin2k_500" / "responses.jsonl"
        three = ROOT / "outputs/simulate" / f"{model}__three-option" / "twin2k_500" / "responses.jsonl"
        current_base = baseline.stat().st_mtime if baseline.is_file() else None
        current_three = three.stat().st_mtime if three.is_file() else None
        boot = bootstrap_d_buy(rows, seed=42, replicates=1000)
        high_p50 = condition_p50(rows, "high")
        low_p50 = condition_p50(rows, "low")
        d_p50 = None
        if high_p50["p50"] is not None and low_p50["p50"] is not None:
            d_p50 = high_p50["p50"] - low_p50["p50"]
        tier_gaps = {tier: d_buy(rows, tier) for tier in range(11)}
        low_tiers = [tier_gaps[tier] for tier in range(5) if tier_gaps[tier] is not None]
        high_tiers = [tier_gaps[tier] for tier in range(6, 11) if tier_gaps[tier] is not None]
        low_mean = sum(low_tiers) / len(low_tiers) if low_tiers else None
        high_mean = sum(high_tiers) / len(high_tiers) if high_tiers else None
        checks = [
            ("1 triplet sha256", run_hash == locked_hash == LOCKED_SHA, run_hash),
            ("2 n_rows", len(rows) == 3720, str(len(rows))),
            (
                "3 condition counts",
                _within(counts.get("high", 0), 930) and _within(counts.get("low", 0), 930) and _within(counts.get("fill", 0), 1860),
                f"high={counts.get('high', 0)} low={counts.get('low', 0)} fill={counts.get('fill', 0)}",
            ),
            ("4 sentence lengths", sentence_lengths_ok(), str({name: len(text) for name, text in NEED_SENTENCES.items()})),
            ("5 prompt spot check", len(spots) == 5 and all(item["pass"] for item in spots), json.dumps(spots, ensure_ascii=False)),
            ("6 parse rate", parse_rate >= 0.98, f"{parse_rate:.6f}"),
            (
                "7 E7",
                e7_pass(boot["d_buy"], boot["ci_low"], boot["ci_high"]),
                f"d_buy={boot['d_buy']} ci=({boot['ci_low']}, {boot['ci_high']}) d_p50={d_p50}",
            ),
            (
                "8 baseline mtimes",
                current_base is not None
                and current_base == mtimes.get(f"{model}:baseline")
                and current_three == mtimes.get(f"{model}:three-option"),
                f"baseline {mtimes.get(f'{model}:baseline')} -> {current_base}; three-option {mtimes.get(f'{model}:three-option')} -> {current_three}",
            ),
        ]
        check_lines.append(f"model {model}")
        failed = False
        for name, passed, detail in checks:
            check_lines.append(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
            failed = failed or not passed
        check_lines.append("RESULT " + ("FAIL" if failed else "PASS"))
        check_lines.append("")
        hard_fail = hard_fail or failed
        e7_row = {
            "model": model,
            "d_buy": boot["d_buy"],
            "d_buy_ci_low": boot["ci_low"],
            "d_buy_ci_high": boot["ci_high"],
            "d_buy_replicates": boot["replicates"],
            "e7_pass": e7_pass(boot["d_buy"], boot["ci_low"], boot["ci_high"]),
            "P_buy_high": observation_p_buy(rows, "high"),
            "P_buy_low": observation_p_buy(rows, "low"),
            "P_buy_fill": observation_p_buy(rows, "fill"),
            "P50_high": high_p50["p50"],
            "P50_low": low_p50["p50"],
            "P50_high_reason": high_p50["reason"],
            "P50_low_reason": low_p50["reason"],
            "P50_high_upward": json.dumps(high_p50["upward_tiers"]),
            "P50_low_upward": json.dumps(low_p50["upward_tiers"]),
            "d_p50": d_p50,
            "d_buy_tiers_0_to_4_mean": low_mean,
            "d_buy_tiers_6_to_10_mean": high_mean,
            "parse_rate": parse_rate,
        }
        for tier, value in tier_gaps.items():
            e7_row[f"d_buy_tier_{tier}"] = value
        e7_rows.append(e7_row)
        for state in ("high", "low", "fill"):
            summary = condition_p50(rows, state)
            model_rates, human_rates = person_rates(rows, humans, state)
            association = appendix_association(model_rates, human_rates)
            metric_rows.append(
                {
                    "model": model,
                    "need_state": state,
                    "n_offers": counts.get(state, 0),
                    "P_buy": observation_p_buy(rows, state),
                    "P50": summary["p50"],
                    "P50_reason": summary["reason"],
                    "sd_ratio": association["sd_ratio"],
                    "cross_person_pearson": association["cross_person_pearson"],
                    "n_complete_subjects": len(model_rates),
                }
            )
    _write_csv(reports / "arm_B_metrics.csv", metric_rows)
    _write_csv(reports / "arm_B_e7.csv", e7_rows)
    (reports / "arm_B_checks.txt").write_text("\n".join(check_lines) + "\n", encoding="utf-8")
    _write_csv(reports / "arm_B_sentences.csv", sentences)
    (reports / "arm_B_e7.json").write_text(json.dumps(e7_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(e7_rows, ensure_ascii=False, indent=2))
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
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _cell(value: object) -> object:
    if isinstance(value, float):
        return repr(value)
    return value


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score Arm B need-inject outputs. No model calls.")
    parser.add_argument("--models", nargs="*", default=["gemini-3.8-flash", "deepseek-flash"])
    parser.add_argument("--mtimes", default=str(ROOT / "outputs/evaluate/arm_b_baseline_mtimes.json"))
    args = parser.parse_args(argv)
    mtimes = json.loads(Path(args.mtimes).read_text(encoding="utf-8")) if Path(args.mtimes).is_file() else {}
    return score(args.models, mtimes)


if __name__ == "__main__":
    raise SystemExit(main())
