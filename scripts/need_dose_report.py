#!/usr/bin/env python3
"""Dose-response and budget reports for the full-sample need arms. No model calls."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popbench.evaluate.arm_a import content_sha256, read_triplets  # noqa: E402
from popbench.evaluate.need_dose_stats import (  # noqa: E402
    DOSE_LENGTH,
    arm_fit,
    bernoulli_aic,
    bootstrap_arm,
    bootstrap_budget_gap,
    bootstrap_coefficients,
    bootstrap_curve_contrasts,
    stack_dose_rows,
    two_sided_p,
)
from popbench.simulate.need_dose import DOSE_SENTENCES, PRICE_CLAUSE, inject_dose  # noqa: E402

LOCKED_SHA = "5ab41b0772a5b26e5802a2236e0e9ae955b6cf2e4b4a72e963a36c53ca853433"
DOSE_ARMS = ("N1", "N2", "N3", "N4")
MODELS = ("gemini-3.8-flash", "deepseek-flash", "deepseek-v4-flash")
FLOOR_LEVEL = 0.03
FLOOR_GAP = 0.02


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _arm_path(model: str, arm: str) -> Path:
    if arm == "BUD":
        return ROOT / "outputs/simulate" / f"{model}__budget" / "twin2k_500" / "responses.jsonl"
    return ROOT / "outputs/simulate" / f"{model}__need-dose" / "twin2k_500" / arm / "responses.jsonl"


def _significant_positive(estimate: float | None, standard_error: float | None) -> bool:
    if estimate is None or standard_error is None:
        return False
    return estimate > 0 and estimate - 1.96 * standard_error > 0


def _tree_now(name: str) -> tuple[str, list[dict]]:
    directory = ROOT / "outputs/simulate" / name
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            data = path.read_bytes()
            files.append(
                {
                    "mtime_ns": path.stat().st_mtime_ns,
                    "path": str(path.relative_to(ROOT / "outputs/simulate")),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    blob = "\n".join(f"{item['path']} {item['sha256']}" for item in files)
    return hashlib.sha256(blob.encode()).hexdigest(), files


def _phase1_unchanged(snapshot: dict) -> tuple[bool, str]:
    problems = []
    for name, recorded in snapshot.items():
        tree, files = _tree_now(name)
        if tree != recorded["tree_sha256"]:
            problems.append(f"{name} content")
        recorded_files = {item["path"]: item for item in recorded["files"]}
        for item in files:
            previous = recorded_files.get(item["path"])
            if previous is None or previous["mtime_ns"] != item["mtime_ns"]:
                problems.append(f"{name} mtime {item['path']}")
    return (not problems), ("; ".join(problems) if problems else "unchanged")


def _prompt_rows(model: str, arm: str, rows: list[dict], stems: dict[tuple[str, str], str]) -> list[dict]:
    found = []
    for row in rows:
        stem = stems.get((str(row["pid"]), str(row["question_id"])), "")
        prompt = inject_dose(stem, arm).rstrip()
        found.append(
            {
                "model": model,
                "need_arm": arm,
                "pid": row["pid"],
                "price_clause_count": len(PRICE_CLAUSE.findall(prompt)),
                "prompt_chars": len(prompt),
                "question_id": row["question_id"],
                "sentence_exact": row.get("sentence") == DOSE_SENTENCES[arm],
            }
        )
    return found


def main() -> int:
    triplets = read_triplets(ROOT / "outputs/evaluate/triplets.parquet")
    locked = content_sha256(triplets)
    outcomes = _load_jsonl(ROOT / "data/prepro/outcomes/twin2k_500/outcomes.jsonl")
    stems: dict[tuple[str, str], str] = {}
    for row in outcomes:
        subject_id = str(row.get("subject_id") or "")
        for item in row.get("items") or []:
            if isinstance(item, dict):
                stems[(subject_id, str(item.get("question_id") or ""))] = str(item.get("question_text") or "")
    snapshot = json.loads(Path("/tmp/need_dose_phase1_snapshot.json").read_text())
    phase1_ok, phase1_detail = _phase1_unchanged(snapshot)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    params = []
    contrasts = []
    shapes = []
    replications = []
    budgets = []
    lengths = []
    checks: list[str] = []
    unmapped_rows = []
    for model in MODELS:
        loaded = {arm: _load_jsonl(_arm_path(model, arm)) for arm in (*DOSE_ARMS, "BUD")}
        phase1 = _load_jsonl(ROOT / "outputs/simulate" / f"{model}__need-inject" / "twin2k_500" / "responses.jsonl")
        checks.append(f"model {model}")
        model_fail = False
        for arm, rows in loaded.items():
            keys = [(str(row.get("pid")), str(row.get("question_id"))) for row in rows]
            duplicate = len(keys) != len(set(keys))
            subjects = Counter(str(row.get("pid")) for row in rows)
            complete_subjects = sum(count == 40 for count in subjects.values())
            mapped = sum(row.get("buy") in (0, 1) for row in rows)
            parse_rate = mapped / len(rows) if rows else 0.0
            run_hash = content_sha256(rows) if rows else ""
            prompts = _prompt_rows(model, arm, rows, stems) if rows else []
            lengths.extend(
                {
                    "model": item["model"],
                    "need_arm": item["need_arm"],
                    "pid": item["pid"],
                    "prompt_chars": item["prompt_chars"],
                    "question_id": item["question_id"],
                }
                for item in prompts
            )
            sentence_hits = sum(item["sentence_exact"] for item in prompts)
            price_hits = sum(item["price_clause_count"] == 1 for item in prompts)
            blank_subjects = sorted(subject for subject, count in subjects.items() if count == 40 and all(
                row.get("buy") not in (0, 1) for row in rows if str(row.get("pid")) == subject
            ))
            for row in rows:
                if row.get("buy") not in (0, 1):
                    unmapped_rows.append(
                        {
                            "model": model,
                            "need_arm": arm,
                            "pid": row.get("pid"),
                            "question_id": row.get("question_id"),
                            "parse_error": row.get("parse_error"),
                            "raw_text": row.get("raw_text"),
                            "unmapped_reason": row.get("unmapped_reason"),
                        }
                    )
            arm_checks = [
                ("1 coverage", len(rows) == 3720 and complete_subjects == 93 and not duplicate and len(subjects) == 93, f"rows={len(rows)} subjects={len(subjects)} complete40={complete_subjects} duplicate={duplicate}"),
                ("2 triplet sha256", run_hash == locked == LOCKED_SHA, run_hash),
                ("3 sentence match", bool(prompts) and sentence_hits == len(prompts), f"{sentence_hits}/{len(prompts)}"),
                ("4 price clause once", bool(prompts) and price_hits == len(prompts), f"{price_hits}/{len(prompts)}"),
                ("5 sentence length", len(DOSE_SENTENCES[arm]) == DOSE_LENGTH[arm], f"{len(DOSE_SENTENCES[arm])}"),
                ("6 parse rate", parse_rate >= 0.99, f"{parse_rate:.6f}"),
                ("7 phase1 unchanged", phase1_ok, phase1_detail),
                ("8 no fully unmapped subject", not blank_subjects, ",".join(blank_subjects) or "none"),
            ]
            checks.append(f"arm {arm}")
            for name, passed, detail in arm_checks:
                checks.append(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
                model_fail = model_fail or not passed
            fit = bootstrap_arm(rows, seed=42, replicates=1000)
            point = fit["fit"] or {}
            params.append(
                {
                    "alpha": point.get("alpha"),
                    "beta": point.get("beta"),
                    "beta_over_human": point.get("beta_over_human"),
                    "model": model,
                    "n": point.get("n"),
                    "need_arm": arm,
                    "p0": point.get("p0"),
                    "p100": point.get("p100"),
                    "p200": point.get("p200"),
                    "se_alpha": fit["se_alpha"],
                    "se_beta": fit["se_beta"],
                }
            )
        dose_rows = {arm: loaded[arm] for arm in DOSE_ARMS}
        linear = bootstrap_coefficients(dose_rows, ["dose", "pct_dose"], seed=42, replicates=1000)
        quadratic = bootstrap_coefficients(dose_rows, ["dose", "dose2", "pct_dose"], seed=42, replicates=1000)
        length_model = bootstrap_coefficients(dose_rows, ["dose", "len", "pct_dose"], seed=42, replicates=1000)
        stacked = stack_dose_rows(dose_rows)
        aic_dose = bernoulli_aic(stacked, ["dose"])
        aic_len = bernoulli_aic(stacked, ["len"])
        contrasts_fit = bootstrap_curve_contrasts(
            dose_rows,
            (("N1", "N2"), ("N2", "N3"), ("N3", "N4")),
            seed=42,
            replicates=1000,
        )
        for item in contrasts_fit:
            item["model"] = model
            contrasts.append(item)
        p100 = {arm: (arm_fit(loaded[arm]) or {}).get("p100") for arm in DOSE_ARMS}
        p0 = {arm: (arm_fit(loaded[arm]) or {}).get("p0") for arm in DOSE_ARMS}
        floor_notes = []
        monotonic = True
        for earlier, later, item in zip(("N1", "N2", "N3"), ("N2", "N3", "N4"), contrasts_fit):
            left = p100[earlier]
            right = p100[later]
            difference = item["difference"]
            significant_drop = (
                difference is not None
                and difference < 0
                and item["ci_high"] is not None
                and item["ci_high"] < 0
            )
            floor = (
                left is not None
                and right is not None
                and left < FLOOR_LEVEL
                and right < FLOOR_LEVEL
                and difference is not None
                and abs(difference) < FLOOR_GAP
            )
            if floor:
                floor_notes.append(f"{earlier}->{later} indistinguishable")
            elif left is None or right is None or right + 1e-12 < left or significant_drop:
                monotonic = False
        gamma = (linear.get("coefficients") or {}).get("dose")
        gamma_se = (linear.get("se") or {}).get("dose")
        trend = _significant_positive(gamma, gamma_se)
        m1 = monotonic and trend
        gamma2 = (quadratic.get("coefficients") or {}).get("dose2")
        gamma2_p = (quadratic.get("p") or {}).get("dose2")
        gamma1_m2 = (quadratic.get("coefficients") or {}).get("dose")
        delta = (quadratic.get("coefficients") or {}).get("pct_dose")
        delta_p = (quadratic.get("p") or {}).get("pct_dose")
        jumps = {item["contrast"]: abs(item["difference"] or 0) for item in contrasts_fit}
        largest = max(jumps, key=jumps.get) if jumps else ""
        if gamma2_p is not None and gamma2_p > 0.05 and gamma1_m2 is not None and gamma1_m2 > 0:
            shape = "linear"
        elif gamma2_p is not None and gamma2_p <= 0.05 and largest == "N2->N3":
            shape = "threshold"
        else:
            shape = "other"
        slope_changes = delta_p is not None and delta_p <= 0.05
        m4_gamma = (length_model.get("coefficients") or {}).get("dose")
        m4_se = (length_model.get("se") or {}).get("dose")
        m4_lambda = (length_model.get("coefficients") or {}).get("len")
        m4_pass = _significant_positive(m4_gamma, m4_se)
        finite_p0 = [value for value in p0.values() if isinstance(value, float)]
        p0_range = (max(finite_p0) - min(finite_p0)) if finite_p0 else None
        m3 = p0_range is not None and p0_range <= 0.03
        shapes.append(
            {
                "aic_dose": aic_dose,
                "aic_len": aic_len,
                "m1_pass": m1,
                "m1_trend_positive": trend,
                "m2_shape": shape,
                "m2_slope_changes": slope_changes,
                "m3_p0_range": p0_range,
                "m3_pass": m3,
                "m4_pass": m4_pass,
                "model": model,
                "floor_notes": "; ".join(floor_notes),
                "linear_gamma1": gamma,
                "linear_gamma1_p": two_sided_p(gamma, gamma_se),
                "linear_gamma1_se": gamma_se,
                "linear_delta": (linear.get("coefficients") or {}).get("pct_dose"),
                "linear_delta_p": (linear.get("p") or {}).get("pct_dose"),
                "m2_gamma1": gamma1_m2,
                "m2_gamma1_se": (quadratic.get("se") or {}).get("dose"),
                "m2_gamma1_p": (quadratic.get("p") or {}).get("dose"),
                "m2_gamma2": gamma2,
                "m2_gamma2_se": (quadratic.get("se") or {}).get("dose2"),
                "m2_gamma2_p": gamma2_p,
                "m2_delta": delta,
                "m2_delta_se": (quadratic.get("se") or {}).get("pct_dose"),
                "m2_delta_p": delta_p,
                "m4_gamma1": m4_gamma,
                "m4_gamma1_se": m4_se,
                "m4_gamma1_p": two_sided_p(m4_gamma, m4_se),
                "m4_lambda": m4_lambda,
                "m4_lambda_se": (length_model.get("se") or {}).get("len"),
                "m4_lambda_p": (length_model.get("p") or {}).get("len"),
            }
        )
        checks.append(
            "M1 "
            + ("PASS" if m1 else "FAIL")
            + f" monotonic={monotonic} trend={trend} gamma1={gamma} se={gamma_se} floors={floor_notes or 'none'}"
        )
        checks.append(f"M2 shape={shape} gamma2={gamma2} p={gamma2_p} delta_p={delta_p} slope_changes={slope_changes}")
        checks.append(f"M3 {'PASS' if m3 else 'FAIL'} p0_range={p0_range}")
        checks.append(f"M4 {'PASS' if m4_pass else 'FAIL'} gamma1={m4_gamma} se={m4_se} lambda={m4_lambda} aic_dose={aic_dose} aic_len={aic_len}")
        low_keys = {(str(row["pid"]), str(row["question_id"])) for row in phase1 if row.get("need_state") == "low"}
        high_keys = {(str(row["pid"]), str(row["question_id"])) for row in phase1 if row.get("need_state") == "high"}
        for label, keys, fresh_arm in (("phase1_low_vs_N1", low_keys, "N1"), ("phase1_high_vs_N4", high_keys, "N4")):
            old = [row for row in phase1 if (str(row["pid"]), str(row["question_id"])) in keys]
            new = [row for row in loaded[fresh_arm] if (str(row["pid"]), str(row["question_id"])) in keys]
            old_fit = arm_fit(old)
            new_fit = arm_fit(new)
            replications.append(
                {
                    "comparison": label,
                    "delta_beta": None if old_fit is None or new_fit is None else new_fit["beta"] - old_fit["beta"],
                    "delta_p100": None if old_fit is None or new_fit is None else new_fit["p100"] - old_fit["p100"],
                    "model": model,
                    "n_new": 0 if new_fit is None else new_fit["n"],
                    "n_old": 0 if old_fit is None else old_fit["n"],
                    "n_triplets": len(keys),
                }
            )
        budgets.append({"model": model, **bootstrap_budget_gap(loaded["N4"], loaded["BUD"], seed=42, replicates=1000)})
        checks.append("RESULT " + ("FAIL" if model_fail else "PASS"))
        checks.append("")
    _write(reports / "need_dose_params.csv", params)
    _write(reports / "need_dose_contrast.csv", contrasts)
    _write(reports / "need_dose_shape.csv", shapes)
    _write(reports / "need_dose_replication.csv", replications)
    _write(reports / "budget_contrast.csv", budgets)
    _write(reports / "prompt_lengths.csv", lengths)
    if unmapped_rows:
        _write(reports / "need_dose_unmapped.csv", unmapped_rows)
    (reports / "need_dose_checks.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("\n".join(checks))
    return 0


def _write(path: Path, rows: list[dict]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
