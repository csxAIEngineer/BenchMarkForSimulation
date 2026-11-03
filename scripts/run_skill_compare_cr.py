#!/usr/bin/env python3
"""Income-stratified CR compare: atus_expanded vs atus_expanded_skill."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from popbench.comms import LLMClient, build_provider
from popbench.datasets import DATASETS
from popbench.pipeline import prepare_bundle, simulate, write_results
from popbench.schema import DatasetBundle

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "runs" / "cr-skill-compare-income50"
PROGRESS = OUT_ROOT / "progress.json"
SEED = 42
N = 50
BATCH_SIZE = 15
MODEL = "deepseek-v4-flash"
PROVIDER = "cr"
METHODS = ("atus_expanded", "atus_expanded_skill")


def twin_income(subject) -> str:
    demo = subject.demographics or {}
    for key in ("QID21", "income", "household_income"):
        value = str(demo.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def bdm_income(subject) -> str:
    return str((subject.demographics or {}).get("income") or "unknown").strip() or "unknown"


def stratified_ids(subjects, *, income_fn, n: int, seed: int) -> tuple[list[str], dict]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for subject in subjects:
        by_stratum[income_fn(subject)].append(subject.id)
    rng = random.Random(seed)
    for ids in by_stratum.values():
        rng.shuffle(ids)
    strata = sorted(by_stratum)
    total = sum(len(by_stratum[s]) for s in strata)
    n = min(n, total)
    raw = {s: len(by_stratum[s]) / total * n for s in strata}
    base = {s: min(int(raw[s]), len(by_stratum[s])) for s in strata}
    chosen = sum(base.values())
    remainders = sorted(strata, key=lambda s: (raw[s] - base[s], len(by_stratum[s])), reverse=True)
    idx = 0
    while chosen < n and idx < len(remainders) * 8:
        s = remainders[idx % len(remainders)]
        if base[s] < len(by_stratum[s]):
            base[s] += 1
            chosen += 1
        idx += 1
    picked: list[str] = []
    counts = {}
    for s in strata:
        take = by_stratum[s][: base[s]]
        counts[s] = len(take)
        picked.extend(take)
    rng.shuffle(picked)
    meta = {
        "n_requested": n,
        "n_selected": len(picked),
        "population": total,
        "strata_population": {s: len(by_stratum[s]) for s in strata},
        "strata_selected": counts,
        "seed": seed,
    }
    return picked, meta


def filter_bundle(bundle: DatasetBundle, ids: set[str]) -> DatasetBundle:
    subjects = tuple(s for s in bundle.subjects if s.id in ids)
    cases = tuple(c for c in bundle.cases if c.subject.id in ids)
    return DatasetBundle(
        name=bundle.name,
        unit=bundle.unit,
        source=bundle.source,
        revision=bundle.revision,
        subjects=subjects,
        cases=cases,
    )


def write_progress(**kwargs) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kwargs}
    PROGRESS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def gold_map(bundle: DatasetBundle) -> dict[tuple[str, str], dict]:
    out = {}
    for case in bundle.cases:
        if case.gold is None:
            continue
        out[(case.subject.id, case.question.id)] = {
            "answer": case.gold.answer,
            "distribution": case.gold.distribution_dict(),
            "payload": case.gold.payload,
        }
    return out


def attach_gold(results, gmap) -> list[dict]:
    rows = []
    for result in results:
        row = result.to_dict()
        row["gold"] = gmap.get((result.subject_id, result.question_id))
        rows.append(row)
    return rows


def score_rows(rows: list[dict]) -> dict:
    n = 0
    hit = 0
    missing = 0
    parse_err = 0
    for row in rows:
        if row.get("parse_error"):
            parse_err += 1
        gold = row.get("gold") or {}
        human = gold.get("answer")
        parsed = row.get("parsed") or {}
        pred = parsed.get("answer")
        if human is None or pred is None:
            missing += 1
            continue
        n += 1
        if str(human).strip().casefold() == str(pred).strip().casefold():
            hit += 1
    return {
        "n_scored": n,
        "accuracy": (hit / n) if n else None,
        "hits": hit,
        "missing_or_unscored": missing,
        "parse_errors": parse_err,
        "n_rows": len(rows),
    }


def run_one(bundle: DatasetBundle, *, method: str, out_dir: Path, job: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "responses.jsonl"
    if done.is_file() and done.stat().st_size > 0 and (out_dir / "manifest.json").is_file():
        print(f"skip existing {out_dir}", flush=True)
        summary = json.loads((out_dir / "run_summary.json").read_text()) if (out_dir / "run_summary.json").is_file() else {"method": method, "out": str(out_dir), "skipped": True}
        return summary
    provider = build_provider(PROVIDER)
    client = LLMClient(provider, model=MODEL, cache_dir=out_dir / "cache", temperature=0)
    t0 = time.time()
    write_progress(status="running", job=job, method=method, dataset=bundle.name, n_cases=len(bundle.cases), n_subjects=len(bundle.subjects))
    results = simulate(bundle, method=method, client=client, limit=None, seed=SEED, batch_size=BATCH_SIZE)
    gmap = gold_map(bundle)
    rows = attach_gold(results, gmap)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    metrics = score_rows(rows)
    manifest = {
        "dataset": bundle.name,
        "method": method,
        "provider": PROVIDER,
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "n_results": len(results),
        "n_subjects": len(bundle.subjects),
        "n_cases": len(bundle.cases),
        "seconds": round(time.time() - t0, 1),
        "seed": SEED,
        "metrics": metrics,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summary = {"job": job, "method": method, "dataset": bundle.name, **metrics, "seconds": manifest["seconds"], "out": str(out_dir)}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    write_progress(status="job_done", **summary)
    return summary


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("CR_API_KEY", "").strip():
        raise SystemExit("CR_API_KEY missing")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    twin = DATASETS["twin2k_500"].load(data_dir)
    twin_ids, twin_meta = stratified_ids(twin.subjects, income_fn=twin_income, n=N, seed=SEED)
    twin_bundle = filter_bundle(twin, set(twin_ids))
    (OUT_ROOT / "twin2k_sample.json").write_text(
        json.dumps({"ids": twin_ids, **twin_meta, "income_field": "QID21"}, ensure_ascii=False, indent=2) + "\n"
    )

    bdm = DATASETS["bdm_lettuce"].load(data_dir)
    bdm_ids, bdm_meta = stratified_ids(bdm.subjects, income_fn=bdm_income, n=N, seed=SEED)
    bdm_bundle = filter_bundle(bdm, set(bdm_ids))
    (OUT_ROOT / "bdm_sample.json").write_text(
        json.dumps({"ids": bdm_ids, **bdm_meta, "income_field": "income"}, ensure_ascii=False, indent=2) + "\n"
    )

    mega_raw = DATASETS["twin2k_mega_study"].load(data_dir)
    mega_prepared = prepare_bundle(mega_raw, method="atus_expanded", twin=twin)
    # keep subjects that are in the twin income sample and have persona after attach
    mega_bundle = filter_bundle(mega_prepared, set(twin_ids))
    if len(mega_bundle.subjects) < N:
        # top up from attached mega subjects by twin income among available
        avail = [s for s in mega_prepared.subjects if twin_income(s) != "unknown" or True]
        extra_ids, mega_meta = stratified_ids(avail, income_fn=twin_income, n=N, seed=SEED + 1)
        mega_bundle = filter_bundle(mega_prepared, set(extra_ids))
    else:
        mega_meta = {"n_selected": len(mega_bundle.subjects), "note": "filtered to twin income sample"}
    (OUT_ROOT / "mega_sample.json").write_text(
        json.dumps({"ids": [s.id for s in mega_bundle.subjects], **mega_meta}, ensure_ascii=False, indent=2) + "\n"
    )

    jobs = [
        ("twin2k_500", twin_bundle),
        ("bdm_lettuce", bdm_bundle),
        ("twin2k_mega_study", mega_bundle),
    ]
    overall = {
        "model": MODEL,
        "provider": PROVIDER,
        "seed": SEED,
        "n_people": N,
        "batch_size": BATCH_SIZE,
        "methods": list(METHODS),
        "datasets": {
            "twin2k_500": {"subjects": len(twin_bundle.subjects), "cases": len(twin_bundle.cases)},
            "bdm_lettuce": {"subjects": len(bdm_bundle.subjects), "cases": len(bdm_bundle.cases)},
            "twin2k_mega_study": {"subjects": len(mega_bundle.subjects), "cases": len(mega_bundle.cases)},
        },
        "runs": [],
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n")
    write_progress(status="started", datasets=overall["datasets"], methods=list(METHODS))
    print("sample ready", json.dumps(overall["datasets"]), flush=True)

    for dataset_name, bundle in jobs:
        for method in METHODS:
            job = f"{dataset_name}__{method}"
            summary = run_one(bundle, method=method, out_dir=OUT_ROOT / dataset_name / method, job=job)
            overall["runs"].append(summary)
            (OUT_ROOT / "manifest.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n")

    # comparison table
    compare = []
    for dataset_name, _bundle in jobs:
        by_method = {}
        for method in METHODS:
            path = OUT_ROOT / dataset_name / method / "run_summary.json"
            if path.is_file():
                by_method[method] = json.loads(path.read_text())
        if len(by_method) == 2:
            a = by_method["atus_expanded"]
            b = by_method["atus_expanded_skill"]
            compare.append(
                {
                    "dataset": dataset_name,
                    "atus_expanded_accuracy": a.get("accuracy"),
                    "atus_expanded_skill_accuracy": b.get("accuracy"),
                    "delta_skill_minus_base": (
                        None
                        if a.get("accuracy") is None or b.get("accuracy") is None
                        else round(b["accuracy"] - a["accuracy"], 4)
                    ),
                    "n_scored_base": a.get("n_scored"),
                    "n_scored_skill": b.get("n_scored"),
                }
            )
    (OUT_ROOT / "compare_vs_human.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2) + "\n")
    overall["compare_vs_human"] = compare
    overall["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (OUT_ROOT / "manifest.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n")
    write_progress(status="finished", compare=compare)
    print("ALL_DONE", json.dumps(compare, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
