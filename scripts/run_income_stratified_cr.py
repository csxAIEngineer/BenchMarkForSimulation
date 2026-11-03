#!/usr/bin/env python3
"""Income-stratified CR simulation for methods 1/2/3/4/6 (no SimBench)."""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from popbench.comms import LLMClient, build_provider
from popbench.datasets import DATASETS
from popbench.pipeline import simulate, write_results
from popbench.schema import DatasetBundle

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "runs" / "cr-deepseek-v4-flash-income100"
SEED = 42
N = 30
BATCH_SIZE = 15
MODEL = "deepseek-v4-flash"
PROVIDER = "cr"

TWIN_METHODS = (
    "demographics_only",
    "full_persona",
    "persona_summary",
    "atus_expanded",
)


def twin_income(subject) -> str:
    demo = subject.demographics or {}
    for key in ("QID21", "income", "household_income"):
        value = str(demo.get(key) or "").strip()
        if value:
            return value
    text = subject.demographic_text or ""
    for line in text.splitlines():
        if "answer:" in line.lower() and any(tok in line for tok in ("$", "income", "000")):
            return line.split(":", 1)[-1].strip()
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
    if not strata:
        raise SystemExit("no subjects for stratified sample")
    # proportional allocation, at least 0; then fill remainder by largest remainders
    total = sum(len(by_stratum[s]) for s in strata)
    n = min(n, total)
    raw = {s: len(by_stratum[s]) / total * n for s in strata}
    base = {s: int(raw[s]) for s in strata}
    # ensure we don't request more than available
    for s in strata:
        base[s] = min(base[s], len(by_stratum[s]))
    chosen = sum(base.values())
    remainders = sorted(strata, key=lambda s: (raw[s] - base[s], len(by_stratum[s])), reverse=True)
    idx = 0
    while chosen < n and idx < len(remainders) * 5:
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


def run_one(bundle: DatasetBundle, *, method: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = build_provider(PROVIDER)
    client = LLMClient(
        provider,
        model=MODEL,
        cache_dir=out_dir / "cache",
        temperature=0,
    )
    t0 = time.time()
    results = simulate(bundle, method=method, client=client, limit=None, seed=SEED, batch_size=BATCH_SIZE)
    write_results(
        out_dir,
        results,
        manifest={
            "dataset": bundle.name,
            "method": method,
            "provider": PROVIDER,
            "model": MODEL,
            "n_results": len(results),
            "n_subjects": len(bundle.subjects),
            "n_cases": len(bundle.cases),
            "seconds": round(time.time() - t0, 1),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
        },
    )
    cached = sum(1 for r in results if r.cached)
    errors = sum(1 for r in results if r.parse_error)
    summary = {
        "method": method,
        "dataset": bundle.name,
        "n_results": len(results),
        "cached": cached,
        "parse_errors": errors,
        "out": str(out_dir),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
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
        json.dumps({"ids": twin_ids, **twin_meta, "income_field": "QID21"}, indent=2) + "\n"
    )

    bdm = DATASETS["bdm_lettuce"].load(data_dir)
    bdm_ids, bdm_meta = stratified_ids(bdm.subjects, income_fn=bdm_income, n=N, seed=SEED)
    bdm_bundle = filter_bundle(bdm, set(bdm_ids))
    (OUT_ROOT / "bdm_sample.json").write_text(
        json.dumps({"ids": bdm_ids, **bdm_meta, "income_field": "income"}, indent=2) + "\n"
    )

    overall = {
        "model": MODEL,
        "provider": PROVIDER,
        "seed": SEED,
            "batch_size": BATCH_SIZE,
        "n_people": N,
        "twin2k_cases": len(twin_bundle.cases),
        "bdm_cases": len(bdm_bundle.cases),
        "twin2k_strata": twin_meta["strata_selected"],
        "bdm_strata": bdm_meta["strata_selected"],
        "runs": [],
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(
        f"sample ready twin_subjects={len(twin_ids)} twin_cases={len(twin_bundle.cases)} "
        f"bdm_subjects={len(bdm_ids)} bdm_cases={len(bdm_bundle.cases)}",
        flush=True,
    )

    for method in TWIN_METHODS:
        summary = run_one(twin_bundle, method=method, out_dir=OUT_ROOT / method)
        overall["runs"].append(summary)
        (OUT_ROOT / "manifest.json").write_text(json.dumps(overall, indent=2) + "\n")

    summary = run_one(
        bdm_bundle,
        method="bdm_survey_persona",
        out_dir=OUT_ROOT / "bdm_survey_persona",
    )
    overall["runs"].append(summary)
    overall["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (OUT_ROOT / "manifest.json").write_text(json.dumps(overall, indent=2) + "\n")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
