#!/usr/bin/env python3
"""Twin2K pricing QID9 n=50 with QID-mapped atus_expanded cards (deepseek-v4-pro only).

Same sample allocator as runs/cr-pricing-models-n50 (seed=42, income-stratified).
Writes a NEW tree under runs/twin2k-pricing-fixedcards-deepseek-n50 — does not
touch the broken-card results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

import pricing_models_cr as cfg  # noqa: E402

OUT_ROOT = ROOT / "runs" / "twin2k-pricing-fixedcards-deepseek-n50"
LOG_PATH = ROOT / "logs" / "twin2k-pricing-fixedcards-deepseek-n50.log"
MODEL = "deepseek-v4-pro"
DATASET = "twin2k_500"
METHOD = cfg.METHOD
SEED = cfg.SEED
N_SUBJECTS = cfg.N_SUBJECTS

# Point shared helpers at the new run tree.
cfg.OUT_ROOT = OUT_ROOT
cfg.LOG_PATH = LOG_PATH
cfg.MODELS = (
    {
        "id": MODEL,
        "label": "deepseek v4 pro",
        "source": "fixed-card Twin2K pricing-only rerun",
    },
)
cfg.MODEL_IDS = (MODEL,)
cfg.DATASETS = (DATASET,)

from popbench.comms import LLMClient, build_provider  # noqa: E402
from popbench.comms.errors import CommsTransportError  # noqa: E402
from popbench.datasets.io import read_jsonl  # noqa: E402
from popbench.datasets.twin2k_500.loader import bundle_from_records as twin_bundle  # noqa: E402
from popbench.models.atus_expanded import render_atus_persona  # noqa: E402
from popbench.pipeline import simulate  # noqa: E402
from pricing_models_cr import (  # noqa: E402
    BATCH_SIZE,
    PROVIDER,
    REQUEST_TIMEOUT,
    TWIN_PRICING_PATH,
    TWIN_QUESTIONS_PER_SUBJECT,
    assert_models_documented,
    assert_uniform_question_count,
    filter_questions,
    is_twin_pricing,
    job_dir,
    job_is_complete,
    planned_batches,
)
from run_pricing_models_cr import BatchProgress, write_json, write_progress  # noqa: E402
from run_skill_compare_cr import (  # noqa: E402
    attach_gold,
    filter_bundle,
    gold_map,
    score_rows,
    stratified_ids,
    twin_income,
)
import run_pricing_models_cr as runner  # noqa: E402

# Keep progress.json under the new OUT_ROOT.
runner.OUT_ROOT = OUT_ROOT
runner.PROGRESS = OUT_ROOT / "progress.json"
runner.SEED = SEED
runner.N_SUBJECTS = N_SUBJECTS
runner.METHOD = METHOD
runner.MODEL_IDS = (MODEL,)
runner.MODELS = cfg.MODELS
runner.PROVIDER = PROVIDER
runner.BATCH_SIZE = BATCH_SIZE
runner.REQUEST_TIMEOUT = REQUEST_TIMEOUT


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export_as_sent_cards(bundle, sample_ids: list[str]) -> dict:
    """Write the Persona blocks that atus_expanded would send (questions omitted)."""
    out = OUT_ROOT / "post-check"
    out.mkdir(parents=True, exist_ok=True)
    by_id = {subject.id: subject for subject in bundle.subjects}
    # One case per subject is enough to render the card.
    case_by_subject = {}
    for case in bundle.cases:
        case_by_subject.setdefault(case.subject.id, case)

    rows = []
    lengths = []
    for index, subject_id in enumerate(sample_ids, start=1):
        case = case_by_subject[subject_id]
        text = render_atus_persona(case, seed=SEED)
        lengths.append(len(text))
        path = out / f"{index:02d}_{subject_id}_persona.txt"
        path.write_text(text + "\n", encoding="utf-8")
        rows.append(
            {
                "index": index,
                "subject_id": subject_id,
                "char_len": len(text),
                "persona": text,
            }
        )
    jsonl = out / "all_50_personas.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    texts = [row["persona"] for row in rows]
    unique = len(set(texts))
    meta = {
        "n": len(rows),
        "method": METHOD,
        "seed": SEED,
        "model": MODEL,
        "git_rev": git_rev(),
        "subjects_path": str(TWIN_PRICING_PATH),
        "sample_path": str(OUT_ROOT / "twin2k_sample.json"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timezone": "Asia/Shanghai",
        "note": (
            "Persona blocks as sent with fixed Twin2K QID→demographics mapping "
            "(atus_expanded). Questions omitted."
        ),
        "char_len": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
        },
        "unique_persona_texts": unique,
        "first_subject_id": sample_ids[0],
        "last_subject_id": sample_ids[-1],
        "first_card_preview": texts[0],
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Mirror for audit convenience.
    mirror = ROOT / "exports" / "post-check-fixed"
    if mirror.exists() or True:
        mirror.mkdir(parents=True, exist_ok=True)
        for path in out.iterdir():
            if path.is_file():
                (mirror / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return meta


def run_one(bundle) -> dict:
    out_dir = job_dir(MODEL, DATASET, root=OUT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches_total = planned_batches(len(bundle.subjects), TWIN_QUESTIONS_PER_SUBJECT, BATCH_SIZE)
    if job_is_complete(out_dir):
        print(f"skip existing {out_dir}", flush=True)
        summary_path = out_dir / "run_summary.json"
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"model": MODEL, "dataset": DATASET, "skipped": True, "out": str(out_dir)}

    provider = build_provider(PROVIDER, timeout=REQUEST_TIMEOUT)
    client = LLMClient(provider, model=MODEL, cache_dir=out_dir / "cache", temperature=0)
    state = {"batches_done": 0, "cases_done": 0, "live_seconds": [], "errors": []}

    def on_error(packages, exc: CommsTransportError) -> None:
        state["errors"].append(
            {
                "subject_id": packages[0].subject_id,
                "question_ids": [package.question_id for package in packages],
                "error": str(exc)[:500],
            }
        )
        print(
            f"batch failed model={MODEL} dataset={DATASET} subject={packages[0].subject_id} {exc}",
            flush=True,
        )
        write_progress(
            status="batch_error",
            model=MODEL,
            dataset=DATASET,
            batches_done=state["batches_done"],
            batches_total=batches_total,
            batches_failed=len(state["errors"]),
        )

    def on_batch(packages, rows, elapsed: float) -> None:
        state["batches_done"] += 1
        state["cases_done"] += len(rows)
        cached = bool(rows) and all(row.cached for row in rows)
        if not cached:
            state["live_seconds"].append(elapsed)
        remaining = batches_total - state["batches_done"]
        eta = None
        if state["live_seconds"] and remaining:
            eta = round((sum(state["live_seconds"]) / len(state["live_seconds"])) * remaining, 1)
        print(
            f"batch {state['batches_done']}/{batches_total} model={MODEL} dataset={DATASET} "
            f"cached={cached} eta_s={eta}",
            flush=True,
        )
        write_progress(
            status="running",
            model=MODEL,
            dataset=DATASET,
            batches_done=state["batches_done"],
            batches_total=batches_total,
            batches_failed=len(state["errors"]),
            cases_done=state["cases_done"],
            eta_seconds_if_remaining_uncached=eta,
        )

    write_progress(
        status="running",
        model=MODEL,
        dataset=DATASET,
        batches_done=0,
        batches_total=batches_total,
        n_subjects=len(bundle.subjects),
        n_cases=len(bundle.cases),
    )
    started = time.time()
    progress = BatchProgress(client, on_batch=on_batch, on_error=on_error)
    results = simulate(bundle, method=METHOD, client=progress, limit=None, seed=SEED, batch_size=BATCH_SIZE)
    if progress.failed or len(results) != len(bundle.cases):
        write_json(out_dir / "batch_errors.json", {"model": MODEL, "dataset": DATASET, "errors": state["errors"]})
        summary = {
            "ok": False,
            "model": MODEL,
            "dataset": DATASET,
            "method": METHOD,
            "failed_batches": progress.failed,
            "n_results": len(results),
            "n_cases": len(bundle.cases),
            "out": str(out_dir),
            "seconds": round(time.time() - started, 1),
        }
        write_progress(status="job_incomplete", **summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary

    rows = attach_gold(results, gold_map(bundle))
    responses_tmp = out_dir / "responses.jsonl.tmp"
    responses_tmp.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    manifest = {
        "dataset": DATASET,
        "method": METHOD,
        "provider": PROVIDER,
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "request_timeout": REQUEST_TIMEOUT,
        "n_results": len(results),
        "n_subjects": len(bundle.subjects),
        "n_cases": len(bundle.cases),
        "seconds": round(time.time() - started, 1),
        "seed": SEED,
        "temperature": 0,
        "git_rev": git_rev(),
        "card_fix": "twin2k_qid_mapped_atus_expanded",
        "operational": score_rows(rows),
    }
    write_json(out_dir / "manifest.json", manifest)
    responses_tmp.replace(out_dir / "responses.jsonl")
    summary = {
        "ok": True,
        "job": f"{DATASET}__{MODEL}",
        "model": MODEL,
        "dataset": DATASET,
        "method": METHOD,
        "n_rows": len(rows),
        "seconds": manifest["seconds"],
        "out": str(out_dir),
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    write_progress(status="job_done", **summary)
    return summary


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("CR_API_KEY", "").strip():
        raise SystemExit("CR_API_KEY missing")
    assert_models_documented((MODEL,))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_progress(status="loading", git_rev=git_rev(), model=MODEL, dataset=DATASET)
    print(f"loading {TWIN_PRICING_PATH.name}", flush=True)
    bundle = filter_questions(twin_bundle(read_jsonl(TWIN_PRICING_PATH)), is_twin_pricing)
    assert_uniform_question_count(bundle, TWIN_QUESTIONS_PER_SUBJECT)
    twin_ids, twin_meta = stratified_ids(
        bundle.subjects, income_fn=twin_income, n=N_SUBJECTS, seed=SEED
    )
    sample = filter_bundle(bundle, set(twin_ids))
    write_json(OUT_ROOT / "twin2k_sample.json", {"ids": twin_ids, **twin_meta, "income_field": "QID21"})

    # Sanity: same 50 as prior broken-card run when that file exists.
    prior = ROOT / "runs" / "cr-pricing-models-n50" / "twin2k_sample.json"
    if prior.is_file():
        prior_ids = json.loads(prior.read_text(encoding="utf-8"))["ids"]
        match = prior_ids == twin_ids
        write_json(
            OUT_ROOT / "sample_match_prior.json",
            {"matches_cr_pricing_models_n50": match, "n": len(twin_ids)},
        )
        print(f"sample match prior cr-pricing-models-n50: {match}", flush=True)

    card_meta = export_as_sent_cards(sample, twin_ids)
    print(
        f"exported {card_meta['n']} cards unique={card_meta['unique_persona_texts']} "
        f"len={card_meta['char_len']}",
        flush=True,
    )
    print("--- first card ---\n" + card_meta["first_card_preview"] + "\n--- end ---", flush=True)

    overall = {
        "provider": PROVIDER,
        "method": METHOD,
        "seed": SEED,
        "n_subjects": N_SUBJECTS,
        "batch_size": BATCH_SIZE,
        "request_timeout": REQUEST_TIMEOUT,
        "git_rev": git_rev(),
        "models": list(cfg.MODELS),
        "datasets": {DATASET: {"subjects": len(sample.subjects), "cases": len(sample.cases)}},
        "card_fix": "twin2k_qid_mapped_atus_expanded",
        "compare_to": "runs/cr-pricing-models-n50/deepseek-v4-pro/twin2k_500/atus_expanded",
        "runs": [],
    }
    write_json(OUT_ROOT / "manifest.json", overall)
    write_progress(status="started", datasets=overall["datasets"], card_meta=card_meta["char_len"])
    print("sample ready " + json.dumps(overall["datasets"]), flush=True)

    try:
        summary = run_one(sample)
    except Exception as exc:
        traceback.print_exc()
        summary = {
            "ok": False,
            "model": MODEL,
            "dataset": DATASET,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_progress(status="job_error", **summary)

    overall["runs"].append(summary)
    write_json(OUT_ROOT / "manifest.json", overall)
    if not summary.get("ok", True):
        write_progress(status="incomplete", failures=[summary])
        print("INCOMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
        return 1

    overall["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(OUT_ROOT / "manifest.json", overall)
    write_progress(status="all_done", runs=overall["runs"])
    print("ALL_DONE " + json.dumps(overall["runs"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
