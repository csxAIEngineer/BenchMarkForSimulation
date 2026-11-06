#!/usr/bin/env python3
"""CR multi-model simulation on pricing-only items.

Subjects: income-stratified n=50, seed 42 (same allocator as
``run_skill_compare_cr.py``). Method: ``atus_expanded`` only.
Datasets: ``data/pricing_only`` twin2k QID9_* and bdm lettuce WTP+purchase.
Mega-study is omitted (no structured pricing items).

Re-running this file skips a job whose responses.jsonl and manifest.json
already exist. Successful batches stay in that job's cache/, so a CR
timeout can be resumed by starting the same command again.
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

from popbench.comms import LLMClient, build_provider  # noqa: E402
from popbench.comms.errors import CommsTransportError  # noqa: E402
from popbench.datasets.bdm_lettuce.loader import bundle_from_records as bdm_bundle  # noqa: E402
from popbench.datasets.io import read_jsonl  # noqa: E402
from popbench.datasets.twin2k_500.loader import bundle_from_records as twin_bundle  # noqa: E402
from popbench.pipeline import simulate  # noqa: E402
from pricing_models_cr import (  # noqa: E402
    BATCH_SIZE,
    BDM_PRICING_PATH,
    BDM_QUESTIONS_PER_SUBJECT,
    DATASETS,
    METHOD,
    MODEL_IDS,
    MODELS,
    N_SUBJECTS,
    OUT_ROOT,
    PROVIDER,
    REQUEST_TIMEOUT,
    SEED,
    TWIN_PRICING_PATH,
    TWIN_QUESTIONS_PER_SUBJECT,
    assert_models_documented,
    assert_uniform_question_count,
    filter_questions,
    is_bdm_pricing,
    is_twin_pricing,
    job_dir,
    job_is_complete,
    planned_batches,
)
from run_skill_compare_cr import (  # noqa: E402
    attach_gold,
    bdm_income,
    filter_bundle,
    gold_map,
    score_rows,
    stratified_ids,
    twin_income,
)

PROGRESS = OUT_ROOT / "progress.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def write_progress(**kwargs: object) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "out_root": str(OUT_ROOT),
        "method": METHOD,
        "seed": SEED,
        "n_subjects": N_SUBJECTS,
        "models": list(MODEL_IDS),
        **kwargs,
    }
    write_json(PROGRESS, payload)


def load_pricing(dataset: str):
    if dataset == "twin2k_500":
        print(f"loading {TWIN_PRICING_PATH.name}", flush=True)
        bundle = filter_questions(twin_bundle(read_jsonl(TWIN_PRICING_PATH)), is_twin_pricing)
        assert_uniform_question_count(bundle, TWIN_QUESTIONS_PER_SUBJECT)
        return bundle
    print(f"loading {BDM_PRICING_PATH.name}", flush=True)
    bundle = filter_questions(bdm_bundle(read_jsonl(BDM_PRICING_PATH)), is_bdm_pricing)
    assert_uniform_question_count(bundle, BDM_QUESTIONS_PER_SUBJECT)
    return bundle


def sample_jobs() -> list[tuple[str, object, dict]]:
    twin = load_pricing("twin2k_500")
    twin_ids, twin_meta = stratified_ids(twin.subjects, income_fn=twin_income, n=N_SUBJECTS, seed=SEED)
    twin_sample = filter_bundle(twin, set(twin_ids))
    write_json(OUT_ROOT / "twin2k_sample.json", {"ids": twin_ids, **twin_meta, "income_field": "QID21"})

    bdm = load_pricing("bdm_lettuce")
    bdm_ids, bdm_meta = stratified_ids(bdm.subjects, income_fn=bdm_income, n=N_SUBJECTS, seed=SEED)
    bdm_sample = filter_bundle(bdm, set(bdm_ids))
    write_json(OUT_ROOT / "bdm_sample.json", {"ids": bdm_ids, **bdm_meta, "income_field": "income"})
    return [
        ("twin2k_500", twin_sample, twin_meta),
        ("bdm_lettuce", bdm_sample, bdm_meta),
    ]


class BatchProgress:
    """Count batches and keep going after a transport failure.

    Successful batches are cached by ``LLMClient``. A failed batch is not
    written into responses.jsonl, so the next launch retries it.
    """

    def __init__(self, inner: LLMClient, *, on_batch, on_error) -> None:
        self._inner = inner
        self._on_batch = on_batch
        self._on_error = on_error
        self.failed = 0

    def complete_batch(self, packages, *, sample_seed: int = 0):
        started = time.time()
        try:
            rows = self._inner.complete_batch(packages, sample_seed=sample_seed)
        except CommsTransportError as exc:
            self.failed += 1
            self._on_error(packages, exc)
            return ()
        self._on_batch(packages, rows, time.time() - started)
        return rows


def run_one(bundle, *, model: str, dataset: str) -> dict:
    out_dir = job_dir(model, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = TWIN_QUESTIONS_PER_SUBJECT if dataset == "twin2k_500" else BDM_QUESTIONS_PER_SUBJECT
    batches_total = planned_batches(len(bundle.subjects), questions, BATCH_SIZE)
    if job_is_complete(out_dir):
        print(f"skip existing {out_dir}", flush=True)
        summary_path = out_dir / "run_summary.json"
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"model": model, "dataset": dataset, "skipped": True, "out": str(out_dir)}

    provider = build_provider(PROVIDER, timeout=REQUEST_TIMEOUT)
    client = LLMClient(provider, model=model, cache_dir=out_dir / "cache", temperature=0)
    state = {"batches_done": 0, "cases_done": 0, "live_seconds": [], "errors": []}

    def on_error(packages, exc: CommsTransportError) -> None:
        question_ids = [package.question_id for package in packages]
        state["errors"].append(
            {
                "subject_id": packages[0].subject_id,
                "question_ids": question_ids,
                "error": str(exc)[:500],
            }
        )
        print(
            f"batch failed model={model} dataset={dataset} subject={packages[0].subject_id} {exc}",
            flush=True,
        )
        write_progress(
            status="batch_error",
            model=model,
            dataset=dataset,
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
            f"batch {state['batches_done']}/{batches_total} model={model} dataset={dataset} "
            f"cached={cached} eta_s={eta}",
            flush=True,
        )
        write_progress(
            status="running",
            model=model,
            dataset=dataset,
            batches_done=state["batches_done"],
            batches_total=batches_total,
            batches_failed=len(state["errors"]),
            cases_done=state["cases_done"],
            eta_seconds_if_remaining_uncached=eta,
        )

    write_progress(
        status="running",
        model=model,
        dataset=dataset,
        batches_done=0,
        batches_total=batches_total,
        n_subjects=len(bundle.subjects),
        n_cases=len(bundle.cases),
    )
    started = time.time()
    progress = BatchProgress(client, on_batch=on_batch, on_error=on_error)
    results = simulate(bundle, method=METHOD, client=progress, limit=None, seed=SEED, batch_size=BATCH_SIZE)
    if progress.failed or len(results) != len(bundle.cases):
        write_json(out_dir / "batch_errors.json", {"model": model, "dataset": dataset, "errors": state["errors"]})
        summary = {
            "ok": False,
            "model": model,
            "dataset": dataset,
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
        "dataset": dataset,
        "method": METHOD,
        "provider": PROVIDER,
        "model": model,
        "batch_size": BATCH_SIZE,
        "request_timeout": REQUEST_TIMEOUT,
        "n_results": len(results),
        "n_subjects": len(bundle.subjects),
        "n_cases": len(bundle.cases),
        "seconds": round(time.time() - started, 1),
        "seed": SEED,
        "temperature": 0,
        "operational": score_rows(rows),
    }
    write_json(out_dir / "manifest.json", manifest)
    responses_tmp.replace(out_dir / "responses.jsonl")
    summary = {
        "ok": True,
        "job": f"{dataset}__{model}",
        "model": model,
        "dataset": dataset,
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
    assert_models_documented()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_progress(status="loading", git_rev=git_rev(), model_sources=list(MODELS))
    print("model ids " + ", ".join(MODEL_IDS), flush=True)
    jobs = sample_jobs()
    overall = {
        "provider": PROVIDER,
        "method": METHOD,
        "seed": SEED,
        "n_subjects": N_SUBJECTS,
        "batch_size": BATCH_SIZE,
        "request_timeout": REQUEST_TIMEOUT,
        "git_rev": git_rev(),
        "models": list(MODELS),
        "datasets": {
            name: {"subjects": len(bundle.subjects), "cases": len(bundle.cases)}
            for name, bundle, _meta in jobs
        },
        "excluded": {"twin2k_mega_study": "no structured pricing items"},
        "runs": [],
    }
    write_json(OUT_ROOT / "manifest.json", overall)
    write_progress(status="started", datasets=overall["datasets"])
    print("sample ready " + json.dumps(overall["datasets"]), flush=True)

    failures: list[dict] = []
    for model in MODEL_IDS:
        for dataset, bundle, _meta in jobs:
            try:
                summary = run_one(bundle, model=model, dataset=dataset)
            except Exception as exc:
                traceback.print_exc()
                summary = {
                    "ok": False,
                    "model": model,
                    "dataset": dataset,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                write_progress(status="job_error", **summary)
            overall["runs"].append(summary)
            write_json(OUT_ROOT / "manifest.json", overall)
            if not summary.get("ok", True):
                failures.append(summary)

    if failures:
        write_progress(status="incomplete", failures=failures)
        print("INCOMPLETE " + json.dumps(failures, ensure_ascii=False), flush=True)
        return 1
    overall["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(OUT_ROOT / "manifest.json", overall)
    write_progress(status="all_done", runs=overall["runs"])
    print("ALL_DONE " + json.dumps(overall["runs"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
