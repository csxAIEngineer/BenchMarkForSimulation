#!/usr/bin/env python3
"""ATUS adults → QID9, deepseek-v4-pro, n=50.

Each subject answers all 40 QID9 questions in ONE CR batch
(simulate groups by subject; batch_size=40).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from popbench.comms import LLMClient, build_provider  # noqa: E402
from popbench.comms.errors import CommsTransportError  # noqa: E402
from popbench.datasets.twin2k_500.loader import (  # noqa: E402
    _optional_text,
    _question_kind,
    _reject_gold_in_prompt,
    _routine,
    _string_map,
)
from popbench.pipeline import simulate  # noqa: E402
from popbench.datasets.io import DatasetError  # noqa: E402
from popbench.schema import (  # noqa: E402
    Case,
    DatasetBundle,
    Gold,
    Question,
    Subject,
)
from run_skill_compare_cr import attach_gold, gold_map, score_rows  # noqa: E402

OUT_ROOT = ROOT / "runs" / "atus-qid9-deepseek-n50"
SUBJECTS_PATH = ROOT / "data" / "atus_qid9" / "atus_qid9_subjects.jsonl"
HUMAN_SHARE_PATH = ROOT / "data" / "atus_qid9" / "twin2k_human_share_qid9.json"
PROGRESS = OUT_ROOT / "progress.json"

MODEL = "deepseek-v4-pro"
PROVIDER = "cr"
METHOD = "atus_expanded"
DATASET = "atus_qid9"
SEED = 42
N_SUBJECTS = 50
BATCH_SIZE = 40  # all QID9_1..40 in one request per subject
REQUEST_TIMEOUT = 600
QUESTIONS_PER_SUBJECT = 40

YES_LABEL = "Yes, I would purchase the product"
NO_LABEL = "No, I would not purchase the product"


def write_json(path: Path, payload: object) -> None:
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
        "model": MODEL,
        "seed": SEED,
        "n_subjects": N_SUBJECTS,
        "batch_size": BATCH_SIZE,
        "note": "one CR call per subject (all 40 QID9)",
        **kwargs,
    }
    write_json(PROGRESS, payload)


def _subject_and_cases(record: dict[str, Any]) -> tuple[Subject, list[Case]]:
    subject_id = str(record.get("id") or "").strip()
    if not subject_id:
        raise DatasetError("ATUS QID9 record is missing id")
    demographics = _string_map(record.get("demographics"))
    subject = Subject(
        id=subject_id,
        dataset=DATASET,
        unit="individual",
        demographics=demographics,
        persona_text=_optional_text(record.get("persona_text")),
        persona_summary=_optional_text(record.get("persona_summary")),
        demographic_text=_optional_text(record.get("demographic_text")),
        routine=_routine(record.get("routine")),
    )
    cases: list[Case] = []
    raw_questions = record.get("questions") or []
    if not isinstance(raw_questions, list):
        raise DatasetError(f"{subject_id} questions must be a list")
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise DatasetError(f"{subject_id} has a question that is not an object")
        qid = str(raw.get("id") or "")
        if not qid.startswith("QID9_"):
            continue
        question = Question(
            id=qid,
            prompt=str(raw.get("prompt") or ""),
            options=tuple(str(option) for option in raw.get("options") or []),
            kind=_question_kind(raw.get("kind")),
        )
        gold_answer = raw.get("gold")
        gold = None if gold_answer in (None, "") else Gold(answer=str(gold_answer))
        _reject_gold_in_prompt(question, gold)
        cases.append(Case(dataset=DATASET, subject=subject, question=question, gold=gold))
    return subject, cases


def bundle_from_records(records: list[dict[str, Any]], *, source: str) -> DatasetBundle:
    subjects: list[Subject] = []
    cases: list[Case] = []
    for record in records:
        subject, subject_cases = _subject_and_cases(record)
        subjects.append(subject)
        cases.extend(subject_cases)
    return DatasetBundle(
        name=DATASET,
        unit="individual",
        source=source,
        revision=None,
        subjects=tuple(subjects),
        cases=tuple(cases),
    )


def sample_records(path: Path, *, n: int, seed: int) -> tuple[list[dict[str, Any]], list[str], dict]:
    """Weight-aware sample with TUFINLWGT/weight when present."""
    print(f"streaming {path}", flush=True)
    ids: list[str] = []
    weights: list[float] = []
    offset = 0
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("id") or "").strip()
            if not sid:
                continue
            w = rec.get("TUFINLWGT", rec.get("weight"))
            try:
                wf = float(w) if w is not None else 0.0
            except (TypeError, ValueError):
                wf = 0.0
            if wf < 0:
                wf = 0.0
            ids.append(sid)
            weights.append(wf)
            offsets.append(start)
            offset += 1
    total = len(ids)
    if total < n:
        raise SystemExit(f"only {total} subjects available, need {n}")
    rng = random.Random(seed)
    has_weight = any(w > 0 for w in weights)
    if has_weight:
        # random.choices without replacement via cumulative draw
        chosen_idx: list[int] = []
        pool_idx = list(range(total))
        pool_w = list(weights)
        for _ in range(n):
            # zero-weight fallback to uniform among remaining
            if sum(pool_w) <= 0:
                pick = rng.randrange(len(pool_idx))
            else:
                pick = rng.choices(range(len(pool_idx)), weights=pool_w, k=1)[0]
            chosen_idx.append(pool_idx[pick])
            del pool_idx[pick]
            del pool_w[pick]
        chosen_idx.sort()
        meta = {"sampling": "weight_without_replacement", "weight_field": "TUFINLWGT|weight"}
    else:
        chosen_idx = sorted(rng.sample(range(total), n))
        meta = {"sampling": "uniform_without_replacement", "weight_field": None}

    chosen_ids = [ids[i] for i in chosen_idx]
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for i in chosen_idx:
            handle.seek(offsets[i])
            records.append(json.loads(handle.readline()))
    meta.update({"n_pool": total, "n_sample": n, "seed": seed, "has_positive_weight": has_weight})
    return records, chosen_ids, meta


class BatchProgress:
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


def normalize_yes_no(text: str | None) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    low = s.casefold()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    if "would not purchase" in low or "would not" in low:
        return "No"
    if "would purchase" in low:
        return "Yes"
    return None


def job_dir() -> Path:
    return OUT_ROOT / MODEL / DATASET / METHOD


def job_is_complete(out_dir: Path) -> bool:
    responses = out_dir / "responses.jsonl"
    manifest = out_dir / "manifest.json"
    return responses.is_file() and responses.stat().st_size > 0 and manifest.is_file()


def build_cards(rows: list[dict], prompts_by_qid: dict[str, str]) -> tuple[dict, list]:
    human = json.loads(HUMAN_SHARE_PATH.read_text(encoding="utf-8"))
    by_qid_human = human["by_qid"]

    # model counts
    counts: dict[str, dict[str, int]] = {}
    other: dict[str, int] = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        parsed = row.get("parsed") or {}
        pred = parsed.get("answer")
        key = normalize_yes_no(pred)
        bucket = counts.setdefault(qid, {"Yes": 0, "No": 0})
        if key in ("Yes", "No"):
            bucket[key] += 1
        else:
            other[qid] = other.get(qid, 0) + 1

    cards = []
    canvas = []
    tvds: list[float] = []
    for i in range(1, QUESTIONS_PER_SUBJECT + 1):
        qid = f"QID9_{i}"
        h = by_qid_human[qid]
        # map long labels → Yes/No
        h_share_raw = h.get("share") or {}
        hy = float(h_share_raw.get(YES_LABEL, h_share_raw.get("Yes", 0.0)))
        hn = float(h_share_raw.get(NO_LABEL, h_share_raw.get("No", 0.0)))
        # renormalize just in case
        hs = hy + hn
        if hs > 0:
            hy, hn = hy / hs, hn / hs
        human_mean = hy

        c = counts.get(qid, {"Yes": 0, "No": 0})
        nY, nN = int(c["Yes"]), int(c["No"])
        n = nY + nN
        if n == 0:
            my = mn = 0.0
            tvd = 1.0
            mean = 0.0
        else:
            my, mn = nY / n, nN / n
            tvd = 0.5 * (abs(my - hy) + abs(mn - hn))
            mean = my
        tvds.append(tvd)
        body = prompts_by_qid.get(qid, "")
        card = {
            "idx": i,
            "qid": qid,
            "title": "定价购买",
            "body": body,
            "human_mean": round(human_mean, 6),
            "human_share": {"Yes": round(hy, 6), "No": round(hn, 6)},
            "models": {
                MODEL: {
                    "counts": {"Yes": nY, "No": nN},
                    "n": n,
                    "tvd": round(tvd, 6),
                    "mean": round(mean, 6),
                }
            },
            "opt_labels": [
                {"key": "Yes", "label": YES_LABEL},
                {"key": "No", "label": NO_LABEL},
            ],
        }
        if other.get(qid):
            card["unparsed"] = other[qid]
        cards.append(card)
        canvas.append(
            {
                "i": i,
                "q": qid,
                "t": "定价购买",
                "b": body,
                "hm": round(human_mean, 6),
                "hs": [round(hy * 100, 1), round(hn * 100, 1)],
                "ds": [nY, nN, n, round(tvd, 6), round(mean, 6)],
            }
        )

    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    payload = {
        "dataset": f"ATUS adults → QID9 (n={N_SUBJECTS})",
        "n": N_SUBJECTS,
        "method": "atus_expanded · base · no skill",
        "model": MODEL,
        "human_source": "twin2k_human_share_qid9.json",
        "mean_tvd": round(mean_tvd, 6) if mean_tvd is not None else None,
        "cards": cards,
    }
    return payload, canvas


def run() -> dict:
    out_dir = job_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = OUT_ROOT / "sample_ids.json"
    if sample_path.is_file() and (OUT_ROOT / "sample_records.jsonl").is_file():
        sample_meta = json.loads(sample_path.read_text(encoding="utf-8"))
        chosen_ids = list(sample_meta["ids"])
        records = []
        with (OUT_ROOT / "sample_records.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        print(f"reusing sample n={len(records)}", flush=True)
    else:
        records, chosen_ids, meta = sample_records(SUBJECTS_PATH, n=N_SUBJECTS, seed=SEED)
        write_json(sample_path, {"ids": chosen_ids, **meta})
        with (OUT_ROOT / "sample_records.jsonl").open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"sampled {len(chosen_ids)} subjects ({meta['sampling']})", flush=True)

    bundle = bundle_from_records(records, source=str(SUBJECTS_PATH))
    # sanity
    qcounts: dict[str, int] = {}
    for case in bundle.cases:
        qcounts[case.subject.id] = qcounts.get(case.subject.id, 0) + 1
    bad = {sid: n for sid, n in qcounts.items() if n != QUESTIONS_PER_SUBJECT}
    if bad:
        raise SystemExit(f"expected {QUESTIONS_PER_SUBJECT} QID9 per subject; bad={list(bad.items())[:3]}")

    prompts_by_qid: dict[str, str] = {}
    for case in bundle.cases:
        prompts_by_qid.setdefault(case.question.id, case.question.prompt)

    batches_total = len(bundle.subjects)  # one batch per subject
    if job_is_complete(out_dir):
        print(f"skip existing {out_dir}", flush=True)
        rows = []
        with (out_dir / "responses.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        cards, canvas = build_cards(rows, prompts_by_qid)
        write_json(OUT_ROOT / "qid_cards.json", cards)
        write_json(OUT_ROOT / "canvas_cards.json", canvas)
        summary = {
            "ok": True,
            "skipped": True,
            "n_results": len(rows),
            "n_cases": len(bundle.cases),
            "mean_tvd": cards["mean_tvd"],
            "out": str(out_dir),
        }
        write_progress(status="skipped_complete", **summary)
        return summary

    provider = build_provider(PROVIDER, timeout=REQUEST_TIMEOUT)
    client = LLMClient(provider, model=MODEL, cache_dir=out_dir / "cache", temperature=0)
    state = {"batches_done": 0, "cases_done": 0, "live_seconds": [], "errors": []}

    def on_error(packages, exc: CommsTransportError) -> None:
        state["errors"].append(
            {
                "subject_id": packages[0].subject_id,
                "n_questions": len(packages),
                "error": str(exc)[:500],
            }
        )
        print(f"batch failed subject={packages[0].subject_id} nq={len(packages)} {exc}", flush=True)
        write_progress(
            status="batch_error",
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
            f"batch {state['batches_done']}/{batches_total} "
            f"subject={packages[0].subject_id} nq={len(packages)} "
            f"cached={cached} elapsed={elapsed:.1f}s eta_s={eta}",
            flush=True,
        )
        write_progress(
            status="running",
            batches_done=state["batches_done"],
            batches_total=batches_total,
            batches_failed=len(state["errors"]),
            cases_done=state["cases_done"],
            eta_seconds_if_remaining_uncached=eta,
            last_subject=packages[0].subject_id,
        )

    write_progress(
        status="running",
        batches_done=0,
        batches_total=batches_total,
        n_subjects=len(bundle.subjects),
        n_cases=len(bundle.cases),
    )
    started = time.time()
    progress = BatchProgress(client, on_batch=on_batch, on_error=on_error)
    results = simulate(
        bundle,
        method=METHOD,
        client=progress,
        limit=None,
        seed=SEED,
        batch_size=BATCH_SIZE,
    )
    elapsed = round(time.time() - started, 1)
    if progress.failed or len(results) != len(bundle.cases):
        write_json(out_dir / "batch_errors.json", {"errors": state["errors"]})
        summary = {
            "ok": False,
            "failed_batches": progress.failed,
            "n_results": len(results),
            "n_cases": len(bundle.cases),
            "seconds": elapsed,
            "out": str(out_dir),
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
    operational = score_rows(rows)
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
        "seconds": elapsed,
        "seed": SEED,
        "temperature": 0,
        "one_call_per_subject": True,
        "operational": operational,
    }
    write_json(out_dir / "manifest.json", manifest)
    responses_tmp.replace(out_dir / "responses.jsonl")

    cards, canvas = build_cards(rows, prompts_by_qid)
    write_json(OUT_ROOT / "qid_cards.json", cards)
    write_json(OUT_ROOT / "canvas_cards.json", canvas)

    summary = {
        "ok": True,
        "model": MODEL,
        "dataset": DATASET,
        "method": METHOD,
        "n_rows": len(rows),
        "n_results": len(results),
        "n_cases": len(bundle.cases),
        "seconds": elapsed,
        "mean_tvd": cards["mean_tvd"],
        "qid_cards": str(OUT_ROOT / "qid_cards.json"),
        "canvas_cards": str(OUT_ROOT / "canvas_cards.json"),
        "out": str(out_dir),
    }
    write_json(out_dir / "run_summary.json", summary)
    write_json(OUT_ROOT / "manifest.json", {
        "provider": PROVIDER,
        "method": METHOD,
        "model": MODEL,
        "seed": SEED,
        "n_subjects": N_SUBJECTS,
        "batch_size": BATCH_SIZE,
        "request_timeout": REQUEST_TIMEOUT,
        "git_rev": git_rev(),
        "one_call_per_subject": True,
        "summary": summary,
    })
    write_progress(status="all_done", **summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("CR_API_KEY", "").strip():
        raise SystemExit("CR_API_KEY missing")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_progress(status="loading", git_rev=git_rev())
    try:
        summary = run()
    except Exception as exc:
        traceback.print_exc()
        write_progress(status="error", error=f"{type(exc).__name__}: {exc}")
        return 1
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
