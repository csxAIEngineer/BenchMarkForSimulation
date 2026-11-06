#!/usr/bin/env python3
"""Income-fixed 50 ATUS persona cards → QID9 via MiniMax-M3.1-Flash.

Uses repaired Personal income (19 ≤50k / 31 over $50k matched to Twin2K
fixed-card order). Does NOT overwrite original user_50_personas.jsonl.
One CR call per subject (batch_size=40).
"""

from __future__ import annotations

import json
import os
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
from popbench.models._prompt import build_package  # noqa: E402
from popbench.schema import Case, DatasetBundle, Question, Subject  # noqa: E402

OUT_ROOT = ROOT / "runs" / "atus-qid9-user50-income-fixed-minimax"
PERSONAS = ROOT / "data" / "atus_qid9" / "user_50_personas_income_fixed.jsonl"
QID_CATALOG = ROOT / "data" / "atus_qid9" / "qid9_question_catalog.json"
HUMAN_SHARE_PATH = ROOT / "data" / "atus_qid9" / "twin2k_human_share_qid9.json"
PROGRESS = OUT_ROOT / "progress.json"

MODEL = "MiniMax-M3.1-Flash"
PROVIDER = "cr"
METHOD = "user_persona_card"  # fixed text from user file
DATASET = "atus_qid9_user50_income_fixed_minimax"
SEED = 42
BATCH_SIZE = 40
REQUEST_TIMEOUT = 600
QUESTIONS_PER_SUBJECT = 40

YES_LABEL = "Yes, I would purchase the product"
NO_LABEL = "No, I would not purchase the product"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_progress(**kwargs: object) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "out_root": str(OUT_ROOT),
        "method": METHOD,
        "model": MODEL,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "note": "income-fixed user persona text (19/31); MiniMax-M3.1-Flash; one CR call per subject (40 QID9)",
        **kwargs,
    }
    write_json(PROGRESS, payload)


def load_questions() -> list[dict[str, Any]]:
    qs = json.loads(QID_CATALOG.read_text(encoding="utf-8"))
    assert len(qs) == 40
    return qs


def load_records() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in PERSONAS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 50:
        raise SystemExit(f"expected 50 personas, got {len(rows)}")
    for r in rows:
        if not str(r.get("text") or "").strip():
            raise SystemExit(f"missing text for {r.get('id')}")
    return rows


def build_bundle(records: list[dict[str, Any]], questions: list[dict[str, Any]]) -> DatasetBundle:
    subjects: list[Subject] = []
    cases: list[Case] = []
    for rec in records:
        sid = str(rec["id"])
        demo = {k: str(v) for k, v in (rec.get("demographics") or {}).items()}
        # parse routine minutes from text if needed — optional; card is fixed
        subject = Subject(
            id=sid,
            dataset=DATASET,
            unit="individual",
            demographics=demo,
            demographic_text=str(rec["text"]),
            persona_text=str(rec["text"]),
            persona_summary=None,
            routine=None,
        )
        subjects.append(subject)
        for q in questions:
            question = Question(
                id=str(q["id"]),
                prompt=str(q["prompt"]),
                options=tuple(q.get("options") or [YES_LABEL, NO_LABEL]),
                kind="multiple_choice",
            )
            cases.append(Case(dataset=DATASET, subject=subject, question=question, gold=None))
    return DatasetBundle(
        name=DATASET,
        unit="individual",
        source=str(PERSONAS),
        revision=None,
        subjects=tuple(subjects),
        cases=tuple(cases),
    )


def packages_for_bundle(bundle: DatasetBundle):
    # persona = subject.persona_text (user card), not atus_expanded
    pkgs = []
    for case in bundle.cases:
        persona = (case.subject.persona_text or case.subject.demographic_text or "").strip()
        pkgs.append(build_package(case, method=METHOD, persona=persona, response_mode="answer"))
    return pkgs


def by_subject(packages: list) -> list[list]:
    order: list[str] = []
    grouped: dict[str, list] = {}
    for p in packages:
        if p.subject_id not in grouped:
            order.append(p.subject_id)
            grouped[p.subject_id] = []
        grouped[p.subject_id].append(p)
    return [grouped[s] for s in order]


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


def build_cards(rows: list[dict], prompts_by_qid: dict[str, str]) -> tuple[dict, list]:
    human = json.loads(HUMAN_SHARE_PATH.read_text(encoding="utf-8"))
    by_qid_human = human["by_qid"]
    counts: dict[str, dict[str, int]] = {}
    other: dict[str, int] = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        parsed = row.get("parsed") or {}
        key = normalize_yes_no(parsed.get("answer"))
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
        h_share_raw = h.get("share") or {}
        hy = float(h_share_raw.get(YES_LABEL, h_share_raw.get("Yes", 0.0)))
        hn = float(h_share_raw.get(NO_LABEL, h_share_raw.get("No", 0.0)))
        hs = hy + hn
        if hs > 0:
            hy, hn = hy / hs, hn / hs
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
        cards.append(
            {
                "idx": i,
                "qid": qid,
                "title": "定价购买",
                "body": body,
                "human_mean": round(hy, 6),
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
                **({"unparsed": other[qid]} if other.get(qid) else {}),
            }
        )
        canvas.append(
            {
                "i": i,
                "q": qid,
                "t": "定价购买",
                "b": body,
                "hm": round(hy, 6),
                "hs": [round(hy * 100, 1), round(hn * 100, 1)],
                "ds": [nY, nN, n, round(tvd, 6), round(mean, 6)],
            }
        )

    mean_tvd = sum(tvds) / len(tvds) if tvds else None
    qid_payload = {
        "dataset": "ATUS user-50 income-fixed persona cards → QID9",
        "n": 50,
        "method": f"{METHOD} · fixed card · no skill",
        "model": MODEL,
        "human_source": "twin2k_human_share_qid9.json",
        "mean_tvd": round(mean_tvd, 6) if mean_tvd is not None else None,
        "cards": cards,
    }
    return qid_payload, canvas


def result_to_row(result) -> dict:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return dict(result)


def main() -> int:
    load_dotenv(ROOT / ".env")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    records = load_records()
    write_json(OUT_ROOT / "sample_ids.json", {"ids": [r["id"] for r in records], "n": len(records)})
    (OUT_ROOT / "sample_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )

    bundle = build_bundle(records, questions)
    packages = packages_for_bundle(bundle)
    groups = by_subject(packages)
    assert len(groups) == 50
    assert all(len(g) == 40 for g in groups)

    out_dir = OUT_ROOT / MODEL / DATASET / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "responses.jsonl").is_file() and (out_dir / "manifest.json").is_file():
        print(f"skip existing complete job {out_dir}", flush=True)
        rows = [json.loads(line) for line in (out_dir / "responses.jsonl").read_text().splitlines() if line.strip()]
        prompts = {q["id"]: q["prompt"] for q in questions}
        qid_payload, canvas = build_cards(rows, prompts)
        write_json(OUT_ROOT / "qid_cards.json", qid_payload)
        write_json(OUT_ROOT / "canvas_cards.json", canvas)
        write_progress(status="complete", mean_tvd=qid_payload["mean_tvd"], n_results=len(rows))
        print("ALL_DONE", qid_payload["mean_tvd"], flush=True)
        return 0

    provider = build_provider(PROVIDER, timeout=REQUEST_TIMEOUT)
    client = LLMClient(provider, model=MODEL, cache_dir=out_dir / "cache", temperature=0)
    state = {"batches_done": 0, "cases_done": 0, "live_seconds": [], "errors": []}
    batches_total = len(groups)

    def on_error(pkgs, exc: CommsTransportError) -> None:
        state["errors"].append(
            {"subject_id": pkgs[0].subject_id, "error": str(exc)[:500]}
        )
        print(f"batch failed subject={pkgs[0].subject_id} {exc}", flush=True)
        write_progress(
            status="batch_error",
            batches_done=state["batches_done"],
            batches_total=batches_total,
            batches_failed=len(state["errors"]),
        )

    def on_batch(pkgs, rows, elapsed: float) -> None:
        state["batches_done"] += 1
        state["cases_done"] += len(rows)
        cached = bool(rows) and all(getattr(r, "cached", False) for r in rows)
        if not cached:
            state["live_seconds"].append(elapsed)
        remaining = batches_total - state["batches_done"]
        eta = None
        if state["live_seconds"] and remaining:
            eta = round((sum(state["live_seconds"]) / len(state["live_seconds"])) * remaining, 1)
        print(
            f"batch {state['batches_done']}/{batches_total} subject={pkgs[0].subject_id} "
            f"cached={cached} eta_s={eta}",
            flush=True,
        )
        write_progress(
            status="running",
            batches_done=state["batches_done"],
            batches_total=batches_total,
            batches_failed=len(state["errors"]),
            cases_done=state["cases_done"],
            eta_seconds_if_remaining_uncached=eta,
            last_subject=pkgs[0].subject_id,
        )

    progress = BatchProgress(client, on_batch=on_batch, on_error=on_error)
    write_progress(status="running", batches_done=0, batches_total=batches_total, n_subjects=50, n_cases=2000)
    started = time.time()
    results = []
    try:
        for group in groups:
            for start in range(0, len(group), BATCH_SIZE):
                chunk = group[start : start + BATCH_SIZE]
                results.extend(progress.complete_batch(chunk, sample_seed=SEED))
    except Exception:
        write_json(OUT_ROOT / "crash.json", {"traceback": traceback.format_exc()})
        write_progress(status="crashed", batches_done=state["batches_done"], batches_total=batches_total)
        raise

    if progress.failed or len(results) != len(bundle.cases):
        summary = {
            "ok": False,
            "failed_batches": progress.failed,
            "n_results": len(results),
            "n_cases": len(bundle.cases),
            "errors": state["errors"],
            "seconds": round(time.time() - started, 1),
        }
        write_json(out_dir / "batch_errors.json", summary)
        write_progress(status="incomplete", **summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1

    rows = [result_to_row(r) for r in results]
    (out_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    write_json(
        out_dir / "manifest.json",
        {
            "dataset": DATASET,
            "method": METHOD,
            "provider": PROVIDER,
            "model": MODEL,
            "batch_size": BATCH_SIZE,
            "n_results": len(rows),
            "n_subjects": 50,
            "n_cases": 2000,
            "seconds": round(time.time() - started, 1),
            "seed": SEED,
            "temperature": 0,
            "persona_source": str(PERSONAS),
        },
    )
    prompts = {q["id"]: q["prompt"] for q in questions}
    qid_payload, canvas = build_cards(rows, prompts)
    write_json(OUT_ROOT / "qid_cards.json", qid_payload)
    write_json(OUT_ROOT / "canvas_cards.json", canvas)
    write_progress(status="complete", mean_tvd=qid_payload["mean_tvd"], n_results=len(rows), seconds=round(time.time() - started, 1))
    print("ALL_DONE mean_tvd=", qid_payload["mean_tvd"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
