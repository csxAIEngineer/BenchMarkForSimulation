"""Multi-turn answers to the twin-2k-50 interview.

Each person is one chat. The model sees the persona card, then answers one
question at a time and keeps its earlier answers in the thread. Gold answers
are not part of the prompt.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from popbench.dao.interview import InterviewError, load_panel, panel_path
from popbench.dao.io import write_jsonl
from popbench.dao.schema import TWIN2K50_TURNS, InterviewRecord, InterviewTurn
from popbench.simulate.client import (
    ModelError,
    ModelSettings,
    chat_with_retry,
    model_settings_from_env,
    parse_choice,
    request_record,
    response_cache_key,
)


def system_message(persona_text: str) -> str:
    return (
        "You are taking a survey as the person described below. "
        "Answer as that person would, and stay consistent across questions. "
        "Choose exactly one of the listed options.\n\n"
        f"{persona_text}"
    )


def turns_user_message(
    turn: InterviewTurn | dict[str, Any],
    *,
    n_turns: int | None = None,
) -> str:
    """One interview turn. Gold answers are not included."""
    if isinstance(turn, InterviewTurn):
        number = turn.turn
        prompt = turn.prompt
        options = turn.options
    else:
        number = turn["turn"]
        prompt = turn["prompt"]
        options = turn["options"]
    total = TWIN2K50_TURNS if n_turns is None else n_turns
    lines = [
        f"Question {number} of {total}.",
        prompt,
        "",
        "Options:",
    ]
    for index, option in enumerate(options, start=1):
        lines.append(f"{index}. {option}")
    lines.extend(
        [
            "",
            'Reply with JSON only: {"answer": "<one option, exactly as written>", '
            '"rationale": "<one short sentence>"}',
        ]
    )
    return "\n".join(lines)


def run_interview(
    data_dir: Path,
    run_dir: Path,
    *,
    panel: str = "nemotron",
    workers: int = 4,
    settings: ModelSettings | None = None,
) -> Path:
    """Answer twin-2k-50 as each person in the named panel. Does not score."""
    try:
        if settings is None:
            settings = model_settings_from_env()
    except RuntimeError as exc:
        raise InterviewError(str(exc)) from exc
    path = panel_path(data_dir, panel)
    if not path.is_file():
        raise InterviewError(f"interview panel is missing: {path}. Run `popbench build`.")
    records = load_panel(path)
    if not records:
        raise InterviewError(f"interview panel is empty: {path}")
    if len(records[0].turns) != TWIN2K50_TURNS:
        raise InterviewError(
            f"{path.name} has {len(records[0].turns)} turns, expected {TWIN2K50_TURNS}"
        )

    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    finished = 0
    lock = threading.Lock()

    def _one(record: InterviewRecord) -> dict[str, Any]:
        nonlocal finished
        result = _answer_record(settings, record, cache_dir)
        with lock:
            finished += 1
            print(f"answered {finished}/{len(records)} {record.id}", flush=True)
        return result

    workers = max(1, min(workers, len(records)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_one, records))
    write_jsonl(run_dir / "responses.jsonl", responses)
    meta = {
        "model": settings.model,
        "base_url": settings.base_url,
        "panel": panel,
        "n_people": len(records),
        "n_turns": TWIN2K50_TURNS,
        "instrument": records[0].instrument,
        "data_dir": str(Path(data_dir)),
        "request": request_record(settings),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run_dir


def run_records(
    records: list[InterviewRecord],
    run_dir: Path,
    *,
    workers: int = 4,
    settings: ModelSettings | None = None,
    panel: str = "atus",
    seed: int | None = None,
) -> Path:
    """Answer whatever turns are on these records. Gold stays out of the prompt."""
    try:
        if settings is None:
            settings = model_settings_from_env()
    except RuntimeError as exc:
        raise InterviewError(str(exc)) from exc
    if not records:
        raise InterviewError("interview panel is empty")
    n_turns = len(records[0].turns)
    if any(len(record.turns) != n_turns for record in records):
        raise InterviewError("every person must answer the same number of turns")

    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    finished = 0
    lock = threading.Lock()

    def _one(record: InterviewRecord) -> dict[str, Any]:
        nonlocal finished
        result = _answer_record(settings, record, cache_dir)
        with lock:
            finished += 1
            print(f"answered {finished}/{len(records)} {record.id}", flush=True)
        return result

    workers = max(1, min(workers, len(records)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_one, records))
    write_jsonl(run_dir / "responses.jsonl", responses)
    meta = {
        "model": settings.model,
        "base_url": settings.base_url,
        "panel": panel,
        "n_people": len(records),
        "n_turns": n_turns,
        "seed": seed,
        "instrument": records[0].instrument,
        "request": request_record(settings),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run_dir


def _answer_record(settings, record: InterviewRecord, cache_dir: Path) -> dict[str, Any]:
    messages = [{"role": "system", "content": system_message(record.persona.text)}]
    answers = []
    n_turns = len(record.turns)
    for turn in record.turns:
        user = turns_user_message(turn, n_turns=n_turns)
        cache_path = cache_dir / f"{response_cache_key(settings.model, record.id, turn.item_id)}.json"
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text())["raw"]
        else:
            try:
                raw = chat_with_retry(settings, messages + [{"role": "user", "content": user}])
            except ModelError as exc:
                raise InterviewError(str(exc)) from exc
            cache_path.write_text(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
        try:
            choice = parse_choice(raw, turn.options)
            answers.append(
                {
                    "item_id": turn.item_id,
                    "answer": choice.answer,
                    "code": turn.options.index(choice.answer) + 1,
                    "rationale": choice.rationale,
                    "raw": raw,
                    "ok": True,
                }
            )
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": raw})
        except ValueError as exc:
            answers.append(
                {
                    "item_id": turn.item_id,
                    "answer": None,
                    "code": None,
                    "raw": raw,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return {"id": record.id, "answers": answers}
