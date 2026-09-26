"""Ask a Nemotron panel the longer closed-ended Twin-2K form.

This is separate from twin-2k-50. Each call still carries the answers already
given, so later questions can stay consistent with earlier ones.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from popbench.dao.interview import InterviewError
from popbench.dao.io import write_jsonl
from popbench.simulate.client import (
    ModelError,
    chat_with_retry,
    model_settings_from_env,
    parse_choice,
    response_cache_key,
)
from popbench.simulate.interview import system_message


def question_message(item: dict[str, Any], n_items: int, previous: list[str]) -> str:
    lines = [f"Question {item['turn']} of {n_items}.", item["prompt"], ""]
    if item["kind"] == "choice":
        lines.append("Options:")
        for index, option in enumerate(item["options"], start=1):
            lines.append(f"{index}. {option}")
        lines.append("")
        lines.append(
            'Reply with JSON only: {"answer": "<one option, exactly as written>", '
            '"rationale": "<one short sentence>"}'
        )
    else:
        lines.append(
            'Reply with JSON only: {"answer": "<a number>", "rationale": "<one short sentence>"}'
        )
    if previous:
        lines.extend(["", "Answers you already gave:"])
        lines.extend(previous)
    return "\n".join(lines)


def answer_survey(records: list[dict[str, Any]], run_dir: Path, workers: int = 4) -> list[dict[str, Any]]:
    """Write one response row per persona. Does not score."""
    try:
        settings = model_settings_from_env()
    except RuntimeError as exc:
        raise InterviewError(str(exc)) from exc
    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    finished = 0
    lock = threading.Lock()

    def _one(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal finished
        result = _answer_record(settings, record, cache_dir)
        with lock:
            finished += 1
            print(f"answered {finished}/{len(records)} {record['id']}", flush=True)
        return result

    workers = max(1, min(workers, len(records)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_one, records))
    write_jsonl(run_dir / "responses.jsonl", responses)
    meta = {
        "model": settings.model,
        "base_url": settings.base_url,
        "panel": "nemotron-full",
        "n_people": len(records),
        "n_turns": len(records[0]["turns"]) if records else 0,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return responses


def _answer_record(settings, record: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    system = system_message(record["persona"]["text"])
    answers = []
    previous: list[str] = []
    n_items = len(record["turns"])
    for turn in record["turns"]:
        user = question_message(turn, n_items, previous)
        cache_path = cache_dir / f"{response_cache_key(settings.model, record['id'], turn['item_id'])}.json"
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text())["raw"]
        else:
            try:
                raw = chat_with_retry(
                    settings,
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
            except ModelError as exc:
                raise InterviewError(str(exc)) from exc
            cache_path.write_text(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
        try:
            if turn["kind"] == "choice":
                choice = parse_choice(raw, tuple(turn["options"]))
                value = turn["options"].index(choice.answer) + 1
                shown = choice.answer
            else:
                value = _parse_number(raw)
                shown = str(value)
            answers.append({"item_id": turn["item_id"], "value": value, "raw": raw, "ok": True})
            if turn["turn"] % 50 == 0:
                print(f"  {record['id']} turn {turn['turn']}/{n_items}", flush=True)
            short = " ".join(turn["prompt"].split())
            if len(short) > 80:
                short = short[:80] + "…"
            previous.append(f"{turn['turn']}. {short} -> {shown}")
        except ValueError as exc:
            answers.append(
                {"item_id": turn["item_id"], "value": None, "raw": raw, "ok": False, "error": str(exc)}
            )
    return {"id": record["id"], "answers": answers}


def _parse_number(raw: str) -> float:
    choice = parse_choice(raw, ())
    match = re.search(r"-?\d+(?:\.\d+)?", choice.answer.replace(",", ""))
    if not match:
        raise ValueError(f"answer is not a number: {choice.answer}")
    return float(match.group(0))
