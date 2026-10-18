"""Ask a panel the longer closed-ended Twin-2K form.

Questions that share an instrument go out in one call. The persona card stays
the system message. A scale does not see answers from any other scale. Each
item is still cached alone, under the model, person, and item id, so a rerun
skips questions that already have a file.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from popbench.dao.interview import InterviewError
from popbench.dao.io import write_jsonl
from popbench.simulate.client import (
    ModelError,
    ModelSettings,
    chat_with_retry,
    model_settings_from_env,
    parse_choice,
    request_record,
    response_cache_key,
    with_cr_temperature_support,
)
from popbench.simulate.interview import system_message

# Shared stems shorter than this stay on each question.
_PREFIX_MIN = 40
_PARSE_ATTEMPTS = 2


def share_blank_instrument(
    records: list[dict[str, Any]],
    name: str,
) -> list[dict[str, Any]]:
    """Copy records so blank-instrument questions share one name.

    Those questions then leave in one request. A question that already names
    an instrument stays on its own scale. The input records are not mutated.
    """
    label = str(name).strip()
    if not label:
        raise ValueError("batch instrument name is empty")
    stamped: list[dict[str, Any]] = []
    for record in records:
        turns: list[dict[str, Any]] = []
        changed = False
        for turn in record["turns"]:
            if str(turn.get("instrument") or "").strip():
                turns.append(turn)
                continue
            copied = dict(turn)
            copied["instrument"] = label
            turns.append(copied)
            changed = True
        stamped.append({**record, "turns": turns} if changed else record)
    return stamped


def instrument_groups(turns: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Contiguous runs that share a non-empty instrument. Others stay one question."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_name = ""
    for turn in turns:
        name = str(turn.get("instrument") or "").strip()
        if not name:
            if current:
                groups.append(current)
                current = []
                current_name = ""
            groups.append([turn])
            continue
        if current and name == current_name:
            current.append(turn)
            continue
        if current:
            groups.append(current)
        current = [turn]
        current_name = name
    if current:
        groups.append(current)
    return groups


def batch_message(turns: list[dict[str, Any]], n_items: int) -> str:
    """One user message for a scale. Shared options are written once. No rationale."""
    if not turns:
        raise ValueError("batch has no questions")
    instrument = str(turns[0].get("instrument") or "").strip()
    lines = [_span(turns, n_items)]
    if instrument:
        lines[0] = f"{instrument}. {lines[0]}"
    prefix, bodies = _shared_prompt_prefix(turns)
    if prefix:
        lines.extend(["", prefix])
    lines.extend(["", _reply_instruction(turns), ""])
    lines.extend(_render_questions(turns, bodies))
    return "\n".join(lines).rstrip() + "\n"


def answer_survey(
    records: list[dict[str, Any]],
    run_dir: Path,
    workers: int = 4,
    settings: ModelSettings | None = None,
    *,
    panel: str = "nemotron-full",
    seed: int | None = None,
    family: str | None = None,
    resume: bool = False,
    batch: bool = False,
) -> list[dict[str, Any]]:
    """Write one response row per persona. Does not score.

    ``resume`` appends each finished person and skips ids already in
    ``responses.jsonl``. Item cache still fills any person who stopped midway.
    """
    try:
        if settings is None:
            settings = model_settings_from_env()
    except RuntimeError as exc:
        raise InterviewError(str(exc)) from exc
    settings = with_cr_temperature_support(settings)
    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    response_path = run_dir / "responses.jsonl"
    n_people = len(records)
    n_turns = len(records[0]["turns"]) if records else 0
    done_ids: set[str] = set()
    if resume:
        done_ids = _completed_response_ids(response_path)
        records = [record for record in records if record["id"] not in done_ids]
        print(f"resume {len(done_ids)}/{n_people} already saved", flush=True)
    finished = len(done_ids) if resume else 0
    lock = threading.Lock()

    def _one(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal finished
        result = _answer_record(settings, record, cache_dir)
        with lock:
            if resume:
                _append_jsonl(response_path, result)
            finished += 1
            print(f"answered {finished}/{n_people} {record['id']}", flush=True)
        return result

    if records:
        workers = max(1, min(workers, len(records)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            responses = list(pool.map(_one, records))
    else:
        responses = []
    if not resume:
        write_jsonl(response_path, responses)
    meta = {
        "model": settings.model,
        "base_url": settings.base_url,
        "panel": panel,
        "n_people": n_people,
        "n_turns": n_turns,
        "request": request_record(settings),
    }
    if seed is not None:
        meta["seed"] = seed
    if family is not None:
        meta["family"] = family
    if batch:
        meta["batch"] = True
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return responses


def _completed_response_ids(path: Path) -> set[str]:
    """Ids already stored. A trailing partial line is dropped."""
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0]
        text = f"{text}\n" if text else ""
        path.write_text(text, encoding="utf-8")
    ids: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        ids.add(json.loads(line)["id"])
    return ids


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _answer_record(settings: ModelSettings, record: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    system = system_message(record["persona"]["text"])
    n_items = len(record["turns"])
    by_id: dict[str, dict[str, Any]] = {}
    for group in instrument_groups(record["turns"]):
        pending: list[dict[str, Any]] = []
        for turn in group:
            cached = _read_cache(_cache_path(cache_dir, settings.model, record["id"], turn["item_id"]))
            if cached is None:
                pending.append(turn)
                continue
            by_id[turn["item_id"]] = _answer_from_raw(turn, cached)
        if not pending:
            continue
        _fill_pending(settings, system, pending, n_items, cache_dir, record["id"], by_id)
    return {"id": record["id"], "answers": [by_id[turn["item_id"]] for turn in record["turns"]]}


def _fill_pending(
    settings: ModelSettings,
    system: str,
    pending: list[dict[str, Any]],
    n_items: int,
    cache_dir: Path,
    persona_id: str,
    by_id: dict[str, dict[str, Any]],
) -> None:
    """Ask one scale. If the reply is short or unmatched, ask the two halves."""
    if not pending:
        return
    label = str(pending[0].get("instrument") or "").strip() or f"question {pending[0]['turn']}"
    print(f"  {persona_id} {label} asking {len(pending)}", flush=True)
    parsed, raw, error = _request_batch(settings, system, pending, n_items, persona_id, label)
    if parsed is not None:
        for turn, row in zip(pending, parsed, strict=True):
            _write_cache(_cache_path(cache_dir, settings.model, persona_id, turn["item_id"]), row["raw"])
            by_id[turn["item_id"]] = row
        return
    if len(pending) > 1:
        middle = len(pending) // 2
        print(f"  {persona_id} {label} splitting {len(pending)}", flush=True)
        _fill_pending(settings, system, pending[:middle], n_items, cache_dir, persona_id, by_id)
        _fill_pending(settings, system, pending[middle:], n_items, cache_dir, persona_id, by_id)
        return
    by_id[pending[0]["item_id"]] = {
        "item_id": pending[0]["item_id"],
        "value": None,
        "raw": raw,
        "ok": False,
        "error": error or "model response could not be parsed",
    }


def _request_batch(
    settings: ModelSettings,
    system: str,
    pending: list[dict[str, Any]],
    n_items: int,
    persona_id: str,
    label: str,
) -> tuple[list[dict[str, Any]] | None, str, str]:
    user = batch_message(pending, n_items)
    call_settings = _call_settings(settings, pending)
    attempts = _PARSE_ATTEMPTS if len(pending) > 1 else _PARSE_ATTEMPTS + 1
    raw = ""
    last_error = "model response could not be parsed"
    for attempt in range(1, attempts + 1):
        try:
            raw = chat_with_retry(
                call_settings,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except ModelError as exc:
            if "no content" in str(exc):
                return None, raw, str(exc)
            raise InterviewError(str(exc)) from exc
        try:
            return _parse_batch(raw, pending), raw, ""
        except ValueError as exc:
            last_error = str(exc)
            print(
                f"  {persona_id} {label} parse failed ({attempt}/{attempts}): {exc}",
                flush=True,
            )
    return None, raw, last_error


def _span(turns: list[dict[str, Any]], n_items: int) -> str:
    numbers = [int(turn["turn"]) for turn in turns]
    if len(numbers) == 1:
        return f"Question {numbers[0]} of {n_items}."
    contiguous = numbers == list(range(numbers[0], numbers[-1] + 1))
    if contiguous:
        return f"Questions {numbers[0]}-{numbers[-1]} of {n_items}."
    listed = ", ".join(str(number) for number in numbers)
    return f"Questions {listed} of {n_items}."


def _reply_instruction(turns: list[dict[str, Any]]) -> str:
    kinds = {turn.get("kind") for turn in turns}
    if kinds == {"numeric"}:
        slot = "<a number>"
    elif kinds == {"choice"}:
        slot = "<one option, exactly as written>"
    else:
        slot = "<one option exactly as written, or a number>"
    return (
        f'Reply with JSON only: {{"answers":["{slot}", ...]}} '
        f"with exactly {len(turns)} strings, one per question, in the order below."
    )


def _render_questions(turns: list[dict[str, Any]], bodies: list[str]) -> list[str]:
    lines: list[str] = []
    runs = _option_runs(turns, bodies)
    if len(runs) == 1 and runs[0][0] is not None:
        lines.extend(_option_lines(runs[0][0]))
        lines.append("")
        for turn, body in runs[0][1]:
            lines.append(f"{turn['turn']}. {body}")
        return lines
    for options, pairs in runs:
        if options is not None and len(pairs) > 1:
            lines.append("These questions use the same options:")
            lines.extend(_option_lines(options))
            lines.append("")
            for turn, body in pairs:
                lines.append(f"{turn['turn']}. {body}")
            lines.append("")
            continue
        for turn, body in pairs:
            lines.append(f"{turn['turn']}. {body}")
            if options is not None:
                lines.extend(_option_lines(options))
            lines.append("")
    return lines


def _option_runs(
    turns: list[dict[str, Any]],
    bodies: list[str],
) -> list[tuple[tuple[str, ...] | None, list[tuple[dict[str, Any], str]]]]:
    runs: list[tuple[tuple[str, ...] | None, list[tuple[dict[str, Any], str]]]] = []
    for turn, body in zip(turns, bodies, strict=True):
        options = tuple(str(option) for option in turn["options"]) if turn.get("kind") == "choice" else None
        if runs and runs[-1][0] == options:
            runs[-1][1].append((turn, body))
        else:
            runs.append((options, [(turn, body)]))
    return runs


def _option_lines(options: tuple[str, ...]) -> list[str]:
    lines = ["Options:"]
    for index, option in enumerate(options, start=1):
        lines.append(f"{index}. {option}")
    return lines


def _shared_prompt_prefix(turns: list[dict[str, Any]]) -> tuple[str, list[str]]:
    prompts = [str(turn["prompt"]) for turn in turns]
    if len(prompts) < 2:
        return "", prompts
    prefix = prompts[0]
    for prompt in prompts[1:]:
        limit = min(len(prefix), len(prompt))
        index = 0
        while index < limit and prefix[index] == prompt[index]:
            index += 1
        prefix = prefix[:index]
        if len(prefix.strip()) < _PREFIX_MIN:
            return "", prompts
    if prefix and not prefix[-1].isspace():
        cut = max(prefix.rfind(" "), prefix.rfind("\n"))
        if cut < _PREFIX_MIN:
            return "", prompts
        prefix = prefix[: cut + 1]
    if len(prefix.strip()) < _PREFIX_MIN:
        return "", prompts
    suffixes = [prompt[len(prefix) :].strip() for prompt in prompts]
    if any(not suffix for suffix in suffixes):
        return "", prompts
    return prefix.strip(), suffixes


def _call_settings(settings: ModelSettings, turns: list[dict[str, Any]]) -> ModelSettings:
    """Cap a short non-thinking reply. Thinking models on CR keep their own budget.

    GLM and DeepSeek reason before the JSON answer. A cap sized for the answer
    alone cuts the reply off, and CR rejects a thinking budget under 2000 tokens.
    """
    from cr_api.client import is_thinking_model

    if settings.transport == "cr" and is_thinking_model(settings.model):
        return settings
    cap = _output_token_cap(turns)
    if settings.max_tokens is not None:
        cap = min(settings.max_tokens, cap)
    return replace(settings, max_tokens=cap)


def _output_token_cap(turns: list[dict[str, Any]]) -> int:
    """Enough room for one option string per question, not a long essay."""
    chars = 64
    for turn in turns:
        options = turn.get("options") or []
        if turn.get("kind") == "choice" and options:
            chars += max(len(str(option)) for option in options) + 16
        else:
            chars += 32
    # A short cap stops a runaway reply. Slack covers JSON quotes and the longest option.
    tokens = chars // 3 + 32 * len(turns) + 64
    return min(4096, max(256, tokens))


def _parse_batch(raw: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answers = _answer_strings(_json_payload(raw), turns)
    rows = []
    for turn, answer in zip(turns, answers, strict=True):
        rows.append(_answer_from_text(turn, answer))
    return rows


def _answer_from_raw(turn: dict[str, Any], raw: str) -> dict[str, Any]:
    """Read one cached model string. Old single-answer files still count."""
    try:
        if turn.get("kind") == "choice":
            choice = parse_choice(raw, tuple(turn["options"]))
            value = list(turn["options"]).index(choice.answer) + 1
        else:
            value = _parse_number(raw)
    except ValueError as exc:
        return {
            "item_id": turn["item_id"],
            "value": None,
            "raw": raw,
            "ok": False,
            "error": str(exc),
        }
    return {"item_id": turn["item_id"], "value": value, "raw": raw, "ok": True}


def _answer_from_text(turn: dict[str, Any], answer: str) -> dict[str, Any]:
    if turn.get("kind") == "choice":
        matched = _match_choice(answer, list(turn["options"]))
        value: float = list(turn["options"]).index(matched) + 1
        stored = json.dumps({"answer": matched}, ensure_ascii=False)
    else:
        stored = json.dumps({"answer": answer.strip()}, ensure_ascii=False)
        value = _parse_number(stored)
        stored = json.dumps({"answer": _number_text(value)}, ensure_ascii=False)
    return {"item_id": turn["item_id"], "value": value, "raw": stored, "ok": True}


def _answer_strings(payload: Any, turns: list[dict[str, Any]]) -> list[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and "answers" in payload:
        values = payload["answers"]
    elif isinstance(payload, dict) and "answer" in payload and len(turns) == 1:
        values = [payload["answer"]]
    else:
        raise ValueError("model response has no answers")
    if not isinstance(values, list):
        raise ValueError("answers is not a list")
    pairs: list[tuple[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            if "answer" not in item:
                raise ValueError("answer object has no answer")
            item_id = item.get("item") or item.get("item_id") or item.get("id") or ""
            pairs.append((str(item_id), str(item["answer"]).strip()))
        else:
            pairs.append(("", str(item).strip()))
    ids = [str(turn["item_id"]) for turn in turns]
    keys = [key for key, _value in pairs]
    if keys and all(keys) and len(keys) == len(set(keys)) and set(keys) == set(ids):
        by_id = {key: value for key, value in pairs}
        return [by_id[item_id] for item_id in ids]
    if len(pairs) != len(turns):
        raise ValueError(f"expected {len(turns)} answers, got {len(pairs)}")
    return [value for _key, value in pairs]


def _json_payload(raw: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        array_at = text.find("[")
        object_at = text.find("{")
        if array_at >= 0 and (object_at < 0 or array_at < object_at):
            end = text.rfind("]")
            if end > array_at:
                return json.loads(text[array_at : end + 1])
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response is not JSON") from None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not JSON") from exc


_INDEX_PREFIX = re.compile(r"^\s*\d+\s*[\.\)\-:]\s*")


def _match_choice(answer: str, options: list[str]) -> str:
    """Match one reply to an option.

    An answer that already names an option is kept. Age bands such as
    ``30-49`` look like a numbered prefix, so stripping first would leave
    ``49``. Otherwise drop a leading prompt number, as in
    ``2. Disagree a little``. That string also contains ``Agree a little``,
    so a bare substring match is ambiguous.
    """
    try:
        return _match_written_option(answer, options)
    except ValueError:
        cleaned = _strip_option_index(answer)
    try:
        return _match_written_option(cleaned, options)
    except ValueError:
        longest = _longest_contained(cleaned, options)
        if longest is not None:
            return longest
        closest = _closest_option(cleaned, options)
        if closest is None:
            raise ValueError(f"answer is not one of the options: {answer}") from None
        return closest


def _strip_option_index(answer: str) -> str:
    text = answer.strip()
    stripped = _INDEX_PREFIX.sub("", text, count=1).strip()
    return stripped or text


def _match_written_option(answer: str, options: list[str]) -> str:
    try:
        return parse_choice(json.dumps({"answer": answer}, ensure_ascii=False), tuple(options)).answer
    except ValueError:
        plain_options = [_plain(option) for option in options]
        matched = parse_choice(
            json.dumps({"answer": _plain(answer)}, ensure_ascii=False),
            tuple(plain_options),
        ).answer
    for original, plain in zip(options, plain_options, strict=True):
        if plain == matched:
            return original
    raise ValueError(f"answer is not one of the options: {answer}")


def _longest_contained(answer: str, options: list[str]) -> str | None:
    folded = _plain(answer).casefold()
    contained = [option for option in options if _plain(option).casefold() in folded]
    if not contained:
        return None
    best_len = max(len(_plain(option)) for option in contained)
    best = [option for option in contained if len(_plain(option)) == best_len]
    if len(best) != 1:
        return None
    return best[0]


def _closest_option(answer: str, options: list[str]) -> str | None:
    """Accept a one-character catalog typo, such as ``Agree strongy``."""
    folded = _plain(answer).casefold()
    if len(folded) < 8:
        return None
    close = [
        option
        for option in options
        if len(_plain(option)) >= 8 and _edit_distance(folded, _plain(option).casefold()) <= 1
    ]
    if len(close) != 1:
        return None
    return close[0]


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[-1] + 1,
                    previous[j - 1] + (char != other),
                )
            )
        previous = current
    return previous[-1]


def _plain(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


def _parse_number(raw: str) -> float:
    choice = parse_choice(raw, ())
    match = re.search(r"-?\d+(?:\.\d+)?", choice.answer.replace(",", ""))
    if not match:
        raise ValueError(f"answer is not a number: {choice.answer}")
    return float(match.group(0))


def _number_text(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _cache_path(cache_dir: Path, model: str, persona_id: str, item_id: str) -> Path:
    return cache_dir / f"{response_cache_key(model, persona_id, item_id)}.json"


def _read_cache(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("raw")
    if not isinstance(raw, str):
        return None
    return raw


def _write_cache(path: Path, raw: str) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
    temporary.replace(path)
