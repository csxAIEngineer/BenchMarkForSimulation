"""Run pricing questions for sampled personas through the CR API.

Outputs are local files under the configured directory. This module does not
write reports and does not read API keys itself; ``CRClient`` does.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from popbench.simulate.prompt import build_messages, question_prompt

try:
    from cr_api import CRClient, CRError
    from cr_api.client import TEXT_MODELS
except ImportError as exc:  # pragma: no cover - package is part of this repo
    raise ImportError("cr_api is required for the simulation module") from exc


def known_model_ids() -> set[str]:
    return {model for family in TEXT_MODELS.values() for model in family}


def run_simulation(
    root: Path,
    config: Mapping[str, object],
    *,
    client: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Call CR once per sampled person and write ``responses.jsonl``."""
    model = str(config["model"])
    if model not in known_model_ids():
        raise ValueError(f"model {model!r} is not in cr_api.TEXT_MODELS")
    temperature = None if config.get("temperature") is None else float(config["temperature"])
    reasoning_effort = _reasoning_effort(config.get("reasoning_effort"))
    timeout = float(config.get("timeout") or 300)
    attempts = int(config.get("attempts") or 3)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    datasets = _string_list(config.get("datasets"), "datasets")
    cards_dir = _path(root, config["cards_dir"])
    outcomes_dir = _path(root, config["outcomes_dir"])
    output_dir = _path(root, config["output_dir"])
    limit = config.get("limit_subjects")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit_subjects must be a positive integer or null")
    use_sample = bool(config.get("use_sample", True))
    if client is None:
        client = CRClient()

    written: list[str] = []
    for dataset in datasets:
        cards = _index_cards(cards_dir / dataset / "cards.jsonl")
        outcomes = _index_outcomes(outcomes_dir / dataset / "outcomes.jsonl")
        order = _subject_order(cards_dir / dataset / "sample.json", cards, use_sample=use_sample)
        if limit is not None:
            order = order[:limit]
        rows: list[dict[str, object]] = []
        storage = _storage_name(model, reasoning_effort)
        cache_dir = output_dir / "cache" / storage / dataset
        for subject_id in order:
            if subject_id not in outcomes:
                raise ValueError(f"{dataset} subject {subject_id} has no local outcomes")
            items = outcomes[subject_id]
            questions = [(str(item["question_id"]), question_prompt(item)) for item in items]
            messages = build_messages(str(cards[subject_id]["text"]), questions)
            raw, from_cache = _cached_or_none(
                cache_dir,
                messages,
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            if raw is None:
                raw = _complete(
                    client,
                    messages,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    attempts=attempts,
                    sleep=sleep,
                    reasoning_effort=reasoning_effort,
                )
                from_cache = False
            subject_rows = _rows(
                dataset=dataset,
                model=model,
                subject_id=subject_id,
                question_ids=[question_id for question_id, _prompt in questions],
                raw=raw,
            )
            if not from_cache and all(row["parse_error"] is None for row in subject_rows):
                _write_cache(
                    cache_dir,
                    messages,
                    model=model,
                    temperature=temperature,
                    content=raw,
                    reasoning_effort=reasoning_effort,
                )
            rows.extend(subject_rows)
        target = output_dir / storage / dataset / "responses.jsonl"
        _write_jsonl(target, rows)
        manifest = {
            "dataset": dataset,
            "model": model,
            "n_questions": len(rows),
            "n_subjects": len(order),
            "provider": "cr",
            "reasoning_effort": "default-omitted" if reasoning_effort is None else reasoning_effort,
            "responses": "responses.jsonl",
            "temperature": temperature,
        }
        _write_json(target.parent / "manifest.json", manifest)
        written.append(str(target))
    return {"model": model, "provider": "cr", "responses": written}


def _reasoning_effort(value: object) -> str | None:
    """None means the request omits the field. Any other value is sent as written."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("reasoning_effort must be a non-empty string or null")


def _storage_name(model: str, reasoning_effort: str | None) -> str:
    """Keep omitted-effort runs on the historical model directory.

    An explicit effort gets its own directory so it cannot read or replace
    ``outputs/simulate/<model>/`` or ``outputs/simulate/cache/<model>/``.
    """
    if reasoning_effort is None:
        return model
    slug = reasoning_effort.strip().lower()
    if not slug or any(character not in "abcdefghijklmnopqrstuvwxyz0-9._-" for character in slug):
        raise ValueError(f"reasoning_effort {reasoning_effort!r} cannot be used as a directory name")
    return f"{model}__reasoning-{slug}"


def _cache_key(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None = None,
) -> str:
    payload: dict[str, object] = {"messages": messages, "model": model, "temperature": temperature}
    # Omitted effort stays out of the hash so existing cache files still match.
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _cached_or_none(
    cache_dir: Path,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None = None,
) -> tuple[str | None, bool]:
    path = cache_dir / f"{_cache_key(messages, model=model, temperature=temperature, reasoning_effort=reasoning_effort)}.json"
    if not path.is_file():
        return None, False
    cached = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(cached, dict) and isinstance(cached.get("content"), str):
        return cached["content"], True
    return None, False


def _write_cache(
    cache_dir: Path,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    content: str,
    reasoning_effort: str | None = None,
) -> None:
    path = cache_dir / f"{_cache_key(messages, model=model, temperature=temperature, reasoning_effort=reasoning_effort)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": content}, ensure_ascii=False) + "\n", encoding="utf-8")


def _complete(
    client: Any,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    timeout: float,
    attempts: int,
    sleep: Callable[[float], None],
    reasoning_effort: str | None = None,
) -> str:
    last: Exception | None = None
    for index in range(attempts):
        try:
            kwargs: dict[str, object] = {"model": model, "temperature": temperature, "timeout": timeout}
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            result = client.chat(messages, **kwargs)
            return str(result.content or "")
        except CRError as exc:
            last = exc
            if "网络错误" not in str(exc) or index + 1 == attempts:
                raise
            sleep(0.2 * (2**index))
    assert last is not None
    raise last


def _rows(
    *,
    dataset: str,
    model: str,
    subject_id: str,
    question_ids: Sequence[str],
    raw: str,
) -> list[dict[str, object]]:
    parsed, error = _parse_object(raw)
    by_id: dict[str, str] = {}
    if error is None and parsed is not None:
        answers = parsed.get("answers")
        if not isinstance(answers, list):
            error = "model reply has no answers list"
        else:
            for item in answers:
                if not isinstance(item, dict) or "question_id" not in item:
                    continue
                by_id[str(item["question_id"])] = "" if item.get("answer") is None else str(item["answer"]).strip()
    rows = []
    for question_id in question_ids:
        answer = by_id.get(question_id)
        row_error = error
        if row_error is None and not answer:
            row_error = f"missing answer for {question_id}"
        rows.append(
            {
                "answer": answer,
                "dataset": dataset,
                "model": model,
                "parse_error": row_error,
                "question_id": question_id,
                "subject_id": subject_id,
            }
        )
    return rows


def _parse_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "model reply does not contain a JSON object"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, "model reply contains malformed JSON"
    if not isinstance(payload, dict):
        return None, "model reply JSON must be an object"
    return payload, None


def _subject_order(sample_path: Path, cards: Mapping[str, Mapping[str, object]], *, use_sample: bool) -> list[str]:
    if not use_sample:
        return list(cards)
    if not sample_path.is_file():
        raise ValueError(f"sample file is missing: {sample_path}. Run scripts/build_prepro.py first.")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    ids = sample.get("subject_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{sample_path} has no subject_ids")
    missing = [str(subject_id) for subject_id in ids if str(subject_id) not in cards]
    if missing:
        raise ValueError(f"sample ids missing from cards: {missing[:5]}")
    return [str(subject_id) for subject_id in ids]


def _index_cards(path: Path) -> dict[str, Mapping[str, object]]:
    rows = _read_jsonl(path, kind="cards")
    found = {}
    for row in rows:
        found[str(row["subject_id"])] = row
    return found


def _index_outcomes(path: Path) -> dict[str, list[Mapping[str, object]]]:
    rows = _read_jsonl(path, kind="outcomes")
    found: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        items = row.get("items")
        if not isinstance(items, list):
            raise ValueError(f"outcomes row {row.get('subject_id')} has no items")
        found[str(row["subject_id"])] = items
    return found


def _read_jsonl(path: Path, *, kind: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"local {kind} file is missing: {path}. Run scripts/build_prepro.py first.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append(payload)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [str(item) for item in value]


def _path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path
