"""Run pricing questions for sampled personas through the CR API.

Outputs are local files under the configured directory. This module does not
write reports and does not read API keys itself; ``CRClient`` does.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from popbench.evaluate.options import NO_OPTION, YES_OPTION, code_purchase, match_option
from popbench.evaluate.stimulus import parse_stimulus
from popbench.simulate.prompt import build_messages, question_prompt
from popbench.simulate.need_inject import annotate_need_inject, inject_need, need_state
from popbench.simulate.three_option import (
    THREE_OPTION_SYSTEM,
    annotate_three_option,
    perm_for,
    three_option_prompt,
)

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
    single_question = _single_question(config.get("questions_per_request"))
    arm = _explicit_arm(config.get("arm"), reasoning_effort, single_question)
    three_option = arm == "three-option"
    need_inject = arm == "need-inject"
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
        if three_option:
            storage = f"{model}__three-option"
        elif need_inject:
            storage = f"{model}__need-inject"
        else:
            storage = _storage_name(model, reasoning_effort, single_question=single_question)
        cache_dir = output_dir / "cache" / storage / dataset
        product_max = _product_maxima(outcomes, order) if three_option else {}
        triplet_index = _triplet_index(root, config) if need_inject else {}
        request_rows: list[dict[str, object]] = []
        for subject_id in order:
            if subject_id not in outcomes:
                raise ValueError(f"{dataset} subject {subject_id} has no local outcomes")
            items = outcomes[subject_id]
            questions = [(str(item["question_id"]), question_prompt(item), item) for item in items]
            if three_option:
                rows.extend(
                    _run_three_option_batch(
                        client,
                        cache_dir,
                        card_text=str(cards[subject_id]["text"]),
                        questions=questions,
                        dataset=dataset,
                        model=model,
                        subject_id=subject_id,
                        temperature=temperature,
                        timeout=timeout,
                        attempts=attempts,
                        sleep=sleep,
                        product_max=product_max,
                    )
                )
                continue
            if need_inject:
                rows.extend(
                    _run_need_inject_batch(
                        client,
                        cache_dir,
                        card_text=str(cards[subject_id]["text"]),
                        questions=questions,
                        dataset=dataset,
                        model=model,
                        subject_id=subject_id,
                        temperature=temperature,
                        timeout=timeout,
                        attempts=attempts,
                        sleep=sleep,
                        triplets=triplet_index,
                    )
                )
                continue
            if single_question:
                subject_rows, subject_requests = _run_single_questions(
                    client,
                    cache_dir,
                    card_text=str(cards[subject_id]["text"]),
                    questions=questions,
                    dataset=dataset,
                    model=model,
                    subject_id=subject_id,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    attempts=attempts,
                    sleep=sleep,
                )
                rows.extend(subject_rows)
                request_rows.extend(subject_requests)
                continue
            messages = build_messages(
                str(cards[subject_id]["text"]),
                [(question_id, prompt) for question_id, prompt, _item in questions],
            )
            raw, from_cache = _cached_or_none(
                cache_dir,
                messages,
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            if raw is None:
                raw, _reasoning = _complete(
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
                question_ids=[question_id for question_id, _prompt, _item in questions],
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
            "questions_per_request": 1 if single_question else "all",
            "reasoning_effort": "default-omitted" if reasoning_effort is None else reasoning_effort,
            "responses": "responses.jsonl",
            "temperature": temperature,
        }
        if arm is not None:
            manifest["arm"] = arm
        _write_json(target.parent / "manifest.json", manifest)
        if single_question:
            _write_jsonl(target.parent / "requests.jsonl", request_rows)
        written.append(str(target))
    return {"model": model, "provider": "cr", "responses": written}


def _reasoning_effort(value: object) -> str | None:
    """None means the request omits the field. Any other value is sent as written."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("reasoning_effort must be a non-empty string or null")


def _explicit_arm(value: object, reasoning_effort: str | None, single_question: bool) -> str | None:
    """Named batched arms copy the default sampling settings and use their own directory."""
    if value is None:
        return None
    if value not in {"three-option", "need-inject"}:
        raise ValueError("arm must be three-option, need-inject, or omitted")
    if reasoning_effort is not None:
        raise ValueError(f"{value} copies the default run and must omit reasoning_effort")
    if single_question:
        raise ValueError(f"{value} asks all 40 questions in one request")
    return str(value)


def _triplet_index(root: Path, config: Mapping[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    """Load the locked (pid, question) price assignment. Do not rebuild it."""
    from popbench.evaluate.arm_a import content_sha256, read_triplets

    path = _path(root, config.get("triplets") or "outputs/evaluate/triplets.parquet")
    rows = read_triplets(path)
    expected = config.get("triplet_sha256")
    if expected is not None and content_sha256(rows) != str(expected):
        raise ValueError("locked triplets do not match triplet_sha256")
    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        index[(str(row["pid"]), str(row["question_id"]))] = row
    return index


def _run_need_inject_batch(
    client: Any,
    cache_dir: Path,
    *,
    card_text: str,
    questions: Sequence[tuple[str, str, Mapping[str, object]]],
    dataset: str,
    model: str,
    subject_id: str,
    temperature: float | None,
    timeout: float,
    attempts: int,
    sleep: Callable[[float], None],
    triplets: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    """One persona plus every pricing question, with a need sentence in each stem."""
    prompts = []
    items_by_question: dict[str, Mapping[str, object]] = {}
    for question_id, stem, item in questions:
        locked = triplets.get((subject_id, question_id))
        if locked is None:
            raise ValueError(f"{subject_id} {question_id} is not in the locked triplets")
        state = need_state(subject_id, str(locked["product_id"]))
        prompts.append((question_id, inject_need(stem, state)))
        items_by_question[question_id] = item
    messages = build_messages(card_text, prompts)
    raw, from_cache = _cached_or_none(
        cache_dir,
        messages,
        model=model,
        temperature=temperature,
        reasoning_effort=None,
    )
    if raw is None:
        raw, _reasoning = _complete(
            client,
            messages,
            model=model,
            temperature=temperature,
            timeout=timeout,
            attempts=attempts,
            sleep=sleep,
            reasoning_effort=None,
        )
        from_cache = False
    subject_rows = _rows(
        dataset=dataset,
        model=model,
        subject_id=subject_id,
        question_ids=[question_id for question_id, _stem, _item in questions],
        raw=raw,
    )
    if not from_cache and all(row["parse_error"] is None for row in subject_rows):
        _write_cache(
            cache_dir,
            messages,
            model=model,
            temperature=temperature,
            content=raw,
            reasoning_effort=None,
        )
    return annotate_need_inject(subject_rows, items_by_question, triplets, full_raw=raw)


def _product_maxima(
    outcomes: Mapping[str, list[dict[str, object]]],
    order: Sequence[str],
) -> dict[str, Decimal]:
    prices: dict[str, list[Decimal]] = defaultdict(list)
    for subject_id in order:
        for item in outcomes[subject_id]:
            stimulus = parse_stimulus(str(item.get("question_text") or ""))
            if stimulus is None:
                continue
            price = stimulus["price_usd"]
            if isinstance(price, Decimal) and price > 0:
                prices[str(stimulus["product"])].append(price)
    return {product: max(values) for product, values in prices.items()}


def _run_three_option_batch(
    client: Any,
    cache_dir: Path,
    *,
    card_text: str,
    questions: Sequence[tuple[str, str, Mapping[str, object]]],
    dataset: str,
    model: str,
    subject_id: str,
    temperature: float | None,
    timeout: float,
    attempts: int,
    sleep: Callable[[float], None],
    product_max: Mapping[str, Decimal],
) -> list[dict[str, object]]:
    """One persona plus every pricing question. Cache lives under ``__three-option``."""
    prompts = []
    items_by_question: dict[str, Mapping[str, object]] = {}
    for question_id, stem, item in questions:
        stimulus = parse_stimulus(stem)
        if stimulus is None:
            raise ValueError(f"{question_id} is not a locked pricing stimulus")
        order = perm_for(subject_id, str(stimulus["product"]))
        prompts.append((question_id, three_option_prompt(stem, order)))
        items_by_question[question_id] = item
    messages = build_messages(card_text, prompts, system=THREE_OPTION_SYSTEM)
    raw, from_cache = _cached_or_none(
        cache_dir,
        messages,
        model=model,
        temperature=temperature,
        reasoning_effort=None,
    )
    if raw is None:
        raw, _reasoning = _complete(
            client,
            messages,
            model=model,
            temperature=temperature,
            timeout=timeout,
            attempts=attempts,
            sleep=sleep,
            reasoning_effort=None,
        )
        from_cache = False
    subject_rows = _rows(
        dataset=dataset,
        model=model,
        subject_id=subject_id,
        question_ids=[question_id for question_id, _stem, _item in questions],
        raw=raw,
    )
    if not from_cache and all(row["parse_error"] is None for row in subject_rows):
        _write_cache(
            cache_dir,
            messages,
            model=model,
            temperature=temperature,
            content=raw,
            reasoning_effort=None,
        )
    return annotate_three_option(subject_rows, items_by_question, product_max, full_raw=raw)


def _single_question(value: object) -> bool:
    """False keeps the historical one-call-per-person batch. True asks each question alone."""
    if value is None or value == "all":
        return False
    if value == 1 or value == "1":
        return True
    raise ValueError("questions_per_request must be 1 or all")


def _storage_name(model: str, reasoning_effort: str | None, *, single_question: bool = False) -> str:
    """Keep omitted-effort batched runs on the historical model directory.

    An explicit effort, or one-question requests, get their own directory so
    they cannot read or replace ``outputs/simulate/<model>/`` or
    ``outputs/simulate/cache/<model>/``. ``none`` plus one question per
    request is ``<model>__reasoning-none__single-q``.
    """
    if reasoning_effort is None and not single_question:
        return model
    parts = [model]
    if reasoning_effort is not None:
        slug = reasoning_effort.strip().lower()
        if not slug or any(character not in "abcdefghijklmnopqrstuvwxyz0-9._-" for character in slug):
            raise ValueError(f"reasoning_effort {reasoning_effort!r} cannot be used as a directory name")
        parts.append(f"reasoning-{slug}")
    if single_question:
        parts.append("single-q")
    return "__".join(parts)


def _cache_key(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None = None,
    subject_id: str | None = None,
    question_id: str | None = None,
) -> str:
    payload: dict[str, object] = {"messages": messages, "model": model, "temperature": temperature}
    # Omitted effort stays out of the hash so existing batched cache files still match.
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if subject_id is not None:
        payload["subject_id"] = subject_id
    if question_id is not None:
        payload["question_id"] = question_id
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
) -> tuple[str, str]:
    last: Exception | None = None
    for index in range(attempts):
        try:
            kwargs: dict[str, object] = {"model": model, "temperature": temperature, "timeout": timeout}
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            result = client.chat(messages, **kwargs)
            reasoning = getattr(result, "reasoning_content", "") or ""
            return str(result.content or ""), str(reasoning)
        except CRError as exc:
            last = exc
            if "网络错误" not in str(exc) or index + 1 == attempts:
                raise
            sleep(0.2 * (2**index))
    assert last is not None
    raise last


def _run_single_questions(
    client: Any,
    cache_dir: Path,
    *,
    card_text: str,
    questions: Sequence[tuple[str, str, Mapping[str, object]]],
    dataset: str,
    model: str,
    subject_id: str,
    temperature: float | None,
    reasoning_effort: str | None,
    timeout: float,
    attempts: int,
    sleep: Callable[[float], None],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Ask one pricing question per request. Cache identity includes that question."""
    codebook = {NO_OPTION: 0, YES_OPTION: 1}
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for question_id, prompt, item in questions:
        messages = build_messages(card_text, [(question_id, prompt)])
        cached = _read_single_cache(
            cache_dir,
            messages,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            subject_id=subject_id,
            question_id=question_id,
        )
        if cached is None:
            content, reasoning = _complete(
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
        else:
            content, reasoning = cached
            from_cache = True
        subject_rows = _rows(
            dataset=dataset,
            model=model,
            subject_id=subject_id,
            question_ids=[question_id],
            raw=content,
        )
        if not from_cache and all(row["parse_error"] is None for row in subject_rows):
            _write_single_cache(
                cache_dir,
                messages,
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                subject_id=subject_id,
                question_id=question_id,
                content=content,
                reasoning_content=reasoning,
            )
        rows.extend(subject_rows)
        parsed = subject_rows[0]
        answer = "" if parsed.get("answer") is None else str(parsed.get("answer"))
        mapped = None if parsed.get("parse_error") else match_option(answer, [NO_OPTION, YES_OPTION])
        purchase = None if mapped is None else code_purchase(answer, codebook)
        stimulus = parse_stimulus(str(item.get("question_text") or ""))
        records.append(
            {
                "dataset": dataset,
                "mapped_option": mapped,
                "model": model,
                "parse_error": parsed.get("parse_error"),
                "price_usd": None if stimulus is None else str(stimulus["price_usd"]),
                "product": None if stimulus is None else stimulus["product"],
                "prompt_system": messages[0]["content"],
                "prompt_user": messages[1]["content"],
                "purchase_code": purchase,
                "question_id": question_id,
                "raw_output": content,
                "reasoning_content": reasoning,
                "reasoning_effort": "default-omitted" if reasoning_effort is None else reasoning_effort,
                "subject_id": subject_id,
                "temperature": temperature,
                "unmapped_reason": None if mapped is not None or parsed.get("parse_error") else "not a unique match to one held-out option",
            }
        )
    return rows, records


def _read_single_cache(
    cache_dir: Path,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    subject_id: str,
    question_id: str,
) -> tuple[str, str] | None:
    path = cache_dir / f"{_cache_key(messages, model=model, temperature=temperature, reasoning_effort=reasoning_effort, subject_id=subject_id, question_id=question_id)}.json"
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(cached, dict) and isinstance(cached.get("content"), str):
        reasoning = cached.get("reasoning_content") or ""
        return cached["content"], reasoning if isinstance(reasoning, str) else ""
    return None


def _write_single_cache(
    cache_dir: Path,
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    subject_id: str,
    question_id: str,
    content: str,
    reasoning_content: str,
) -> None:
    path = cache_dir / f"{_cache_key(messages, model=model, temperature=temperature, reasoning_effort=reasoning_effort, subject_id=subject_id, question_id=question_id)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"content": content, "reasoning_content": reasoning_content}
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


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
