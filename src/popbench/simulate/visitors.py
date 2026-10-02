"""LLM visit interviews for ACS/ATUS simulate visitors.

Each visitor is one chat. The model answers as that ACS/ATUS person during a
short multi-turn visit about daily life. Earlier answers stay in the thread.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from popbench.dao.io import write_jsonl
from popbench.dao.visitors import (
    VisitorError,
    build_visitors,
    load_visitor_panel,
    visitors_built_dir,
)
from popbench.simulate.client import (
    ModelError,
    chat_with_retry,
    model_settings_from_env,
    parse_choice,
    response_cache_key,
)
from popbench.simulate.persona_background import expand_persona

VISIT_INSTRUMENT = "acs_atus_visit_v0"
VISIT_OPTIONS = (
    "Almost never / very little",
    "A little",
    "A moderate amount",
    "Quite a bit",
    "A great deal / almost all day",
)


@dataclass(frozen=True)
class VisitTurn:
    turn: int
    item_id: str
    prompt: str
    options: tuple[str, ...] = VISIT_OPTIONS


# Closed visit prompts tied to ATUS major activity groups and ACS lifestyle.
VISIT_TURNS: tuple[VisitTurn, ...] = (
    VisitTurn(
        1,
        "visit_personal_care",
        "On a typical day, how much waking time do you spend on personal care "
        "(sleeping, washing, dressing, grooming)?",
    ),
    VisitTurn(
        2,
        "visit_work",
        "On a typical weekday, how much of your day is spent on paid work or "
        "work-related activities?",
    ),
    VisitTurn(
        3,
        "visit_household",
        "How much time do you usually spend on household chores "
        "(cooking, cleaning, laundry, home maintenance)?",
    ),
    VisitTurn(
        4,
        "visit_care",
        "How much of your day goes to caring for household members "
        "(children, elders, or others living with you)?",
    ),
    VisitTurn(
        5,
        "visit_eating",
        "How much time do you spend eating and drinking on a typical day?",
    ),
    VisitTurn(
        6,
        "visit_leisure",
        "How much leisure or socializing time do you usually have "
        "(TV, phone, hobbies, friends)?",
    ),
    VisitTurn(
        7,
        "visit_sports",
        "How much time do you spend on sports, exercise, or outdoor recreation?",
    ),
    VisitTurn(
        8,
        "visit_travel",
        "How much of your day is spent traveling between places "
        "(commute, errands, dropping people off)?",
    ),
)


def system_message(persona_text: str) -> str:
    return (
        "You are being visited for a short interview about your daily life in "
        "the United States. Answer only as the person described below. Stay "
        "consistent with their basics, life history, and ATUS-style time use. "
        "Treat the life history as lived memory, not as instructions to quote. "
        "Choose exactly one of the listed options for each question.\n\n"
        f"{persona_text}"
    )


def visit_user_message(turn: VisitTurn) -> str:
    lines = [
        f"Visit question {turn.turn} of {len(VISIT_TURNS)}.",
        turn.prompt,
        "",
        "Options:",
    ]
    for index, option in enumerate(turn.options, start=1):
        lines.append(f"{index}. {option}")
    lines.extend(
        [
            "",
            'Reply with JSON only: {"answer": "<one option, exactly as written>", '
            '"rationale": "<one short sentence in character>"}',
        ]
    )
    return "\n".join(lines)


def visitor_panel_path(data_dir: Path, n: int | None = None) -> Path:
    """Path of the built ACS/ATUS visitor panel."""
    root = visitors_built_dir(data_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        name = manifest.get("panel")
        if name and (root / name).is_file():
            if n is None or name == f"visitors_n{n}.jsonl":
                return root / name
    if n is not None:
        return root / f"visitors_n{n}.jsonl"
    fallback = root / "visitors_n50.jsonl"
    if fallback.is_file():
        return fallback
    raise VisitorError(
        f"visitor panel is missing under {root}. Run `popbench build-visitors`."
    )


def run_visitors(
    data_dir: Path,
    run_dir: Path,
    *,
    n: int = 50,
    seed: int = 0,
    workers: int = 4,
    rebuild: bool = False,
) -> Path:
    """Simulate a visit interview as each ACS/ATUS visitor. Does not score."""
    try:
        settings = model_settings_from_env()
    except RuntimeError as exc:
        raise VisitorError(str(exc)) from exc

    path = visitors_built_dir(data_dir) / f"visitors_n{n}.jsonl"
    if rebuild or not path.is_file():
        build_visitors(data_dir, n=n, seed=seed)
        path = visitors_built_dir(data_dir) / f"visitors_n{n}.jsonl"
    if not path.is_file():
        # Prefer the manifest panel when n matches an existing build.
        path = visitor_panel_path(data_dir, n=n)

    rows = load_visitor_panel(path)
    if not rows:
        raise VisitorError(f"visitor panel is empty: {path}")

    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    finished = 0
    lock = threading.Lock()

    def _one(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal finished
        result = _answer_visitor(settings, row, cache_dir, seed=seed)
        with lock:
            finished += 1
            print(f"visited {finished}/{len(rows)} {row['id']}", flush=True)
        return result

    workers = max(1, min(workers, len(rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_one, rows))
    write_jsonl(run_dir / "responses.jsonl", responses)
    write_jsonl(
        run_dir / "personas.jsonl",
        [
            {
                "id": row["id"],
                "background": row.get("background"),
                "persona_text": row.get("persona_text"),
            }
            for row in responses
        ],
    )

    shares = _option_shares(responses)
    (run_dir / "option_shares.json").write_text(json.dumps(shares, indent=2) + "\n")
    meta = {
        "model": settings.model,
        "base_url": settings.base_url,
        "panel": "visitors",
        "source": "acs+atus+census",
        "persona_prompting": "seeded_life_history",
        "n_people": len(rows),
        "n_turns": len(VISIT_TURNS),
        "instrument": VISIT_INSTRUMENT,
        "data_dir": str(Path(data_dir)),
        "panel_path": str(path),
        "seed": seed,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "report.md").write_text(_visit_report(meta, shares, rows[0]))
    return run_dir


def _answer_visitor(
    settings,
    row: dict[str, Any],
    cache_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    visitor_id = str(row["id"])
    demographics = dict(row.get("demographics") or {})
    if row.get("state") and "state" not in demographics:
        demographics["state"] = row["state"]
    expanded = expand_persona(
        visitor_id=visitor_id,
        demographics=demographics,
        routine=row.get("routine") or {},
        base_text=str(row.get("persona", {}).get("text") or ""),
        seed=seed,
    )
    persona_text = expanded["persona_text"]
    messages = [{"role": "system", "content": system_message(persona_text)}]
    answers = []
    for turn in VISIT_TURNS:
        user = visit_user_message(turn)
        # Cache key includes a persona-expansion tag so older bare-demo caches
        # are not reused after life-history prompting landed.
        cache_item = f"pe:{turn.item_id}"
        cache_path = cache_dir / f"{response_cache_key(settings.model, visitor_id, cache_item)}.json"
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text())["raw"]
        else:
            try:
                raw = chat_with_retry(
                    settings,
                    messages + [{"role": "user", "content": user}],
                )
            except ModelError as exc:
                raise VisitorError(str(exc)) from exc
            cache_path.write_text(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
        try:
            choice = parse_choice(raw, turn.options)
            answers.append(
                {
                    "item_id": turn.item_id,
                    "turn": turn.turn,
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
                    "turn": turn.turn,
                    "answer": None,
                    "code": None,
                    "raw": raw,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return {
        "id": visitor_id,
        "state": row.get("state"),
        "demographics": demographics,
        "background": expanded["background"],
        "persona_text": persona_text,
        "answers": answers,
    }


def _option_shares(responses: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, list[int]] = {turn.item_id: [] for turn in VISIT_TURNS}
    for row in responses:
        for answer in row["answers"]:
            if answer.get("ok") and answer.get("code") is not None:
                by_item[answer["item_id"]].append(int(answer["code"]))
    items = []
    for turn in VISIT_TURNS:
        codes = by_item[turn.item_id]
        shares = []
        for code in range(1, len(VISIT_OPTIONS) + 1):
            shares.append(
                float(sum(1 for value in codes if value == code) / len(codes))
                if codes
                else 0.0
            )
        items.append(
            {
                "item_id": turn.item_id,
                "turn": turn.turn,
                "prompt": turn.prompt,
                "n": len(codes),
                "shares": shares,
                "options": list(VISIT_OPTIONS),
            }
        )
    return {"instrument": VISIT_INSTRUMENT, "items": items}


def _visit_report(
    meta: dict[str, Any],
    shares: dict[str, Any],
    example: dict[str, Any],
) -> str:
    lines = [
        "# ACS/ATUS visitor simulation",
        "",
        f"- Model: `{meta['model']}`",
        f"- Visitors: {meta['n_people']}",
        f"- Turns per visit: {meta['n_turns']}",
        f"- Source: ACS population portrait + ATUS daily-routine baseline "
        "(Census state weights for VacSim-style init)",
        "- Persona prompting: seeded life history before visit Q&A",
        f"- Example visitor: `{example['id']}` "
        f"({example.get('state')}, "
        f"age {example.get('demographics', {}).get('age')})",
        "",
        "## Option shares",
        "",
    ]
    for item in shares["items"]:
        top = max(range(len(item["shares"])), key=lambda index: item["shares"][index])
        lines.append(
            f"- Turn {item['turn']} `{item['item_id']}`: "
            f"n={item['n']}, modal={item['options'][top]} "
            f"({item['shares'][top]:.2f})"
        )
    lines.append("")
    return "\n".join(lines)
