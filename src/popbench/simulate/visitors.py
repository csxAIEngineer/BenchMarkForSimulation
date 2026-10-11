"""LLM visit interviews for ACS/ATUS simulate visitors.

Each visitor is one chat. The model answers as that ACS/ATUS person during a
short multi-turn visit about daily life. Earlier answers stay in the thread.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from popbench.dao.atus_populations import population_visitors_dir
from popbench.dao.io import read_jsonl, write_jsonl
from popbench.dao.visitors import (
    VisitorError,
    build_visitors,
    load_visitor_panel,
    visitors_built_dir,
)
from popbench.simulate.client import (
    ModelError,
    ModelSettings,
    SettingsError,
    chat_with_retry,
    model_settings_from_env,
    parse_choice,
    request_record,
    response_cache_key,
)
from popbench.simulate.persona_background import _compose_card, expand_persona

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
    settings: ModelSettings | None = None,
) -> Path:
    """Simulate a visit interview as each ACS/ATUS visitor. Does not score."""
    try:
        if settings is None:
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
        "request": request_record(settings),
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


POPULATION_PERSONA_SEED = 0
_LIFE_SLOTS = (
    "childhood",
    "schooling",
    "work_path",
    "family",
    "moves",
    "turning_point",
)
_EDUCATION_RULES = {
    "less_than_high_school": "They left school before a diploma. Do not give them a college degree.",
    "high_school": "They finished high school and did not earn a bachelor's degree.",
    "some_college": "They have some college or a certificate, but not a bachelor's degree.",
    "bachelor_or_higher": "They completed at least a bachelor's degree.",
    "unknown": "Their education path is uneven. Do not invent a specific degree.",
}


def simulate_population_personas(data_dir: Path, *, workers: int = 64) -> dict[str, Any]:
    """Ask the DeepSeek API for one life history per ATUS adult, then reuse that card.

    Each ``TUCASEID`` is one person. The life history comes from ``DEEPSEEK_MODEL``
    at the DeepSeek API. Group files keep every membership, and each membership
    receives that same card. Basics and ATUS minutes stay on the row. A rerun
    reads this model's cache and does not change the card.
    """
    root = population_visitors_dir(data_dir)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise VisitorError(
            f"population visitor cards are missing under {root}. "
            "Run `popbench build-population-visitors`."
        )
    manifest = json.loads(manifest_path.read_text())
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        raise VisitorError(
            f"population visitor manifest has no groups: {manifest_path}. "
            "Run `popbench build-population-visitors`."
        )
    try:
        settings = model_settings_from_env()
    except SettingsError as exc:
        raise VisitorError(str(exc)) from exc

    cache_dir = root / ".persona_cache" / settings.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[dict[str, Any], Path, list[dict[str, Any]]]] = []
    adults: dict[str, dict[str, Any]] = {}
    for group in groups:
        panel_name = group.get("panel")
        if not panel_name:
            raise VisitorError(f"population visitor group has no panel: {group}")
        panel_path = root / str(panel_name)
        if not panel_path.is_file():
            raise VisitorError(
                f"population visitor panel is missing: {panel_path}. "
                "Run `popbench build-population-visitors`."
            )
        rows = read_jsonl(panel_path)
        if not rows:
            raise VisitorError(f"population visitor panel is empty: {panel_path}")
        loaded.append((group, panel_path, rows))
        for row in rows:
            case = _case_key(row)
            if case not in adults:
                adults[case] = row
    adult_ids = sorted(adults, key=lambda case: (0, int(case)) if case.isdigit() else (1, case))
    n_group_rows = sum(len(rows) for _, _, rows in loaded)
    print(f"DeepSeek API will write {len(adult_ids)} adult persona cards", flush=True)
    finished = 0
    lock = threading.Lock()
    n_adults = len(adult_ids)

    def _one(case: str) -> tuple[str, dict[str, Any]]:
        nonlocal finished
        persona_id = _adult_persona_id(case)
        expanded = _population_persona(
            settings,
            adults[case],
            cache_dir,
            persona_id=persona_id,
        )
        with lock:
            finished += 1
            if finished == 1 or finished % 50 == 0 or finished == n_adults:
                print(f"persona {finished}/{n_adults} {persona_id}", flush=True)
        return case, expanded

    workers_n = max(1, min(workers, n_adults))
    with ThreadPoolExecutor(max_workers=workers_n) as pool:
        cards = dict(pool.map(_one, adult_ids))

    adult_records = [_adult_record(case, adults[case], cards[case]) for case in adult_ids]
    write_jsonl(root / "adults" / "personas.jsonl", adult_records)

    written_groups: list[dict[str, Any]] = []
    for group, panel_path, rows in loaded:
        updated: list[dict[str, Any]] = []
        blocks: list[str] = []
        for index, row in enumerate(rows):
            case = _case_key(row)
            expanded = cards[case]
            persona_id = _adult_persona_id(case)
            persona = dict(row.get("persona") or {})
            persona["id"] = persona_id
            persona["text"] = expanded["persona_text"]
            updated.append(
                {
                    **row,
                    "persona": persona,
                    "persona_id": persona_id,
                    "background": expanded["background"],
                    "persona_text": expanded["persona_text"],
                }
            )
            blocks.append(
                _persona_card_block(
                    row,
                    expanded["persona_text"],
                    index,
                    len(rows),
                    state=expanded["state"],
                    visitor_id=persona_id,
                )
            )
        write_jsonl(panel_path, updated)
        cards_path = panel_path.parent / "persona_cards.txt"
        cards_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        (panel_path.parent / "example.md").write_text(blocks[0] + "\n", encoding="utf-8")
        written_groups.append(
            {
                "factor": group.get("factor"),
                "level": group.get("level"),
                "n_people": len(rows),
                "panel": str(panel_path.relative_to(root)),
                "cards": str(cards_path),
            }
        )
        print(
            f"wrote {group.get('factor')}/{group.get('level')} ({len(rows)})",
            flush=True,
        )

    return {
        "model": settings.model,
        "n_groups": len(written_groups),
        "n_cards": n_adults,
        "n_adults": n_adults,
        "n_group_rows": n_group_rows,
        "dir": root,
        "groups": written_groups,
    }


def _case_key(row: dict[str, Any]) -> str:
    case = (row.get("demographics") or {}).get("TUCASEID")
    if case is None or str(case).strip() == "":
        raise VisitorError(f"population visitor {row.get('id')} has no TUCASEID")
    text = str(case).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _adult_persona_id(case: str) -> str:
    return f"visitor-atus-{case}"


def _adult_record(case: str, row: dict[str, Any], expanded: dict[str, Any]) -> dict[str, Any]:
    persona_id = _adult_persona_id(case)
    persona = dict(row.get("persona") or {})
    persona["id"] = persona_id
    persona["text"] = expanded["persona_text"]
    return {
        "id": persona_id,
        "TUCASEID": int(case) if case.isdigit() else case,
        "source": "atus-population",
        "persona": persona,
        "routine": row.get("routine"),
        "demographics": row.get("demographics"),
        "background": expanded["background"],
        "persona_text": expanded["persona_text"],
    }


def _write_persona_cache(
    cache_path: Path,
    raw: str,
    background: dict[str, Any],
    persona_text: str,
) -> None:
    payload = {"raw": raw, "background": background, "persona_text": persona_text}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(cache_path)


def _population_persona(
    settings: ModelSettings,
    row: dict[str, Any],
    cache_dir: Path,
    *,
    persona_id: str | None = None,
) -> dict[str, Any]:
    visitor_id = persona_id or str(row["id"])
    facts = _aligned_population_facts(row)
    routine = {str(key): float(value) for key, value in (row.get("routine") or {}).items()}
    cache_path = cache_dir / (
        f"{response_cache_key(settings.model, visitor_id, _persona_cache_item(facts, routine))}.json"
    )
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        return {
            "background": cached["background"],
            "persona_text": cached["persona_text"],
            "state": facts["state"],
        }
    slots, raw = _request_life_history(settings, _life_history_messages(visitor_id, facts, routine))
    background = {
        "seed": POPULATION_PERSONA_SEED,
        "visitor_id": visitor_id,
        "model": settings.model,
        **slots,
    }
    persona_text = _compose_card(
        base_text=str((row.get("persona") or {}).get("text") or ""),
        age=facts["age"],
        sex=str(facts["sex"]),
        state=facts["state"],
        education=str(facts["education"]),
        marital=str(facts["marital_status"]),
        race=str(facts["race"]),
        income_over_50k=bool(facts["income_over_50k"]),
        work_hours=facts["work_hours"],
        background=background,
        routine=routine,
        diary_day=False,
    )
    _write_persona_cache(cache_path, raw, background, persona_text)
    return {"background": background, "persona_text": persona_text, "state": facts["state"]}


def _aligned_population_facts(row: dict[str, Any]) -> dict[str, Any]:
    """Map population-table fields onto the names expand_persona expects.

    Rows have no state, personal income, or weekly hours. Those stay at the
    expand_persona defaults: unknown state, income at or below $50,000, and
    unknown hours. ``partner`` of ``spouse`` is Married; every other partner
    value is Unknown, because the life-history pools have no unmarried-partner
    or no-partner slot.
    """
    demo = dict(row.get("demographics") or {})
    if "education_level" in demo and demo.get("education_level") not in (None, ""):
        education = demo.get("education_level")
    else:
        education = demo.get("education") or "unknown"
    if "race_ethnicity" in demo and demo.get("race_ethnicity") not in (None, ""):
        race = demo.get("race_ethnicity")
    else:
        race = demo.get("race") or "Unknown"
    if "partner" in demo:
        marital = "Married" if demo.get("partner") == "spouse" else "Unknown"
    else:
        marital = demo.get("marital_status") or "Unknown"
    work_hours = demo.get("work_hours")
    if work_hours is not None:
        work_hours = float(work_hours)
    return {
        "age": demo.get("age"),
        "sex": demo.get("sex") or "Unknown",
        "state": demo.get("state") or demo.get("state_name"),
        "education": str(education),
        "marital_status": str(marital),
        "race": str(race),
        "income_over_50k": bool(demo.get("income_over_50k")),
        "work_hours": work_hours,
        "employment": demo.get("employment"),
        "children": demo.get("children"),
        "metro": demo.get("metro"),
        "school": demo.get("school"),
        "partner": demo.get("partner"),
    }


def _life_history_messages(
    visitor_id: str,
    facts: dict[str, Any],
    routine: dict[str, float],
) -> list[dict[str, str]]:
    if facts["income_over_50k"]:
        income = "Personal income: over $50,000."
    else:
        income = (
            "Personal income: $50,000 or less. Describe modest or uneven pay, "
            "not a high-paying career."
        )
    if facts["work_hours"] is None:
        hours = "Usual weekly work hours: unknown. Do not invent an exact hour count."
    else:
        hours = f"Usual weekly work hours: {float(facts['work_hours']):.0f}."
    minutes = "\n".join(
        f"- {key}: {value:.0f}"
        for key, value in sorted(routine.items(), key=lambda item: -item[1])
    )
    education = str(facts["education"])
    user = "\n".join(
        [
            "Write this person's life history. Reply with JSON only, using exactly these keys: "
            "childhood, schooling, work_path, family, moves, turning_point.",
            "Each value is one or two English sentences. Do not contradict the facts.",
            "",
            f"Visitor id: {visitor_id}",
            f"Age: {facts['age']}",
            f"Sex: {facts['sex']}",
            "State: unknown. Do not name a current US state.",
            f"Education: {education}. {_EDUCATION_RULES.get(education, _EDUCATION_RULES['unknown'])}",
            (
                f"Marital status: {facts['marital_status']}. "
                f"{_partner_rule(facts.get('partner'), str(facts['marital_status']))}"
            ),
            f"Race: {facts['race']}",
            income,
            hours,
            f"Employment: {facts.get('employment') or 'unknown'}. {_employment_rule(facts.get('employment'))}",
            (
                f"Children in household: {facts.get('children') or 'unknown'}. "
                f"{_children_rule(facts.get('children'))}"
            ),
            f"Metropolitan status: {facts.get('metro') or 'unknown'}",
            f"School enrollment: {facts.get('school') or 'unknown'}. {_school_rule(facts.get('school'))}",
            "",
            "Diary-day minutes are one diary day, not a usual week. "
            "Usual employment can differ from that day's work minutes.",
            minutes,
            "",
            "turning_point should help explain the largest time-use categories.",
            "moves may say how they arrived, but the current state stays unknown.",
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You write short life histories for survey persona cards. "
                "Reply with JSON only."
            ),
        },
        {"role": "user", "content": user},
    ]


def _partner_rule(partner: Any, marital: str) -> str:
    if partner == "spouse" or marital == "Married":
        return "They have a spouse. Do not describe them as single."
    if partner == "unmarried_partner":
        return "They live with an unmarried partner. Do not call them married."
    if partner == "none":
        return "They have no spouse or partner. Do not invent one."
    if marital == "Never married":
        return "They have never married. Do not invent a current spouse."
    return "Do not invent a current spouse."


def _employment_rule(employment: Any) -> str:
    if employment == "not_employed":
        return "They are not currently employed. Do not give them a current full-time job."
    if employment == "part_time":
        return "They work part time."
    if employment == "full_time":
        return "They usually work full time."
    return "Do not invent a job title."


def _children_rule(children: Any) -> str:
    if children and children != "no_household_child":
        return "A child lives in the household; family text may include that care."
    if children == "no_household_child":
        return "No child under 18 lives in the household. Do not invent one."
    return "Do not invent a child in the household."


def _school_rule(school: Any) -> str:
    if school == "enrolled":
        return "They are currently enrolled in school."
    if school == "not_enrolled":
        return "They are not currently enrolled in school."
    return "School enrollment is unknown."


def _request_life_history(
    settings: ModelSettings,
    messages: list[dict[str, str]],
) -> tuple[dict[str, str], str]:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            raw = chat_with_retry(settings, messages)
        except ModelError as exc:
            message = str(exc)
            if exc.status == 403:
                message += (
                    " Cached cards are kept. After access is restored, rerun "
                    "`popbench simulate-population-personas`."
                )
            raise VisitorError(message) from exc
        try:
            return _parse_life_history(raw), raw
        except ValueError as exc:
            last_error = exc
    raise VisitorError(f"{settings.model} life history was not usable JSON: {last_error}")


def _parse_life_history(raw: str) -> dict[str, str]:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response is not JSON")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response is not a JSON object")
    slots: dict[str, str] = {}
    for key in _LIFE_SLOTS:
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} is missing")
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError(f"{key} is empty")
        slots[key] = cleaned
    return slots


def _persona_cache_item(facts: dict[str, Any], routine: dict[str, float]) -> str:
    payload = {
        "age": facts["age"],
        "sex": facts["sex"],
        "education": facts["education"],
        "marital_status": facts["marital_status"],
        "race": facts["race"],
        "income_over_50k": facts["income_over_50k"],
        "work_hours": facts["work_hours"],
        "employment": facts["employment"],
        "children": facts["children"],
        "metro": facts["metro"],
        "school": facts["school"],
        "partner": facts.get("partner"),
        "routine": {key: routine[key] for key in sorted(routine)},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"population-persona-v2:{digest}"


def _persona_card_block(
    row: dict[str, Any],
    persona_text: str,
    index: int,
    n_people: int,
    *,
    state: Any,
    visitor_id: str | None = None,
) -> str:
    state_token = "None" if not state else str(state)
    header = (
        f"# visitor={visitor_id or row['id']} panel_index={index}/{n_people} "
        f"draw_seed={POPULATION_PERSONA_SEED} state={state_token}"
    )
    return f"{header}\n===== SYSTEM =====\n{system_message(persona_text)}"


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
