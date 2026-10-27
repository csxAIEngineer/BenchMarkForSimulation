"""Answer twin2K-economy-bechmark with one recent flash from the full store.

The model, temperature, and seed stay fixed: deepseek-v4-pro, 0, and 0.
Each ATUS card receives one free Jin10 paragraph. The pool is every flash
``scripts/read_recent.py`` treats as paragraph text, with no row limit.
Cards do not receive the other flashes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from popbench.dao.economy_benchmark import (
    DATASET_NAME,
    build_economy_benchmark_panels,
    responses_cover,
)
from popbench.dao.interview import InterviewError
from popbench.evaluate.economy_benchmark import evaluate_economy_benchmark
from popbench.simulate.client import (
    DEFAULT_BASE_URL,
    ModelSettings,
    model_settings_from_env,
    run_slug,
)
from popbench.simulate.recent import AGE_BANDS, RECENT_EVENT_HEADING, age_band_of, append_recent
from popbench.simulate.recent_store import EMPTY_STORE_MESSAGE, free_flashes
from popbench.simulate.survey import answer_survey

MODEL = "deepseek-v4-pro"
TEMPERATURE = 0.0
SEED = 0
DRAW_PREFIX = "economy-benchmark-recent:"
DRAW = f'sha256("{DRAW_PREFIX}" + person_id) % free_flash_count'


class EconomyBenchmarkRecentError(InterviewError):
    """This recent economy-benchmark run cannot use the saved directory."""


def default_run_dir(
    n: int,
    *,
    splice_recent: bool = True,
    emphasis: str | None = None,
) -> Path:
    """Directory for this locked model, temperature, and seed.

    A stressed instruction gets its own suffix. Leaving recent off removes
    the ``-recent`` suffix, so those answers cannot reuse a spliced cache.
    """
    name = f"{DATASET_NAME}-n{n}-seed{SEED}-t0"
    if splice_recent:
        name += "-recent"
    if str(emphasis or "").strip():
        name += "-emphasis"
    return Path.cwd() / "runs" / "compare" / name


def economy_benchmark_recent_kind(person_id: str) -> str:
    """Flash id drawn for this person. The same id always matches."""
    return str(_assigned(_rows(), person_id)["id"])


def economy_benchmark_recent_text(age_band: str, person_id: str) -> str:
    """Plain text of the drawn flash. Age band is checked and does not change it."""
    _require_band(age_band)
    return _plain(_assigned(_rows(), person_id))


def economy_benchmark_recent_payload(people: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Full free-flash list and one assigned paragraph per person."""
    rows = _rows()
    assigned = [_person_row(rows, person) for person in people]
    return {
        "dataset": DATASET_NAME,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "draw": DRAW,
        "source": "free Jin10 flashes; scripts/read_recent.py paragraph rows, no limit",
        "kinds": [str(row["id"]) for row in rows],
        "people": assigned,
    }


def economy_benchmark_recent_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Draw rule and per-person flash ids, without the paragraphs."""
    kinds = [str(kind) for kind in payload["kinds"]]
    return {
        "draw": str(payload["draw"]),
        "n_flashes": len(kinds),
        "kinds": kinds,
        "by_id": {str(row["id"]): str(row["kind"]) for row in payload["people"]},
    }


def attach_economy_benchmark_recent(
    record: Mapping[str, Any],
    *,
    text: str | None = None,
) -> dict[str, Any]:
    """Copy a survey record and append one stored flash.

    The input record and its persona text are not modified. ``text`` is the
    paragraph already chosen for this person; omitting it reads the store.
    """
    _require_band(age_band_of(record))
    paragraph = _supplied_text(text) if text is not None else economy_benchmark_recent_text(
        age_band_of(record),
        str(record["id"]),
    )
    persona = dict(record["persona"])
    persona["text"] = append_recent(str(persona["text"]), paragraph, heading=RECENT_EVENT_HEADING)
    updated = dict(record)
    updated["persona"] = persona
    return updated


def run_economy_benchmark_recent(
    data_dir: Path,
    *,
    n: int = 50,
    workers: int = 4,
    run_dir: Path | None = None,
    score_only: bool = False,
    splice_recent: bool = True,
    emphasis: str | None = None,
) -> dict[str, Any]:
    """Answer the locked economy slice, then score it.

    The default splices one full-store flash onto each card. ``splice_recent``
    false leaves the ATUS card unchanged. ``emphasis`` is a stressed
    instruction placed above that card when the questions are asked.
    """
    _require_positive("n", n)
    _require_positive("workers", workers)
    stressed = str(emphasis or "").strip()
    root = (
        Path(run_dir)
        if run_dir is not None
        else default_run_dir(n, splice_recent=splice_recent, emphasis=stressed)
    )
    slug = run_slug(MODEL)
    recent_report = None
    if score_only and splice_recent:
        payload = _read_saved_recent(root)
        saved_n = len(payload["people"])
        if saved_n != n:
            raise EconomyBenchmarkRecentError(
                f"recent.json has {saved_n} people; pass --n {saved_n}"
            )
        print(f"scoring saved {DATASET_NAME} recent run -> {root}", flush=True)
        built = build_economy_benchmark_panels(Path(data_dir), n=n, seed=SEED, out_dir=root)
        recent_report = economy_benchmark_recent_report(payload)
    elif score_only:
        stressed = _read_plain_note(root, stressed)
        print(f"scoring saved {DATASET_NAME} plain run -> {root}", flush=True)
        built = build_economy_benchmark_panels(Path(data_dir), n=n, seed=SEED, out_dir=root)
    else:
        built = build_economy_benchmark_panels(Path(data_dir), n=n, seed=SEED, out_dir=root)
        if splice_recent:
            payload, reuse = _prepare_recent(root, built, slug)
            recent_report = economy_benchmark_recent_report(payload)
            detail = f"recent={len(payload['kinds'])}"
        else:
            reuse = _prepare_plain(root, built, slug, stressed)
            detail = "recent=0"
        if stressed:
            detail += " emphasis=1"
        print(
            f"answer {built['n']} ATUS personas on {DATASET_NAME} "
            f"model={MODEL} temperature=0 seed={SEED} api={DEFAULT_BASE_URL} {detail} -> {root}",
            flush=True,
        )
        _answer_panels(
            built,
            root,
            slug,
            workers=workers,
            reuse=reuse,
            emphasis=stressed,
            splice_recent=splice_recent,
        )
    eval_kwargs: dict[str, Any] = {
        "slug": slug,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "recent": recent_report,
    }
    if stressed:
        eval_kwargs["emphasis"] = stressed
    if not splice_recent:
        eval_kwargs["splice_recent"] = False
    return evaluate_economy_benchmark(root, built["panels"], **eval_kwargs)


def _person_row(rows: Sequence[Mapping[str, Any]], person: Mapping[str, Any]) -> dict[str, Any]:
    person_id = str(person["id"])
    if not person_id.strip():
        raise ValueError("economy benchmark recent needs a person id")
    age_band = _require_band(age_band_of(person))
    flash = _assigned(rows, person_id)
    return {
        "id": person_id,
        "kind": str(flash["id"]),
        "age_band": age_band,
        "time": str(flash["time"]),
        "topics": list(flash["topics"]),
        "text": _plain(flash),
    }


def _prepare_recent(
    root: Path,
    built: Mapping[str, Any],
    slug: str,
) -> tuple[dict[str, Any], bool]:
    """Write the splice. Reuse saved answers only after a finished run."""
    payload = dict(economy_benchmark_recent_payload(built["personas"]))
    path = Path(root) / "recent.json"
    reuse = False
    if path.is_file():
        saved = _load_recent(path)
        if _same_recent(saved, payload):
            reuse = saved.get("answered") is True
        elif _has_saved_work(root, slug, built["panels"]):
            raise EconomyBenchmarkRecentError(
                f"recent store no longer matches {path}; choose a new --run directory"
            )
    elif _has_saved_work(root, slug, built["panels"]):
        raise EconomyBenchmarkRecentError(
            f"recent store no longer matches {path}; choose a new --run directory"
        )
    payload["answered"] = reuse
    _write_recent(path, payload)
    by_text = {str(row["id"]): str(row["text"]) for row in payload["people"]}
    for panel in built["panels"]:
        panel["records"] = [_attach_known(record, by_text) for record in panel["records"]]
    return payload, reuse


def _attach_known(record: Mapping[str, Any], by_text: Mapping[str, str]) -> dict[str, Any]:
    person_id = str(record["id"])
    try:
        paragraph = by_text[person_id]
    except KeyError as exc:
        raise EconomyBenchmarkRecentError(f"no recent flash for {person_id}") from exc
    return attach_economy_benchmark_recent(record, text=paragraph)


def _answer_panels(
    built: Mapping[str, Any],
    root: Path,
    slug: str,
    *,
    workers: int,
    reuse: bool,
    emphasis: str = "",
    splice_recent: bool = True,
) -> None:
    settings = None
    for panel in built["panels"]:
        run_dir = Path(root) / slug / str(panel["family"])
        if reuse and responses_cover(run_dir / "responses.jsonl", panel["records"]):
            print(f"scoring {MODEL} {panel['family']}", flush=True)
            continue
        if settings is None:
            settings = _model_settings()
        print(
            f"simulating {MODEL} {panel['family']} "
            f"({len(panel['items'])} items) -> {run_dir}",
            flush=True,
        )
        answer_kwargs: dict[str, Any] = {
            "panel": "atus",
            "seed": SEED,
            "family": panel["family"],
        }
        if emphasis:
            answer_kwargs["emphasis"] = emphasis
        answer_survey(
            panel["records"],
            run_dir,
            workers=workers,
            settings=settings,
            **answer_kwargs,
        )
    if splice_recent:
        _mark_answered(Path(root) / "recent.json")
    else:
        _mark_plain_answered(Path(root) / "run.json")


def _read_saved_recent(root: Path) -> dict[str, Any]:
    path = Path(root) / "recent.json"
    if not path.is_file():
        raise EconomyBenchmarkRecentError(f"missing {path}")
    payload = _load_recent(path)
    people = payload.get("people")
    kinds = payload.get("kinds")
    if payload.get("draw") != DRAW or not isinstance(people, list) or not people:
        raise EconomyBenchmarkRecentError(f"{path} is not an economy benchmark recent file")
    if not isinstance(kinds, list) or not kinds:
        raise EconomyBenchmarkRecentError(f"{path} is not an economy benchmark recent file")
    for row in people:
        if not isinstance(row, dict):
            raise EconomyBenchmarkRecentError(f"{path} people rows need id and kind")
        person_id = str(row.get("id") or "").strip()
        kind = str(row.get("kind") or "").strip()
        if not person_id or not kind:
            raise EconomyBenchmarkRecentError(f"{path} people rows need id and kind")
    return payload


def _prepare_plain(
    root: Path,
    built: Mapping[str, Any],
    slug: str,
    emphasis: str,
) -> bool:
    """Record a run that does not splice recent. Reuse only the same instruction."""
    path = Path(root) / "recent.json"
    if path.is_file():
        raise EconomyBenchmarkRecentError(
            f"{path} splices recent flashes; choose a new --run directory"
        )
    note = _plain_note(emphasis)
    note_path = Path(root) / "run.json"
    reuse = False
    if note_path.is_file():
        saved = _load_recent(note_path)
        if _same_plain(saved, note):
            reuse = saved.get("answered") is True
        elif _has_saved_work(root, slug, built["panels"]):
            raise EconomyBenchmarkRecentError(
                f"saved emphasis differs from {note_path}; choose a new --run directory"
            )
    elif _has_saved_work(root, slug, built["panels"]):
        raise EconomyBenchmarkRecentError(
            f"saved answers have no run note; choose a new --run directory"
        )
    note["answered"] = reuse
    _write_recent(note_path, note)
    return reuse


def _plain_note(emphasis: str) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "splice_recent": False,
        "emphasis": emphasis,
        "transport": "deepseek",
        "base_url": DEFAULT_BASE_URL,
    }


def _same_plain(saved: Mapping[str, Any], note: Mapping[str, Any]) -> bool:
    return (
        saved.get("splice_recent") is False
        and saved.get("model") == note["model"]
        and saved.get("emphasis") == note["emphasis"]
        and saved.get("transport") == note["transport"]
        and saved.get("base_url") == note["base_url"]
    )


def _read_plain_note(root: Path, emphasis: str) -> str:
    path = Path(root) / "run.json"
    if not path.is_file():
        raise EconomyBenchmarkRecentError(f"missing {path}")
    saved = _load_recent(path)
    if saved.get("splice_recent") is not False:
        raise EconomyBenchmarkRecentError(f"{path} is not a plain economy run")
    stored = str(saved.get("emphasis") or "").strip()
    supplied = emphasis.strip()
    if supplied and stored and supplied != stored:
        raise EconomyBenchmarkRecentError(f"{path} emphasis differs from this command")
    return stored or supplied


def _mark_plain_answered(path: Path) -> None:
    payload = dict(_load_recent(path))
    if payload.get("answered") is True:
        return
    payload["answered"] = True
    _write_recent(path, payload)


def _model_settings() -> ModelSettings:
    """deepseek-v4-pro on https://api.deepseek.com. This command does not use CR.

    Thinking is disabled. The official API ignores temperature while thinking
    is enabled, so temperature 0 would not apply.
    """
    base = model_settings_from_env()
    settings = replace(
        base,
        base_url=DEFAULT_BASE_URL,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=None,
        extra_body={"thinking": {"type": "disabled"}},
        transport="deepseek",
    )
    if settings.model != MODEL or settings.transport != "deepseek":
        raise EconomyBenchmarkRecentError(f"model resolved to {settings.model}")
    if settings.base_url != DEFAULT_BASE_URL:
        raise EconomyBenchmarkRecentError(f"DeepSeek base URL resolved to {settings.base_url}")
    return settings


def _mark_answered(path: Path) -> None:
    payload = dict(_load_recent(path))
    if payload.get("answered") is True:
        return
    payload["answered"] = True
    _write_recent(path, payload)


def _same_recent(saved: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    if saved.get("draw") != payload["draw"]:
        return False
    if list(saved.get("kinds") or []) != list(payload["kinds"]):
        return False
    saved_people = saved.get("people")
    if not isinstance(saved_people, list):
        return False
    if [row.get("id") for row in saved_people] != [row["id"] for row in payload["people"]]:
        return False
    for old, new in zip(saved_people, payload["people"], strict=True):
        if old.get("kind") != new["kind"] or old.get("text") != new["text"]:
            return False
    return True


def _has_saved_work(root: Path, slug: str, panels: Sequence[Mapping[str, Any]]) -> bool:
    for panel in panels:
        run_dir = Path(root) / slug / str(panel["family"])
        if (run_dir / "responses.jsonl").is_file():
            return True
        cache = run_dir / "cache"
        if cache.is_dir() and any(cache.glob("*.json")):
            return True
    return False


def _write_recent(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_recent(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EconomyBenchmarkRecentError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise EconomyBenchmarkRecentError(f"{path} must be an object")
    return payload


def _rows() -> list[dict[str, Any]]:
    rows = free_flashes()
    if not rows:
        raise EconomyBenchmarkRecentError(EMPTY_STORE_MESSAGE)
    return rows


def _assigned(rows: Sequence[Mapping[str, Any]], person_id: str) -> Mapping[str, Any]:
    text = str(person_id)
    if not text.strip():
        raise ValueError("economy benchmark recent needs a person id")
    digest = hashlib.sha256(f"{DRAW_PREFIX}{text}".encode()).digest()
    return rows[int.from_bytes(digest, "big") % len(rows)]


def _plain(flash: Mapping[str, Any]) -> str:
    paragraph = str(flash["plain"]).strip()
    if not paragraph:
        raise ValueError(f"flash {flash['id']} has no text")
    return paragraph


def _supplied_text(text: str) -> str:
    paragraph = str(text).strip()
    if not paragraph:
        raise ValueError("recent text is empty")
    return paragraph


def _require_band(age_band: str) -> str:
    if age_band not in AGE_BANDS:
        raise KeyError(f"no recent text for age band {age_band}")
    return age_band


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EconomyBenchmarkRecentError(f"{name} must be a positive integer")
