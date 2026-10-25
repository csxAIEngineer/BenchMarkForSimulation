"""Recent text appended for the recent-experience pilot.

Round 0 is the 10.12 BFI-44 run. A recent event is one free Jin10 flash.
The paragraph on the card is that flash's plain text. Rounds 1–3 share the
newest flash. From round 4, each person is assigned one flash from the person
id and the round. The temperature 0.8 recent run draws with
``sha256("t08-recent:" + person_id)`` and does not read these rounds. The
education-level category rerun draws with
``sha256("education-categories-recent:" + person_id) % free_flash_count``.
Age band is checked and does not change the text. The older round sentences
stay in this file and are not appended to a card. ``answer_survey`` itself
never calls this module.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from popbench.evaluate.metrics import total_variation_distance
from popbench.simulate.recent_store import EMPTY_STORE_MESSAGE, free_flashes

# Farthest then closest, in the order published in the 10.12 significance note.
FAR_ITEMS: tuple[str, ...] = (
    "QID25_39",
    "QID25_21",
    "QID25_19",
    "QID25_27",
    "QID25_35",
    "QID25_6",
    "QID25_41",
    "QID25_18",
)
NEAR_ITEMS: tuple[str, ...] = (
    "QID25_3",
    "QID25_13",
    "QID25_33",
    "QID25_12",
    "QID25_32",
    "QID25_28",
    "QID25_16",
    "QID25_11",
)
FROZEN_ITEMS: tuple[str, ...] = FAR_ITEMS + NEAR_ITEMS

# down / up / light move the mean toward Twin-2K. protect must not be pushed away.
ITEM_MOVE: dict[str, str] = {
    "QID25_39": "down",
    "QID25_21": "down",
    "QID25_19": "down",
    "QID25_27": "up",
    "QID25_35": "down",
    "QID25_6": "down",
    "QID25_41": "light",
    "QID25_18": "protect",
    "QID25_3": "protect",
    "QID25_13": "protect",
    "QID25_33": "protect",
    "QID25_12": "protect",
    "QID25_32": "protect",
    "QID25_28": "protect",
    "QID25_16": "protect",
    "QID25_11": "protect",
}
ITEM_LABEL: dict[str, str] = {
    "QID25_39": "容易紧张",
    "QID25_21": "偏安静",
    "QID25_19": "常担心",
    "QID25_27": "冷淡疏离",
    "QID25_35": "喜欢常规",
    "QID25_6": "矜持",
    "QID25_41": "艺术兴趣少",
    "QID25_18": "杂乱",
    "QID25_3": "彻底",
    "QID25_13": "可靠",
    "QID25_33": "有效率",
    "QID25_12": "少争吵",
    "QID25_32": "体贴",
    "QID25_28": "坚持",
    "QID25_16": "热情",
    "QID25_11": "精力",
}

AGE_BANDS: tuple[str, ...] = ("18-29", "30-49", "50-64", "65+")

# Relative to round 0. Far TVD must fall by at least this much.
FAR_TVD_DROP_MIN = 0.02
# Near TVD and near mean-absolute gap may rise by at most this much.
NEAR_TVD_RISE_MAX = 0.02
NEAR_ABS_RISE_MAX = 0.05

RECENT_HEADING = "## Recent weeks"

_ROUND_1: dict[str, str] = {
    "18-29": (
        "In the past two weeks, this person picked up an extra closing shift after someone called out. "
        "The line at the register was long. This person asked the supervisor one question about the drawer, "
        "and that settled it. A quiz was moved to an earlier day, and this person worked one practice problem "
        "from a different chapter. On the bus, this person left a group chat unread. On the walk home, "
        "this person stopped longer on a mural than on the other storefronts."
    ),
    "30-49": (
        "In the past two weeks, this person's kitchen sink backed up on a weeknight. This person made a short "
        "call to the building desk, got a time window, and washed that night's dishes in a basin. One grocery "
        "run went to a different store because the usual one had closed for inventory. A neighbor knocked about "
        "a package. This person handed it over at the doorway and ended the talk there. A song on the radio "
        "during the dishes got a minute of humming."
    ),
    "50-64": (
        "In the past two weeks, this person got a supplier email with the wrong delivery day and sent one reply "
        "correcting the date. In a meeting that ran long, this person made one comment when the schedule came up. "
        "Lunch was at a different table in the break room one day, and this person did not join the conversation "
        "there. On the way out, this person noticed a poster in the lobby."
    ),
    "65+": (
        "In the past two weeks, this person's clinic checkup ran late. This person mentioned the delay once at "
        "the desk and then read a magazine while waiting. One walk turned down a side street instead of staying "
        "on the usual block. A neighbor waved from a porch. This person nodded and did not stop to talk. "
        "The magazine had a short piece about a local painter."
    ),
}

# Round 1 made quiet/reserved and nervous/worry worse or flat, and the
# withdrawn beats pushed enthusiasm and energy further from Twin-2K.
# Round 2 only strengthens speaking and a snag that is handled in one step,
# and drops the unread-chat / did-not-join / nod-and-leave lines.
_ROUND_2: dict[str, str] = {
    "18-29": (
        "In the past two weeks, during a closing shift, the card reader failed on one order. "
        "This person tried another card, asked the customer a question about the receipt, and the line "
        "kept moving. In class, this person answered out loud when the instructor asked about an assignment, "
        "and worked one practice problem from a different chapter. A classmate asked about the quiz in a "
        "group chat, and this person answered in one sentence. On the walk home, this person stopped longer "
        "on a mural than on the other storefronts."
    ),
    "30-49": (
        "In the past two weeks, the kitchen sink backed up on a weeknight. This person called the building "
        "desk, asked for a time window, and washed that night's dishes in a basin. While the water ran, "
        "this person called a friend and talked through a grocery list. The grocery run went to a different "
        "store. A neighbor brought a package to the door; this person said thanks and went back to the dishes. "
        "A song on the radio got a minute of humming."
    ),
    "50-64": (
        "In the past two weeks, a supplier email had the wrong delivery day. This person wrote one line back "
        "with the right day. In a meeting, this person answered a question about the schedule out loud, then "
        "turned back to the notes. Lunch was at a different table in the break room, and this person asked "
        "someone there about the afternoon. On the way out, this person noticed a poster in the lobby."
    ),
    "65+": (
        "In the past two weeks, a clinic checkup ran late. This person asked at the desk how much longer, "
        "then talked with the person in the next chair about a magazine. One walk turned down a side street "
        "instead of staying on the usual block. A neighbor waved from a porch; this person waved back and "
        "said they were heading to the store. The magazine had a short piece about a local painter."
    ),
}

# Round 2's repairs and thank-yous pushed conscientiousness, kindness, and
# organization away from Twin-2K, and the warmth undid the cold/aloof gain.
# Round 3 drops problems, fixes, and warm exchanges. It strengthens the routine
# changes that moved, and keeps one short reply so aloofness is still there
# without a string of unanswered chats.
_ROUND_3: dict[str, str] = {
    "18-29": (
        "In the past two weeks, this person took a different bus to a closing shift, bought lunch from a "
        "different counter, and looked at a practice problem from a different chapter. In class, this person "
        "gave the instructor a short reply and did not add another comment. On the walk home, this person "
        "stopped longer on a mural than on the other storefronts."
    ),
    "30-49": (
        "In the past two weeks, this person washed up at the sink, cooked a different weeknight meal, and did "
        "the grocery run at a different store. A neighbor stopped by the door; this person gave a short answer "
        "and stepped back inside. A song on the radio got a minute of humming."
    ),
    "50-64": (
        "In the past two weeks, this person read a supplier email, then spent the afternoon on a different "
        "stack of forms than the day before, and ate lunch at a different table. In a meeting, this person "
        "answered in a few words and did not add another point. On the way out, this person noticed a poster "
        "in the lobby."
    ),
    "65+": (
        "In the past two weeks, this person went to a clinic checkup on a different morning and walked back "
        "by a side street instead of the usual block. A neighbor waved from a porch; this person lifted a hand "
        "and kept walking. A magazine had a short piece about a local painter."
    ),
}

# Later rounds are added here after a rejected round. Do not edit an earlier round in place.
# These sentences stay in the file. Card text is read from the Jin10 store.
RECENT_ROUNDS: dict[int, dict[str, str]] = {1: _ROUND_1, 2: _ROUND_2, 3: _ROUND_3}

RECENT_EVENT_HEADING = "## Recent event"

# From round 4, each person is assigned one free flash. The salt keeps a round stable.
_PERSONAL_SALTS: dict[int, str] = {
    4: "v4841:",
    5: "v17358:",
    6: "v6282:",
}

# Temperature 0.8 recent run. Not a pilot round.
# sha256(seed + ":" + person_id) mod the free-flash count. The same id and seed always match.
UNDILUTED_EVENT_SEED = "t08-recent"

# Education-level category rerun. Not a pilot round and not the temperature draw.
# sha256("education-categories-recent:" + person_id) mod the free-flash count.
EDUCATION_RECENT_DRAW = 'sha256("education-categories-recent:" + person_id) % free_flash_count'


def _flashes() -> list[dict[str, Any]]:
    """Free Jin10 flashes, newest first. Empty store fails with the fetch hint."""
    rows = free_flashes()
    if not rows:
        raise RuntimeError(EMPTY_STORE_MESSAGE)
    return rows


def _require_band(age_band: str) -> str:
    if age_band not in AGE_BANDS:
        raise KeyError(f"no recent text for age band {age_band}")
    return age_band


def education_event_ids() -> tuple[str, ...]:
    """Free flash ids, newest first."""
    return tuple(str(row["id"]) for row in _flashes())


def _row_plain(row: Mapping[str, Any]) -> str:
    paragraph = str(row["plain"]).strip()
    if not paragraph:
        raise ValueError(f"flash {row['id']} has no text")
    return paragraph


def _assigned_row(material: str, *, wide: bool = False) -> dict[str, Any]:
    """One free flash for this hash. The index and the row come from one read."""
    rows = _flashes()
    if wide:
        digest = hashlib.sha256(material.encode()).digest()
        index = int.from_bytes(digest, "big") % len(rows)
    else:
        index = _hash_slot(material, len(rows))
    return rows[index]


def _plain_for_id(flash_id: str) -> str:
    for row in _flashes():
        if str(row["id"]) == flash_id:
            return _row_plain(row)
    raise KeyError(f"no recent event for flash {flash_id}")


def _hash_slot(material: str, size: int) -> int:
    digest = hashlib.sha256(material.encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def has_recent_round(round_index: int) -> bool:
    """True when this round has band text or a per-person draw."""
    return round_index in RECENT_ROUNDS or round_index in _PERSONAL_SALTS


def recent_is_personal(round_index: int) -> bool:
    """True when each person gets their own event."""
    return round_index in _PERSONAL_SALTS


def recent_heading(round_index: int) -> str:
    """Section title appended for this round."""
    if round_index in _PERSONAL_SALTS:
        return RECENT_EVENT_HEADING
    if round_index in RECENT_ROUNDS:
        return RECENT_HEADING
    raise KeyError(f"no recent text for round {round_index}")


def event_kind(person_id: str, round_index: int) -> str:
    """Flash id drawn for this person and round. The same inputs always match."""
    try:
        salt = _PERSONAL_SALTS[round_index]
    except KeyError as exc:
        raise KeyError(f"no event draw for round {round_index}") from exc
    if not str(person_id).strip():
        raise ValueError("recent event needs a person id")
    return str(_assigned_row(f"{salt}{person_id}:{round_index}")["id"])


def recent_text(age_band: str, round_index: int, person_id: str | None = None) -> str:
    """Plain text of the Jin10 flash for this age band and round.

    Rounds 1–3 share the newest free flash. From round 4 the flash depends on
    ``person_id``, and the same id and round always return the same text.
    Age band is checked and does not change the paragraph.
    """
    _require_band(age_band)
    if round_index in _PERSONAL_SALTS:
        if not person_id or not str(person_id).strip():
            raise ValueError("recent text for this round needs a person id")
        salt = _PERSONAL_SALTS[round_index]
        return _row_plain(_assigned_row(f"{salt}{person_id}:{round_index}"))
    if round_index not in RECENT_ROUNDS:
        raise KeyError(f"no recent text for round {round_index}")
    return _row_plain(_flashes()[0])


def append_recent(persona_text: str, recent: str, *, heading: str | None = None) -> str:
    """Return the card plus a recent section.

    The caller's string is left as it was. Strings are not edited in place.
    Rounds 1–3 use ``## Recent weeks``. Later rounds pass ``## Recent event``.
    """
    paragraph = recent.strip()
    if not paragraph:
        raise ValueError("recent text is empty")
    title = RECENT_HEADING if heading is None else heading.strip()
    if not title:
        raise ValueError("recent heading is empty")
    return f"{persona_text.rstrip()}\n\n{title}\n{paragraph}\n"


def round_recent_payload(
    round_index: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Flash text for one pilot round, from a single read of the store."""
    rows = _flashes()
    if round_index in _PERSONAL_SALTS:
        salt = _PERSONAL_SALTS[round_index]
        text: dict[str, str] = {}
        kind: dict[str, str] = {}
        for record in records:
            person_id = str(record["id"])
            if not person_id.strip():
                raise ValueError("recent event needs a person id")
            _require_band(age_band_of(record))
            index = _hash_slot(f"{salt}{person_id}:{round_index}", len(rows))
            flash = rows[index]
            text[person_id] = _row_plain(flash)
            kind[person_id] = str(flash["id"])
        return {"round": round_index, "by": "person", "text": text, "kind": kind}
    if round_index not in RECENT_ROUNDS:
        raise KeyError(f"no recent text for round {round_index}")
    paragraph = _row_plain(rows[0])
    flash_id = str(rows[0]["id"])
    for record in records:
        _require_band(age_band_of(record))
    return {
        "round": round_index,
        "by": "age_band",
        "text": {band: paragraph for band in AGE_BANDS},
        "kind": {band: flash_id for band in AGE_BANDS},
    }


def undiluted_event_kind(person_id: str, *, seed: str = UNDILUTED_EVENT_SEED) -> str:
    """Flash id drawn for this person. The same id and seed always match."""
    if not str(person_id).strip():
        raise ValueError("undiluted event needs a person id")
    if not str(seed).strip():
        raise ValueError("undiluted event seed is empty")
    return str(_assigned_row(f"{seed}:{person_id}")["id"])


def undiluted_event_of_kind(kind: str, age_band: str) -> str:
    """Plain text of one stored flash. Age band is checked and does not change it."""
    _require_band(age_band)
    return _plain_for_id(str(kind))


def undiluted_event_text(
    age_band: str,
    person_id: str,
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> str:
    """Undiluted flash for this person. Age band does not change the text."""
    _require_band(age_band)
    if not str(person_id).strip():
        raise ValueError("undiluted event needs a person id")
    if not str(seed).strip():
        raise ValueError("undiluted event seed is empty")
    return _row_plain(_assigned_row(f"{seed}:{person_id}"))


def undiluted_event_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> dict[str, Any]:
    """Flash catalog and one id per person, from a single read of the store."""
    if not str(seed).strip():
        raise ValueError("undiluted event seed is empty")
    rows = _flashes()
    events = [{"id": str(row["id"]), "text": _row_plain(row)} for row in rows]
    people = []
    for record in records:
        person_id = str(record["id"])
        if not person_id.strip():
            raise ValueError("undiluted event needs a person id")
        index = _hash_slot(f"{seed}:{person_id}", len(rows))
        people.append({"id": person_id, "kind": str(rows[index]["id"])})
    return {"seed": seed, "events": events, "people": people}


def undiluted_event_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> dict[str, int]:
    """How many people drew each free flash."""
    payload = undiluted_event_payload(records, seed=seed)
    counts = {str(event["id"]): 0 for event in payload["events"]}
    for row in payload["people"]:
        counts[str(row["kind"])] += 1
    return counts


def attach_undiluted_event(
    record: Mapping[str, Any],
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> dict[str, Any]:
    """Copy a survey record and append one stored flash.

    The input record and its persona text are not modified.
    """
    persona = dict(record["persona"])
    persona["text"] = append_recent(
        str(persona["text"]),
        undiluted_event_text(str(persona["age_band"]), str(record["id"]), seed=seed),
        heading=RECENT_EVENT_HEADING,
    )
    updated = dict(record)
    updated["persona"] = persona
    return updated


def education_recent_kind(person_id: str) -> str:
    """Flash id for this person. The same id always matches."""
    text = str(person_id)
    if not text.strip():
        raise ValueError("education recent needs a person id")
    return str(_assigned_row(f"education-categories-recent:{text}", wide=True)["id"])


def education_recent_text(age_band: str, person_id: str) -> str:
    """Stored flash for this person. Age band is checked and does not change it."""
    _require_band(age_band)
    text = str(person_id)
    if not text.strip():
        raise ValueError("education recent needs a person id")
    return _row_plain(_assigned_row(f"education-categories-recent:{text}", wide=True))


def education_recent_row(person: Mapping[str, Any]) -> dict[str, Any]:
    """Flash id, time, topics, and the plain paragraph for one person."""
    person_id = str(person["id"])
    if not person_id.strip():
        raise ValueError("education recent needs a person id")
    age_band = _require_band(age_band_of(person))
    flash = _assigned_row(f"education-categories-recent:{person_id}", wide=True)
    return {
        "id": person_id,
        "kind": str(flash["id"]),
        "age_band": age_band,
        "time": str(flash["time"]),
        "topics": list(flash["topics"]),
        "text": _row_plain(flash),
    }


def education_recent_payload(people: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recent-event file: the draw, the stored flashes, and one row per person."""
    rows = _flashes()
    assigned = []
    for person in people:
        person_id = str(person["id"])
        if not person_id.strip():
            raise ValueError("education recent needs a person id")
        age_band = _require_band(age_band_of(person))
        digest = hashlib.sha256(f"education-categories-recent:{person_id}".encode()).digest()
        flash = rows[int.from_bytes(digest, "big") % len(rows)]
        assigned.append(
            {
                "id": person_id,
                "kind": str(flash["id"]),
                "age_band": age_band,
                "time": str(flash["time"]),
                "topics": list(flash["topics"]),
                "text": _row_plain(flash),
            }
        )
    return {
        "draw": EDUCATION_RECENT_DRAW,
        "kinds": [str(row["id"]) for row in rows],
        "people": assigned,
    }


def education_recent_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Stored flashes and the per-id draw, without the paragraphs."""
    return {
        "draw": payload["draw"],
        "kinds": list(payload["kinds"]),
        "by_id": {row["id"]: row["kind"] for row in payload["people"]},
    }


def attach_education_recent(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a survey record and append this person's stored flash.

    The input record and its persona text are not modified.
    """
    persona = dict(record["persona"])
    persona["text"] = append_recent(
        str(persona["text"]),
        education_recent_text(age_band_of(record), str(record["id"])),
        heading=RECENT_EVENT_HEADING,
    )
    updated = dict(record)
    updated["persona"] = persona
    return updated



def age_band_of(person: Mapping[str, Any]) -> str:
    """Age band on a persona row or an interview record."""
    demographics = person.get("demographics") or {}
    if isinstance(demographics, Mapping) and demographics.get("age_band"):
        return str(demographics["age_band"])
    persona = person.get("persona")
    if isinstance(persona, Mapping) and persona.get("age_band"):
        return str(persona["age_band"])
    if person.get("age_band"):
        return str(person["age_band"])
    raise KeyError("persona has no age band")


def first_per_band(
    personas: Sequence[Mapping[str, Any]],
    n_per_band: int = 8,
) -> list[Mapping[str, Any]]:
    """First ``n_per_band`` people of each age band, in the order they are stored."""
    if n_per_band < 1:
        raise ValueError("n_per_band must be positive")
    counts = {band: 0 for band in AGE_BANDS}
    chosen: list[Mapping[str, Any]] = []
    for person in personas:
        band = age_band_of(person)
        if band not in counts or counts[band] >= n_per_band:
            continue
        counts[band] += 1
        chosen.append(person)
    missing = [band for band, count in counts.items() if count < n_per_band]
    if missing:
        raise ValueError(f"not enough people in age bands: {', '.join(missing)}")
    return chosen


def attach_recent(record: Mapping[str, Any], round_index: int) -> dict[str, Any]:
    """Copy a survey record and append that round's recent text.

    The input record and its persona text are not modified.
    """
    persona = dict(record["persona"])
    persona["text"] = append_recent(
        str(persona["text"]),
        recent_text(str(persona["age_band"]), round_index, person_id=str(record["id"])),
        heading=recent_heading(round_index),
    )
    updated = dict(record)
    updated["persona"] = persona
    return updated


def score_frozen(
    responses: Sequence[Mapping[str, Any]],
    human_by_item: Mapping[str, Mapping[str, Any]],
    item_ids: Sequence[str] = FROZEN_ITEMS,
) -> dict[str, dict[str, float]]:
    """TVD and raw agreement means for the frozen items.

    Human shares are the Twin-2K full-sample shares. Simulated means use the
    1–5 codes on ``responses``, skipping answers that are not ok.
    """
    scores: dict[str, dict[str, float]] = {}
    for item_id in item_ids:
        if item_id not in human_by_item:
            raise KeyError(f"human baseline is missing {item_id}")
        human = human_by_item[item_id]
        shares = [float(value) for value in human["shares"]]
        if len(shares) != 5:
            raise ValueError(f"{item_id} human shares must have length 5")
        codes = _codes(responses, item_id)
        if not codes:
            raise ValueError(f"no usable answers for {item_id}")
        counts = [0, 0, 0, 0, 0]
        for code in codes:
            counts[code - 1] += 1
        scores[item_id] = {
            "tvd": float(total_variation_distance(counts, shares)),
            "sim_mean": sum(codes) / len(codes),
            "human_mean": float(human["mean_code"]),
            "n_sim": float(len(codes)),
        }
    return scores


def compare_rounds(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, float | bool]:
    """Deltas of a later round against round 0, plus whether it is accepted."""
    base = _group_means(baseline)
    cand = _group_means(candidate)
    far_drop = base["far_mean_tvd"] - cand["far_mean_tvd"]
    near_tvd_rise = cand["near_mean_tvd"] - base["near_mean_tvd"]
    near_abs_rise = cand["near_mean_abs"] - base["near_mean_abs"]
    accepted = (
        far_drop >= FAR_TVD_DROP_MIN - 1e-9
        and near_tvd_rise <= NEAR_TVD_RISE_MAX + 1e-9
        and near_abs_rise <= NEAR_ABS_RISE_MAX + 1e-9
    )
    return {
        "accepted": accepted,
        "far_mean_tvd_baseline": base["far_mean_tvd"],
        "far_mean_tvd": cand["far_mean_tvd"],
        "far_tvd_drop": far_drop,
        "near_mean_tvd_baseline": base["near_mean_tvd"],
        "near_mean_tvd": cand["near_mean_tvd"],
        "near_tvd_rise": near_tvd_rise,
        "near_mean_abs_baseline": base["near_mean_abs"],
        "near_mean_abs": cand["near_mean_abs"],
        "near_abs_rise": near_abs_rise,
    }


def accept_round(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> bool:
    """True when far-item TVD falls and the near items stay within the caps."""
    return bool(compare_rounds(baseline, candidate)["accepted"])


def _group_means(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    for item_id in FROZEN_ITEMS:
        row = scores.get(item_id)
        if row is None or any(key not in row for key in ("tvd", "sim_mean", "human_mean")):
            raise KeyError(f"scores are missing tvd and means for {item_id}")

    def _mean_tvd(item_ids: Sequence[str]) -> float:
        return sum(float(scores[item_id]["tvd"]) for item_id in item_ids) / len(item_ids)

    def _mean_abs(item_ids: Sequence[str]) -> float:
        total = 0.0
        for item_id in item_ids:
            row = scores[item_id]
            total += abs(float(row["sim_mean"]) - float(row["human_mean"]))
        return total / len(item_ids)

    return {
        "far_mean_tvd": _mean_tvd(FAR_ITEMS),
        "near_mean_tvd": _mean_tvd(NEAR_ITEMS),
        "near_mean_abs": _mean_abs(NEAR_ITEMS),
    }


def _codes(responses: Sequence[Mapping[str, Any]], item_id: str) -> list[int]:
    codes: list[int] = []
    for response in responses:
        for answer in response.get("answers") or []:
            if answer.get("item_id") != item_id:
                continue
            code = _code(answer)
            if code is not None:
                codes.append(code)
    return codes


def _code(answer: Mapping[str, Any]) -> int | None:
    if answer.get("ok") is False:
        return None
    value = answer.get("value")
    if value is None:
        return None
    code = int(value)
    if code < 1 or code > 5:
        return None
    return code
