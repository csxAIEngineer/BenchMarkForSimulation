"""Recent text appended for the recent-experience pilot.

Round 0 is the 10.12 BFI-44 run. Rounds 1–3 are one ordinary paragraph per
age band. From round 4, each person gets one major event drawn from the
person id and the round. The temperature 0.8 recent run uses a separate
undiluted pool, drawn from the person id and a fixed seed, and does not
read these rounds. The education-level category rerun draws one of three
events with ``sha256("education-categories-recent:" + person_id) % 3``.
Age band changes only the place. That draw does not read these rounds or
the undiluted pool. ``answer_survey`` itself never calls this module.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from popbench.evaluate.metrics import total_variation_distance

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
RECENT_ROUNDS: dict[int, dict[str, str]] = {1: _ROUND_1, 2: _ROUND_2, 3: _ROUND_3}

RECENT_EVENT_HEADING = "## Recent event"

# Age band changes only where the person was. The event itself stays the same size.
_PLACE: dict[str, str] = {
    "18-29": "on a closing shift",
    "30-49": "at home",
    "50-64": "at work",
    "65+": "at the residence",
}

_PURCHASES: tuple[str, ...] = ("a ticket to a show", "a class", "a short trip")
_SPEECHES: tuple[str, ...] = (
    "this person's name was called and this person had to speak a few sentences where others could hear",
    "a local news crew came and this person had to speak a few sentences on the spot",
    "this person had to speak a few sentences from a stage",
)
# Round 6 drops "had to". That wording raised nervousness, worry, quiet, and reserve.
_SPEECHES_SPOKEN: tuple[str, ...] = (
    "this person's name was called and this person spoke a few sentences to the others there, then answered a further question",
    "a local news crew came and this person spoke a few sentences to them, then answered a further question",
    "this person spoke a few sentences from a stage to the people there, then answered a further question",
)

# sha256(salt + person_id + ":" + round) mod 8. The salt is chosen so the stored
# first eight personas of each age band land on that round's mix. Later rounds
# add their own salt and slots. Do not edit an earlier draw in place.
#
# Round 4 is five windfalls, two forced speeches, one distant blast.
# Round 5 keeps one blast: the blast group did not push nervousness further
# from Twin-2K. Speech goes from two to three because quiet and reserved did
# not fall. Windfall fills the other four slots.
# Round 6 keeps one blast for the same reason and raises speech from three to
# four. Quiet and reserved still did not fall.
_PERSONAL_DRAWS: dict[int, tuple[str, tuple[str, ...]]] = {
    4: (
        "v4841:",
        ("windfall", "windfall", "windfall", "windfall", "windfall", "speech", "speech", "blast"),
    ),
    5: (
        "v17358:",
        ("windfall", "windfall", "windfall", "windfall", "speech", "speech", "speech", "blast"),
    ),
    6: (
        "v6282:",
        ("windfall", "windfall", "windfall", "speech", "speech", "speech", "speech", "blast"),
    ),
}

EVENT_KIND_LABEL: dict[str, str] = {
    "windfall": "中彩票",
    "speech": "被迫开口",
    "blast": "核爆炸",
}

# Temperature 0.8 recent run. Not a pilot round: rounds 4–6 stay as they were.
# sha256(seed + ":" + person_id) mod 5. The same id and seed always match.
UNDILUTED_EVENT_SEED = "t08-recent"
UNDILUTED_EVENT_KINDS: tuple[str, ...] = (
    "windfall",
    "blast",
    "speech",
    "rupture",
    "new_task",
)
UNDILUTED_EVENT_LABEL: dict[str, str] = {
    "windfall": "横财应酬",
    "blast": "远处爆炸",
    "speech": "当众说完",
    "rupture": "决裂",
    "new_task": "新任务",
}


def has_recent_round(round_index: int) -> bool:
    """True when this round has band text or a per-person draw."""
    return round_index in RECENT_ROUNDS or round_index in _PERSONAL_DRAWS


def recent_is_personal(round_index: int) -> bool:
    """True when each person gets their own event."""
    return round_index in _PERSONAL_DRAWS


def recent_heading(round_index: int) -> str:
    """Section title appended for this round."""
    if round_index in _PERSONAL_DRAWS:
        return RECENT_EVENT_HEADING
    if round_index in RECENT_ROUNDS:
        return RECENT_HEADING
    raise KeyError(f"no recent text for round {round_index}")


def event_kind(person_id: str, round_index: int) -> str:
    """Event drawn for this person and round. The same inputs always match."""
    try:
        salt, slots = _PERSONAL_DRAWS[round_index]
    except KeyError as exc:
        raise KeyError(f"no event draw for round {round_index}") from exc
    if len(slots) != 8:
        raise ValueError(f"round {round_index} draw must have 8 slots")
    return slots[_draw_slot(person_id, round_index, salt)]


def recent_text(age_band: str, round_index: int, person_id: str | None = None) -> str:
    """Recent paragraph for this age band and round.

    Rounds 1–3 depend only on the band and the round. From round 4 the
    paragraph also depends on ``person_id``, and the same id and round always
    return the same text.
    """
    if round_index in _PERSONAL_DRAWS:
        if not person_id:
            raise ValueError("recent text for this round needs a person id")
        try:
            place = _PLACE[age_band]
        except KeyError as exc:
            raise KeyError(f"no recent text for age band {age_band}") from exc
        return _personal_text(place, round_index, person_id)
    try:
        by_band = RECENT_ROUNDS[round_index]
    except KeyError as exc:
        raise KeyError(f"no recent text for round {round_index}") from exc
    try:
        return by_band[age_band]
    except KeyError as exc:
        raise KeyError(f"no recent text for age band {age_band}") from exc


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


def _draw_slot(person_id: str, round_index: int, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}{person_id}:{round_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 8


def _detail_index(person_id: str, round_index: int, size: int) -> int:
    digest = hashlib.sha256(f"{person_id}:{round_index}:detail".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def _windfall_text(place: str, purchase: str) -> str:
    return (
        "In the past two weeks, a lottery prize this person had already claimed was paid into the account. "
        "The money was already there. "
        f"This person was {place} when it arrived, and then stopped the days that were already on the calendar. "
        "Relatives and strangers called. This person answered one of those calls, said a few words, "
        "and did not answer the other calls. "
        f"This person also bought {purchase}, which is not an everyday item."
    )


def _personal_text(place: str, round_index: int, person_id: str) -> str:
    kind = event_kind(person_id, round_index)
    if kind == "windfall":
        # Round 6 drops the stopped calendar, the unanswered calls, and the
        # class. Those lines pushed messiness and the conscientiousness items
        # away from Twin-2K, and the unanswered calls kept quiet and reserved up.
        if round_index >= 6:
            return (
                "In the past two weeks, a lottery prize this person had already claimed was paid into the account. "
                "The money was already there. "
                f"This person was {place} when it arrived. "
                "Relatives and strangers called. This person answered and said a few words."
            )
        # Round 4 rotates ticket, class, and short trip. Ticket and short trip
        # pushed energy and enthusiasm away from Twin-2K, so round 5 keeps only
        # the class.
        if round_index >= 5:
            purchase = "a class"
        else:
            purchase = _PURCHASES[_detail_index(person_id, round_index, len(_PURCHASES))]
        return _windfall_text(place, purchase)
    if kind == "speech":
        # Round 4 ends by leaving. That exit pushed enthusiasm away from Twin-2K.
        # Round 5 stops after the compelled speech. Round 6 drops "had to":
        # that wording raised nervousness, worry, quiet, and reserve.
        if round_index >= 6:
            situation = _SPEECHES_SPOKEN[_detail_index(person_id, round_index, len(_SPEECHES_SPOKEN))]
            return f"In the past two weeks, while this person was {place}, {situation}."
        situation = _SPEECHES[_detail_index(person_id, round_index, len(_SPEECHES))]
        if round_index >= 5:
            return f"In the past two weeks, while this person was {place}, {situation}."
        return (
            f"In the past two weeks, while this person was {place}, {situation}. "
            "After the words were said, this person left."
        )
    if kind == "blast":
        # Round 6 drops "did not stop to talk". It carried quiet and reserved up.
        if round_index >= 6:
            return (
                "In the past two weeks, a nuclear explosion far from this person had already ended. "
                "This person was outside the blast. The danger has passed. "
                f"This person was {place} when the news arrived."
            )
        return (
            "In the past two weeks, a nuclear explosion far from this person had already ended. "
            "This person was outside the blast. The danger has passed. "
            f"This person was {place} when the news arrived, and afterward did not stop to talk."
        )
    raise ValueError(f"unknown event kind {kind}")


def event_places() -> dict[str, str]:
    """Where the person was, by age band. This is the only band difference."""
    return dict(_PLACE)


def undiluted_event_kind(person_id: str, *, seed: str = UNDILUTED_EVENT_SEED) -> str:
    """Undiluted event drawn for this person. The same id and seed always match."""
    if not str(person_id).strip():
        raise ValueError("undiluted event needs a person id")
    if not str(seed).strip():
        raise ValueError("undiluted event seed is empty")
    index = _undiluted_index(str(person_id), str(seed), len(UNDILUTED_EVENT_KINDS))
    return UNDILUTED_EVENT_KINDS[index]


def undiluted_event_of_kind(kind: str, age_band: str) -> str:
    """Text for one undiluted kind. Age band changes only the place."""
    try:
        place = _PLACE[age_band]
    except KeyError as exc:
        raise KeyError(f"no recent text for age band {age_band}") from exc
    try:
        write = _UNDILUTED_TEXT[kind]
    except KeyError as exc:
        raise KeyError(f"no undiluted event for kind {kind}") from exc
    return write(place)


def undiluted_event_text(
    age_band: str,
    person_id: str,
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> str:
    """Undiluted event for this person. Age band changes only the place."""
    return undiluted_event_of_kind(undiluted_event_kind(person_id, seed=seed), age_band)


def undiluted_event_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> dict[str, int]:
    """How many people drew each undiluted kind."""
    counts = {kind: 0 for kind in UNDILUTED_EVENT_KINDS}
    for record in records:
        counts[undiluted_event_kind(str(record["id"]), seed=seed)] += 1
    return counts


def attach_undiluted_event(
    record: Mapping[str, Any],
    *,
    seed: str = UNDILUTED_EVENT_SEED,
) -> dict[str, Any]:
    """Copy a survey record and append one undiluted event.

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


# Education-level category rerun. Not a pilot round and not the undiluted pool.
# sha256("education-categories-recent:" + person_id) mod 3. The same id always
# matches. Age band changes only the place. Everyday uses the name-called
# sentence only, not the other round-6 openings.
EDUCATION_RECENT_DRAW = 'sha256("education-categories-recent:" + person_id) % 3'
EDUCATION_RECENT_KINDS: tuple[str, ...] = ("windfall", "everyday", "blast")
EDUCATION_RECENT_LABEL: dict[str, str] = {
    "windfall": "中彩票",
    "everyday": "日常琐事",
    "blast": "核爆炸",
}


def education_recent_kind(person_id: str) -> str:
    """Event key for this person. The same id always matches."""
    text = str(person_id)
    if not text.strip():
        raise ValueError("education recent needs a person id")
    digest = hashlib.sha256(f"education-categories-recent:{text}".encode()).digest()
    return EDUCATION_RECENT_KINDS[int.from_bytes(digest, "big") % len(EDUCATION_RECENT_KINDS)]


def education_recent_of_kind(kind: str, age_band: str) -> str:
    """Text for one education-category event. Age band changes only the place."""
    try:
        place = _PLACE[age_band]
    except KeyError as exc:
        raise KeyError(f"no recent text for age band {age_band}") from exc
    try:
        write = _EDUCATION_TEXT[kind]
    except KeyError as exc:
        raise KeyError(f"no education recent event for kind {kind}") from exc
    return write(place)


def education_recent_text(age_band: str, person_id: str) -> str:
    """Education-category event for this person. Age band changes only the place."""
    return education_recent_of_kind(education_recent_kind(person_id), age_band)


def education_recent_row(person: Mapping[str, Any]) -> dict[str, str]:
    """Kind, age band, place, and the full paragraph for one person."""
    person_id = str(person["id"])
    if not person_id.strip():
        raise ValueError("education recent needs a person id")
    age_band = age_band_of(person)
    kind = education_recent_kind(person_id)
    try:
        place = _PLACE[age_band]
    except KeyError as exc:
        raise KeyError(f"no recent text for age band {age_band}") from exc
    return {
        "id": person_id,
        "kind": EDUCATION_RECENT_LABEL[kind],
        "age_band": age_band,
        "place": place,
        "text": education_recent_of_kind(kind, age_band),
    }


def education_recent_payload(people: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recent-event file: the draw, the three events, and one row per person."""
    return {
        "draw": EDUCATION_RECENT_DRAW,
        "kinds": [EDUCATION_RECENT_LABEL[kind] for kind in EDUCATION_RECENT_KINDS],
        "people": [education_recent_row(person) for person in people],
    }


def education_recent_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Three events and the per-id draw, without the paragraphs."""
    return {
        "draw": payload["draw"],
        "kinds": list(payload["kinds"]),
        "by_id": {row["id"]: row["kind"] for row in payload["people"]},
    }


def attach_education_recent(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a survey record and append this person's education-category event.

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


def _education_windfall(place: str) -> str:
    return (
        "In the past two weeks, a lottery prize this person had already claimed was paid into the account. "
        "The money was already there. "
        f"This person was {place} when it arrived. "
        "Relatives and strangers called. This person answered and said a few words."
    )


def _education_everyday(place: str) -> str:
    return (
        f"In the past two weeks, while this person was {place}, "
        "this person's name was called and this person spoke a few sentences to the others there, "
        "then answered a further question."
    )


def _education_blast(place: str) -> str:
    return (
        "In the past two weeks, a nuclear explosion far from this person had already ended. "
        "This person was outside the blast. The danger has passed. "
        f"This person was {place} when the news arrived."
    )


_EDUCATION_TEXT = {
    "windfall": _education_windfall,
    "everyday": _education_everyday,
    "blast": _education_blast,
}


def _undiluted_index(person_id: str, seed: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{person_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def _undiluted_windfall(place: str) -> str:
    return (
        "In the past two weeks, a lottery prize this person had already claimed was paid into the account. "
        "The money was already there. "
        f"This person was {place} when it arrived. "
        "The days that were already on the calendar were dropped. "
        "Relatives and strangers kept calling this person to come sit with them. "
        "This person now goes to those gatherings and stays through them, "
        "instead of keeping the day that had been planned."
    )


def _undiluted_blast(place: str) -> str:
    return (
        "In the past two weeks, a nuclear explosion far from this person had already ended. "
        "This person was outside the blast. "
        f"This person was {place} when the news arrived. "
        "Since then this person still startles at a loud sound, still checks the window before stepping out, "
        "and still leaves a room when the noise keeps up."
    )


def _undiluted_speech(place: str) -> str:
    return (
        f"In the past two weeks, while this person was {place}, this person stood where others could hear "
        "and finished saying a matter that had been held back, through to the last sentence. "
        "Since then, when people are gathered and a point is this person's to give, "
        "this person says it through to the end instead of stopping after the opening line."
    )


def _undiluted_rupture(place: str) -> str:
    return (
        f"In the past two weeks, while this person was {place}, this person and someone close parted. "
        "The parting is already done. "
        "This person no longer calls that person, no longer answers that person, "
        "and no longer goes to the places where they used to meet."
    )


def _undiluted_new_task(place: str) -> str:
    return (
        f"In the past two weeks, while this person was {place}, this person took up shaping small figures out of clay, "
        "which this person had never done. "
        "This person now sits down to that clay when the evening comes. "
        "That sitting was not part of the days before."
    )


_UNDILUTED_TEXT = {
    "windfall": _undiluted_windfall,
    "blast": _undiluted_blast,
    "speech": _undiluted_speech,
    "rupture": _undiluted_rupture,
    "new_task": _undiluted_new_task,
}


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
