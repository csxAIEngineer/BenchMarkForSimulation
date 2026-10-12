"""Recent-experience text and the frozen-item accept rule."""

import pytest

import hashlib

from popbench.simulate.recent import (
    AGE_BANDS,
    FAR_ITEMS,
    FROZEN_ITEMS,
    NEAR_ITEMS,
    RECENT_ROUNDS,
    UNDILUTED_EVENT_KINDS,
    UNDILUTED_EVENT_SEED,
    accept_round,
    append_recent,
    attach_recent,
    attach_undiluted_event,
    event_kind,
    event_places,
    first_per_band,
    recent_is_personal,
    recent_text,
    score_frozen,
    undiluted_event_counts,
    undiluted_event_kind,
    undiluted_event_of_kind,
    undiluted_event_text,
)

# First eight stored personas in each age band. Round 4's draw is fixed so
# these eight are five lottery payouts, two forced speeches, and one blast.
_PILOT_EIGHT = {
    "18-29": (
        "visitor-atus-20230807230256",
        "visitor-atus-20230402231408",
        "visitor-atus-20230111222213",
        "visitor-atus-20230101232146",
        "visitor-atus-20231009232235",
        "visitor-atus-20231111230237",
        "visitor-atus-20230806230199",
        "visitor-atus-20230908232196",
    ),
    "30-49": (
        "visitor-atus-20231009230972",
        "visitor-atus-20230303231076",
        "visitor-atus-20231110230613",
        "visitor-atus-20230112221142",
        "visitor-atus-20230503230911",
        "visitor-atus-20230201232405",
        "visitor-atus-20230605230839",
        "visitor-atus-20231009231327",
    ),
    "50-64": (
        "visitor-atus-20230605231877",
        "visitor-atus-20230302232450",
        "visitor-atus-20231009231501",
        "visitor-atus-20231111232217",
        "visitor-atus-20230402232455",
        "visitor-atus-20230706231104",
        "visitor-atus-20230605230083",
        "visitor-atus-20231210231044",
    ),
    "65+": (
        "visitor-atus-20230101231591",
        "visitor-atus-20230504231203",
        "visitor-atus-20230112222050",
        "visitor-atus-20230807232028",
        "visitor-atus-20230403230769",
        "visitor-atus-20230908231260",
        "visitor-atus-20231211230834",
        "visitor-atus-20230201231816",
    ),
}


def _panel(far_tvd: float, near_tvd: float, near_gap: float) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for item_id in FAR_ITEMS:
        scores[item_id] = {"tvd": far_tvd, "sim_mean": 3.0, "human_mean": 3.0}
    for item_id in NEAR_ITEMS:
        scores[item_id] = {
            "tvd": near_tvd,
            "sim_mean": 3.0 + near_gap,
            "human_mean": 3.0,
        }
    return scores


def test_append_recent_does_not_change_the_card():
    card = "## Basics\nAge: 24\n\n## Life history\nSchooling: left before a diploma.\n"
    recent = recent_text("18-29", 1)
    updated = append_recent(card, recent)
    assert card == "## Basics\nAge: 24\n\n## Life history\nSchooling: left before a diploma.\n"
    assert updated.startswith(card.rstrip())
    assert "## Recent weeks" in updated
    assert recent in updated
    assert updated.index("## Life history") < updated.index("## Recent weeks")


def test_attach_recent_leaves_the_record_alone():
    record = {
        "id": "visitor-atus-1",
        "persona": {"id": "visitor-atus-1", "age_band": "30-49", "text": "card\n"},
        "turns": [{"item_id": "QID25_1"}],
    }
    updated = attach_recent(record, 1)
    assert record["persona"]["text"] == "card\n"
    assert updated["persona"]["text"] != record["persona"]["text"]
    assert "## Recent weeks" in updated["persona"]["text"]
    assert updated["turns"] == record["turns"]
    assert updated["persona"] is not record["persona"]


def test_same_person_same_round_text_is_stable():
    person = {"id": "visitor-atus-20230807230256", "age_band": "18-29"}
    first = recent_text(person["age_band"], 1)
    second = recent_text(person["age_band"], 1)
    assert first == second
    assert append_recent("card", first) == append_recent("card", second)
    assert len({recent_text(band, 1) for band in AGE_BANDS}) == len(AGE_BANDS)


def test_recent_rounds_stay_on_ordinary_events():
    """Rounds 1–3 stay on small events. Round 4 and later are not checked here."""
    scenes = {
        "18-29": "shift",
        "30-49": "sink",
        "50-64": "email",
        "65+": "clinic",
    }
    forbidden = (
        "i am a",
        "nervous",
        "outgoing",
        "reserved",
        "disorganized",
        "bachelor",
        "married",
        "divorced",
        "promoted",
    )
    assert 1 in RECENT_ROUNDS
    for round_index in sorted(RECENT_ROUNDS):
        if round_index >= 4:
            continue
        for band, scene in scenes.items():
            text = recent_text(band, round_index).lower()
            assert "past two weeks" in text
            assert scene in text
            assert "lottery" not in text
            assert "nuclear" not in text
            for phrase in forbidden:
                assert phrase not in text
        if round_index > 1:
            assert recent_text("18-29", round_index) != recent_text("18-29", 1)


def test_round_4_eight_people_include_lottery_and_nuclear():
    purchases = ("a ticket to a show", "a class", "a short trip")
    banned = ("invest", "budget", "donat", "flee", "energetic", "enthusiastic")
    assert recent_is_personal(4)
    assert not recent_is_personal(1)
    for band, ids in _PILOT_EIGHT.items():
        texts = [recent_text(band, 4, person_id=person_id) for person_id in ids]
        assert [recent_text(band, 4, person_id=person_id) for person_id in ids] == texts
        assert sum("lottery" in text.lower() for text in texts) == 5
        assert sum("nuclear" in text.lower() for text in texts) == 1
        speeches = [text for text in texts if "lottery" not in text.lower() and "nuclear" not in text.lower()]
        assert len(speeches) == 2
        for text in texts:
            lowered = text.lower()
            assert "past two weeks" in lowered
            for phrase in banned:
                assert phrase not in lowered
        for text in texts:
            if "lottery" in text.lower():
                assert "The money was already there." in text
                assert "stopped the days that were already on the calendar" in text
                assert "said a few words" in text
                assert "did not answer the other calls" in text
                assert sum(purchase in text for purchase in purchases) == 1
        for text in speeches:
            assert "speak" in text.lower()
            assert "After the words were said, this person left." in text
        blast = [text for text in texts if "nuclear" in text.lower()]
        assert len(blast) == 1
        assert "outside the blast" in blast[0]
        assert "The danger has passed." in blast[0]
        assert "did not stop to talk" in blast[0]


def test_round_4_same_person_is_stable_and_card_is_not_rewritten():
    person_id = _PILOT_EIGHT["18-29"][0]
    card = "## Basics\nAge: 24\n\n## Life history\nSchooling: left before a diploma.\n"
    record = {
        "id": person_id,
        "persona": {"id": person_id, "age_band": "18-29", "text": card},
        "turns": [{"item_id": "QID25_1"}],
    }
    first = recent_text("18-29", 4, person_id=person_id)
    second = recent_text("18-29", 4, person_id=person_id)
    assert first == second
    assert event_kind(person_id, 4) == event_kind(person_id, 4)
    updated = attach_recent(record, 4)
    assert record["persona"]["text"] == card
    assert updated["persona"] is not record["persona"]
    assert updated["turns"] == record["turns"]
    assert "## Recent event" in updated["persona"]["text"]
    assert first in updated["persona"]["text"]
    assert updated["persona"]["text"].index("## Life history") < updated["persona"]["text"].index(
        "## Recent event"
    )
    with pytest.raises(ValueError):
        recent_text("18-29", 4)


def test_round_5_adds_speech_and_drops_the_harmful_aftermath():
    for band, ids in _PILOT_EIGHT.items():
        texts = [recent_text(band, 5, person_id=person_id) for person_id in ids]
        assert texts == [recent_text(band, 5, person_id=person_id) for person_id in ids]
        assert sum("lottery" in text.lower() for text in texts) == 4
        assert sum("nuclear" in text.lower() for text in texts) == 1
        speeches = [text for text in texts if "speak" in text.lower()]
        assert len(speeches) == 3
        for text in speeches:
            assert "left" not in text.lower()
        for text in texts:
            if "lottery" in text.lower():
                assert "a class" in text
                assert "ticket" not in text.lower()
                assert "short trip" not in text.lower()
    person_id = _PILOT_EIGHT["18-29"][0]
    assert recent_text("18-29", 4, person_id=person_id) != recent_text("18-29", 5, person_id=person_id)


def test_round_6_speaks_more_and_drops_the_quiet_lines():
    for band, ids in _PILOT_EIGHT.items():
        texts = [recent_text(band, 6, person_id=person_id) for person_id in ids]
        assert texts == [recent_text(band, 6, person_id=person_id) for person_id in ids]
        assert sum("lottery" in text.lower() for text in texts) == 3
        assert sum("nuclear" in text.lower() for text in texts) == 1
        speeches = [text for text in texts if "spoke a few sentences" in text]
        assert len(speeches) == 4
        for text in speeches:
            assert "had to" not in text.lower()
            assert "left" not in text.lower()
        for text in texts:
            if "lottery" not in text.lower():
                continue
            assert "calendar" not in text.lower()
            assert "did not answer" not in text.lower()
            assert "class" not in text.lower()
            assert "said a few words" in text
        blast = [text for text in texts if "nuclear" in text.lower()]
        assert blast
        assert "did not stop to talk" not in blast[0]
        assert "The danger has passed." in blast[0]
        assert "outside the blast" in blast[0]


def test_round_4_age_band_changes_place_only():
    places = {
        "18-29": "on a closing shift",
        "30-49": "at home",
        "50-64": "at work",
        "65+": "at the residence",
    }
    person_id = _PILOT_EIGHT["50-64"][0]
    kind = event_kind(person_id, 4)
    texts = {band: recent_text(band, 4, person_id=person_id) for band in AGE_BANDS}
    for band, place in places.items():
        assert place in texts[band]
        assert event_kind(person_id, 4) == kind
    assert ("lottery" in texts["18-29"].lower()) == ("lottery" in texts["65+"].lower())
    assert ("nuclear" in texts["18-29"].lower()) == ("nuclear" in texts["65+"].lower())
    assert "at the residence" not in texts["18-29"]
    assert "on a closing shift" not in texts["65+"]
    assert recent_text("18-29", 1) == recent_text("18-29", 1, person_id=person_id)


def test_first_per_band_keeps_file_order():
    personas = []
    for band in AGE_BANDS:
        for index in range(3):
            personas.append(
                {
                    "id": f"{band}-{index}",
                    "demographics": {"age_band": band},
                }
            )
    chosen = first_per_band(personas, n_per_band=2)
    assert [person["id"] for person in chosen] == [
        "18-29-0",
        "18-29-1",
        "30-49-0",
        "30-49-1",
        "50-64-0",
        "50-64-1",
        "65+-0",
        "65+-1",
    ]


def test_accept_round_when_far_tvd_drops_and_near_items_hold():
    baseline = _panel(0.50, 0.10, 0.10)
    candidate = _panel(0.50 - 0.02, 0.10 + 0.02, 0.10 + 0.05)
    assert accept_round(baseline, candidate)


def test_accept_round_rejects_when_near_tvd_rises_past_the_cap():
    baseline = _panel(0.50, 0.10, 0.10)
    candidate = _panel(0.50 - 0.03, 0.10 + 0.03, 0.10)
    assert not accept_round(baseline, candidate)


def test_accept_round_rejects_when_far_tvd_does_not_drop():
    baseline = _panel(0.50, 0.10, 0.10)
    candidate = _panel(0.50 - 0.019, 0.10, 0.10)
    assert not accept_round(baseline, candidate)
    assert not accept_round(baseline, _panel(0.50, 0.10, 0.10))


def test_accept_round_rejects_when_near_mean_gap_grows_past_the_cap():
    baseline = _panel(0.50, 0.10, 0.10)
    candidate = _panel(0.50 - 0.03, 0.10, 0.10 + 0.06)
    assert not accept_round(baseline, candidate)


def test_undiluted_event_is_stable_and_place_is_the_only_band_change():
    person_id = _PILOT_EIGHT["18-29"][0]
    digest = hashlib.sha256(f"{UNDILUTED_EVENT_SEED}:{person_id}".encode()).digest()
    expected = UNDILUTED_EVENT_KINDS[int.from_bytes(digest[:8], "big") % 5]
    assert undiluted_event_kind(person_id) == expected
    assert undiluted_event_text("18-29", person_id) == undiluted_event_text("18-29", person_id)
    places = event_places()
    kind = undiluted_event_kind(person_id)
    bare = {
        band: undiluted_event_text(band, person_id).replace(places[band], "{place}")
        for band in AGE_BANDS
    }
    assert len(set(bare.values())) == 1
    assert places["18-29"] in undiluted_event_text("18-29", person_id)
    assert places["65+"] not in undiluted_event_text("18-29", person_id)
    assert undiluted_event_of_kind(kind, "50-64") == undiluted_event_text("50-64", person_id)


def test_undiluted_event_changes_what_the_person_does_now():
    banned = (
        "i am a",
        "nervous",
        "outgoing",
        "reserved",
        "disorganized",
        "energetic",
        "enthusiastic",
        "bachelor",
        "married",
        "divorced",
        "promoted",
        "spouse",
        "husband",
        "wife",
        "the danger has passed",
        "a few sentences",
        "said a few words",
        "had to",
        "did not answer",
        "did not stop to talk",
    )
    signatures = {
        "windfall": ("lottery", "calendar were dropped", "stays through them"),
        "blast": ("nuclear", "still startles", "still leaves a room"),
        "speech": ("through to the last sentence", "through to the end"),
        "rupture": ("someone close parted", "no longer calls"),
        "new_task": ("had never done", "now sits down to that clay"),
    }
    seen = set()
    for index in range(80):
        seen.add(undiluted_event_kind(f"person-{index}"))
    assert seen == set(UNDILUTED_EVENT_KINDS)
    for kind in UNDILUTED_EVENT_KINDS:
        text = undiluted_event_of_kind(kind, "30-49")
        lowered = text.lower()
        assert "past two weeks" in lowered
        assert "at home" in lowered
        for phrase in signatures[kind]:
            assert phrase in lowered
        for phrase in banned:
            assert phrase not in lowered
    counts = undiluted_event_counts(
        [{"id": f"person-{index}"} for index in range(80)]
    )
    assert sum(counts.values()) == 80
    assert set(counts) == set(UNDILUTED_EVENT_KINDS)


def test_attach_undiluted_event_leaves_the_card_alone():
    person_id = _PILOT_EIGHT["30-49"][0]
    card = "## Basics\nAge: 40\n\n## Life history\nSchooling: left before a diploma.\n"
    record = {
        "id": person_id,
        "persona": {"id": person_id, "age_band": "30-49", "text": card},
        "turns": [{"item_id": "QID25_1"}],
    }
    updated = attach_undiluted_event(record)
    assert record["persona"]["text"] == card
    assert updated["persona"] is not record["persona"]
    assert updated["turns"] == record["turns"]
    assert "## Recent event" in updated["persona"]["text"]
    assert undiluted_event_text("30-49", person_id) in updated["persona"]["text"]
    assert updated["persona"]["text"].index("## Life history") < updated["persona"]["text"].index(
        "## Recent event"
    )
    with pytest.raises(ValueError):
        undiluted_event_kind("")
    with pytest.raises(KeyError):
        undiluted_event_of_kind("windfall", "no-such-band")


def test_score_frozen_uses_human_shares_and_raw_codes():
    human = {
        item_id: {"shares": [0.2, 0.2, 0.2, 0.2, 0.2], "mean_code": 3.0}
        for item_id in FROZEN_ITEMS
    }
    responses = [
        {
            "id": str(index),
            "answers": [{"item_id": item_id, "value": 5, "ok": True} for item_id in FROZEN_ITEMS],
        }
        for index in range(4)
    ]
    responses.append(
        {
            "id": "bad",
            "answers": [{"item_id": "QID25_39", "value": 1, "ok": False}],
        }
    )
    scores = score_frozen(responses, human)
    assert scores["QID25_39"]["n_sim"] == pytest.approx(4)
    assert scores["QID25_39"]["sim_mean"] == pytest.approx(5)
    assert scores["QID25_39"]["human_mean"] == pytest.approx(3)
    assert scores["QID25_39"]["tvd"] == pytest.approx(0.8)
