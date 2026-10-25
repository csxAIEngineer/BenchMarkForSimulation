"""Recent-experience text and the frozen-item accept rule."""

import pytest

import hashlib

from recent_db import FLASHES
from popbench.simulate.recent_store import upsert_flashes
from popbench.simulate.recent import (
    AGE_BANDS,
    FAR_ITEMS,
    FROZEN_ITEMS,
    NEAR_ITEMS,
    RECENT_ROUNDS,
    UNDILUTED_EVENT_SEED,
    accept_round,
    append_recent,
    attach_education_recent,
    attach_recent,
    attach_undiluted_event,
    education_event_ids,
    education_recent_kind,
    education_recent_payload,
    education_recent_report,
    education_recent_text,
    event_kind,
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


def _body(flash_id: str) -> str:
    return FLASHES[int(flash_id.rsplit("-", 1)[-1])]


def _log_recent(heading: str, lines: list[str]) -> None:
    """Print the recent paragraphs this test just checked."""
    print(heading)
    for line in lines:
        print(line)


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
    assert len({recent_text(band, 1) for band in AGE_BANDS}) == 1
    assert recent_text("18-29", 1) == FLASHES[0]


def test_rounds_1_to_3_share_the_latest_flash():
    """Rounds 1–3 are not drawn per person. The stored round sentences stay unused."""
    assert 1 in RECENT_ROUNDS
    assert "kitchen sink" in RECENT_ROUNDS[1]["30-49"]
    body = FLASHES[0]
    for round_index in sorted(RECENT_ROUNDS):
        if round_index >= 4:
            continue
        texts = [recent_text(band, round_index) for band in AGE_BANDS]
        assert set(texts) == {body}
        assert recent_text("30-49", round_index) != RECENT_ROUNDS[1]["30-49"]
        assert "lottery" not in texts[0]
        assert "nuclear" not in texts[0]
    _log_recent(
        "recent 第1–3轮",
        [f"[第1–3轮 {band}] {recent_text(band, 1)}" for band in AGE_BANDS],
    )


def test_round_4_eight_people_follow_the_stored_flash():
    assert recent_is_personal(4)
    assert not recent_is_personal(1)
    for band, ids in _PILOT_EIGHT.items():
        texts = [recent_text(band, 4, person_id=person_id) for person_id in ids]
        assert [recent_text(band, 4, person_id=person_id) for person_id in ids] == texts
        for person_id, text in zip(ids, texts, strict=True):
            assert text == _body(event_kind(person_id, 4))
            assert recent_text("65+", 4, person_id=person_id) == text
    _log_recent(
        "recent 第4轮",
        [
            f"[第4轮 {band} {person_id} {event_kind(person_id, 4)}] "
            f"{recent_text(band, 4, person_id=person_id)}"
            for band, ids in _PILOT_EIGHT.items()
            for person_id in ids
        ],
    )


def test_education_recent_is_the_stored_flash():
    assert education_event_ids() == tuple(f"flash-{index:02d}" for index in range(len(FLASHES)))
    person_id = "visitor-atus-20230807230256"
    kind = education_recent_kind(person_id)
    text = education_recent_text("18-29", person_id)
    assert text == _body(kind)
    assert education_recent_text("65+", person_id) == text
    assert "lottery" not in text
    assert "nuclear" not in text
    _log_recent(
        "recent 教育",
        [f"[教育 {kind} {band}] {education_recent_text(band, person_id)}" for band in ("18-29", "65+")],
    )


def test_education_recent_draw_is_stable_and_leaves_the_record():
    person_id = "visitor-atus-20230807230256"
    card = "## Basics\nAge: 24\n"
    record = {
        "id": person_id,
        "persona": {"id": person_id, "age_band": "18-29", "text": card},
        "turns": [{"item_id": "Q1"}],
    }
    digest = hashlib.sha256(f"education-categories-recent:{person_id}".encode()).digest()
    index = int.from_bytes(digest, "big") % len(FLASHES)
    kind = f"flash-{index:02d}"
    assert education_recent_kind(person_id) == kind
    assert education_recent_text("18-29", person_id) == education_recent_text("18-29", person_id)
    paragraph = FLASHES[index]
    updated = attach_education_recent(record)
    assert record["persona"]["text"] == card
    assert record["turns"] == [{"item_id": "Q1"}]
    assert updated["persona"] is not record["persona"]
    assert updated["persona"]["text"] == f"{card.rstrip()}\n\n## Recent event\n{paragraph}\n"
    payload = education_recent_payload([record])
    assert payload["kinds"] == list(education_event_ids())
    assert payload["people"][0]["kind"] == kind
    assert "place" not in payload["people"][0]
    assert payload["people"][0]["text"] == paragraph
    assert education_recent_report(payload)["by_id"][person_id] == kind


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


def test_round_5_draw_uses_the_stored_flash():
    for band, ids in _PILOT_EIGHT.items():
        texts = [recent_text(band, 5, person_id=person_id) for person_id in ids]
        assert texts == [recent_text(band, 5, person_id=person_id) for person_id in ids]
        for person_id, text in zip(ids, texts, strict=True):
            assert text == _body(event_kind(person_id, 5))
    assert any(
        event_kind(person_id, 4) != event_kind(person_id, 5)
        for ids in _PILOT_EIGHT.values()
        for person_id in ids
    )
    _log_recent(
        "recent 第5轮",
        [
            f"[第5轮 {band} {person_id} {event_kind(person_id, 5)}] "
            f"{recent_text(band, 5, person_id=person_id)}"
            for band, ids in _PILOT_EIGHT.items()
            for person_id in ids
        ],
    )


def test_round_6_draw_uses_the_stored_flash():
    for band, ids in _PILOT_EIGHT.items():
        texts = [recent_text(band, 6, person_id=person_id) for person_id in ids]
        assert texts == [recent_text(band, 6, person_id=person_id) for person_id in ids]
        for person_id, text in zip(ids, texts, strict=True):
            assert text == _body(event_kind(person_id, 6))
    _log_recent(
        "recent 第6轮",
        [
            f"[第6轮 {band} {person_id} {event_kind(person_id, 6)}] "
            f"{recent_text(band, 6, person_id=person_id)}"
            for band, ids in _PILOT_EIGHT.items()
            for person_id in ids
        ],
    )


def test_round_4_age_band_does_not_change_the_flash():
    person_id = _PILOT_EIGHT["50-64"][0]
    kind = event_kind(person_id, 4)
    texts = {band: recent_text(band, 4, person_id=person_id) for band in AGE_BANDS}
    assert set(texts.values()) == {_body(kind)}
    assert recent_text("18-29", 1) == FLASHES[0]


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


def test_undiluted_event_is_a_stored_flash():
    person_id = _PILOT_EIGHT["18-29"][0]
    digest = hashlib.sha256(f"{UNDILUTED_EVENT_SEED}:{person_id}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(FLASHES)
    flash_id = f"flash-{index:02d}"
    assert undiluted_event_kind(person_id) == flash_id
    text = undiluted_event_text("18-29", person_id)
    assert text == FLASHES[index]
    assert undiluted_event_text("65+", person_id) == text
    assert undiluted_event_of_kind(flash_id, "50-64") == text


def test_undiluted_draws_cover_the_stored_flashes():
    flash_ids = [f"flash-{index:02d}" for index in range(len(FLASHES))]
    seen = {undiluted_event_kind(f"person-{index}") for index in range(80)}
    assert seen == set(flash_ids)
    for index, flash_id in enumerate(flash_ids):
        assert undiluted_event_of_kind(flash_id, "30-49") == FLASHES[index]
    counts = undiluted_event_counts([{"id": f"person-{index}"} for index in range(80)])
    assert sum(counts.values()) == 80
    assert set(counts) == set(flash_ids)
    _log_recent(
        "recent 温度0.8",
        [
            f"[温度0.8 {flash_id}] {undiluted_event_of_kind(flash_id, '30-49')}"
            for flash_id in flash_ids
        ],
    )


def test_kinds_cycle_across_one_flash(tmp_path, monkeypatch):
    path = tmp_path / "one.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    upsert_flashes(
        [
            {
                "id": "only",
                "time": "2026-10-19 01:00:00",
                "channel": [1],
                "important": 0,
                "type": 0,
                "content": "Only this flash.",
                "vip_locked": False,
                "vip_title": "",
                "vip_level": 0,
                "site_tags": [],
                "topics": ["地缘冲突"],
            }
        ]
    )
    assert education_recent_text("50-64", "person-1") == "Only this flash."
    assert undiluted_event_text("65+", "person-1") == "Only this flash."
    person_id = _PILOT_EIGHT["18-29"][0]
    assert recent_text("18-29", 4, person_id=person_id) == "Only this flash."
    assert recent_text("30-49", 1) == "Only this flash."
    with pytest.raises(KeyError):
        undiluted_event_of_kind("missing", "18-29")


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
