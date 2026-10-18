"""Instrument batches for the closed-ended survey, and the per-item cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from popbench.dao.interview import InterviewError
from popbench.simulate.client import ModelError, ModelSettings, parse_choice, response_cache_key
from popbench.simulate.survey import (
    answer_survey,
    batch_message,
    instrument_groups,
    share_blank_instrument,
)

OPTIONS = ("Disagree strongly", "Disagree a little", "Agree a little", "Agree strongly")
STEM = (
    "Here are a number of characteristics that may or may not apply to you. "
    "Please indicate the extent to which you agree. "
)


def _turn(
    turn: int,
    item_id: str,
    prompt: str,
    *,
    instrument: str = "Big Five",
    options: tuple[str, ...] = OPTIONS,
    kind: str = "choice",
) -> dict:
    return {
        "turn": turn,
        "item_id": item_id,
        "kind": kind,
        "family": "personality",
        "block": "Personality",
        "instrument": instrument,
        "prompt": prompt,
        "options": list(options),
    }


def _settings() -> ModelSettings:
    return ModelSettings(
        base_url="https://example.invalid",
        api_key="test",
        model="deepseek-flash",
        temperature=0,
        max_tokens=None,
        transport="deepseek",
    )


def test_same_instrument_is_one_batch_and_shared_options_are_written_once():
    turns = [
        _turn(1, "QID25_1", STEM + "Is talkative"),
        _turn(2, "QID25_2", STEM + "Tends to find fault with others"),
        _turn(3, "QID125_1", "Numbness or tingling", instrument="Anxiety", options=("Not at all", "A little")),
    ]
    groups = instrument_groups(turns)
    assert [len(group) for group in groups] == [2, 1]
    text = batch_message(groups[0], n_items=3)
    assert "Big Five" in text
    assert text.count("Options:") == 1
    assert text.count("Disagree a little") == 1
    assert text.count(STEM.strip()) == 1
    assert "Is talkative" in text
    assert "Tends to find fault with others" in text
    assert "rationale" not in text
    assert "Answers you already gave" not in text
    assert "Anxiety" not in text


def test_different_option_lists_are_each_written_once():
    turns = [
        _turn(1, "QID31", "Which spending habit fits?", options=("Tightwad", "Spendthrift")),
        _turn(2, "QID32", "Trouble limiting spending?", options=("Never", "Rarely", "Often")),
        _turn(3, "QID33", "Trouble spending money?", options=("Never", "Rarely", "Often")),
    ]
    text = batch_message(turns, n_items=3)
    assert text.count("Options:") == 2
    assert text.count("Tightwad") == 1
    assert text.count("Never") == 1
    assert "rationale" not in text


def test_blank_instrument_stays_one_question_at_a_time():
    turns = [
        _turn(1, "Q1", "First?", instrument=""),
        _turn(2, "Q2", "Second?", instrument=""),
    ]
    assert [group[0]["item_id"] for group in instrument_groups(turns)] == ["Q1", "Q2"]


def test_shared_blank_instrument_is_one_request_and_leaves_named_scales():
    first = _turn(1, "Q1", "Sex?", instrument="")
    second = _turn(2, "Q2", "Age band?", instrument="")
    named = _turn(3, "Q3", "Is talkative", instrument="Big Five")
    records = [{"id": "p1", "turns": [first, second, named]}]
    stamped = share_blank_instrument(records, "demographics")
    assert first["instrument"] == ""
    assert second["instrument"] == ""
    groups = instrument_groups(stamped[0]["turns"])
    assert [[turn["item_id"] for turn in group] for group in groups] == [["Q1", "Q2"], ["Q3"]]
    assert stamped[0]["turns"][0]["instrument"] == "demographics"
    assert stamped[0]["turns"][2] is named


def test_cr_thinking_models_keep_the_caller_token_budget(tmp_path: Path, monkeypatch):
    settings = ModelSettings(
        base_url="https://api.creative-reasoning.com",
        api_key="test",
        model="deepseek-v4.1-flash",
        temperature=0,
        max_tokens=None,
        transport="cr",
    )
    seen: list[int | None] = []

    def fake_chat(call_settings, messages):
        seen.append(call_settings.max_tokens)
        return json.dumps({"answers": ["Disagree strongly"]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-1",
        "persona": {"text": "Age: 40"},
        "turns": [_turn(1, "QID25_1", STEM + "Is talkative")],
    }
    answer_survey([record], tmp_path, settings=settings)
    assert seen == [None]


def test_cached_items_are_skipped_and_new_answers_use_the_same_key(tmp_path: Path, monkeypatch):
    settings = _settings()
    persona_id = "visitor-atus-1"
    cached_turn = _turn(1, "QID25_1", STEM + "Is talkative")
    fresh_turn = _turn(2, "QID25_2", STEM + "Is reserved")
    other = _turn(3, "QID125_1", "Numbness or tingling", instrument="Anxiety", options=("Not at all", "A little"))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_path = cache_dir / f"{response_cache_key(settings.model, persona_id, cached_turn['item_id'])}.json"
    original = json.dumps(
        {"raw": json.dumps({"answer": "Agree a little", "rationale": "kept from the old call"})}
    )
    cached_path.write_text(original + "\n")
    calls: list[list[dict[str, str]]] = []

    def fake_chat(call_settings, messages):
        calls.append(messages)
        user = messages[1]["content"]
        assert "Age: 40" in messages[0]["content"]
        assert "long life history" in messages[0]["content"]
        assert call_settings.max_tokens is not None
        assert call_settings.max_tokens <= 4096
        assert "rationale" not in user
        if len(calls) == 1:
            assert "Is talkative" not in user
            assert "Is reserved" in user
            assert "Numbness" not in user
            return json.dumps({"answers": ["Disagree strongly"]})
        assert "Is reserved" not in user
        assert "Numbness or tingling" in user
        return json.dumps({"answers": [{"item": "QID125_1", "answer": "A little"}]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": persona_id,
        "persona": {"text": "Age: 40\nSex: Female\nA long life history stays on the card."},
        "turns": [cached_turn, fresh_turn, other],
    }
    responses = answer_survey([record], tmp_path, settings=settings, panel="atus", seed=0, family="personality")
    assert len(calls) == 2
    assert cached_path.read_text() == original + "\n"
    fresh_path = cache_dir / f"{response_cache_key(settings.model, persona_id, 'QID25_2')}.json"
    other_path = cache_dir / f"{response_cache_key(settings.model, persona_id, 'QID125_1')}.json"
    fresh_raw = json.loads(fresh_path.read_text())["raw"]
    other_raw = json.loads(other_path.read_text())["raw"]
    assert parse_choice(fresh_raw, OPTIONS).answer == "Disagree strongly"
    assert "rationale" not in fresh_raw
    assert parse_choice(other_raw, ("Not at all", "A little")).answer == "A little"
    by_id = {row["item_id"]: row for row in responses[0]["answers"]}
    assert by_id["QID25_1"]["value"] == 3
    assert by_id["QID25_1"]["ok"] is True
    assert "rationale" in by_id["QID25_1"]["raw"]
    assert by_id["QID25_2"]["value"] == 1
    assert by_id["QID125_1"]["value"] == 2


def test_numbered_option_text_prefers_disagree_a_little(tmp_path: Path, monkeypatch):
    settings = _settings()

    def fake_chat(_settings, _messages):
        return json.dumps({"answers": ["2. Disagree a little"]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-3",
        "persona": {"text": "Age: 40"},
        "turns": [_turn(1, "QID25_1", STEM + "Is talkative")],
    }
    responses = answer_survey([record], tmp_path, settings=settings)
    assert responses[0]["answers"][0]["value"] == 2
    assert responses[0]["answers"][0]["ok"] is True


def test_persistent_empty_content_is_a_failed_answer(tmp_path: Path, monkeypatch):
    settings = _settings()

    def fake_chat(_settings, _messages):
        raise ModelError("model request failed after retries: model response has no content", retryable=True)

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-empty",
        "persona": {"text": "Age: 40"},
        "turns": [_turn(1, "QID13", "How old are you?", instrument="", options=("18-29", "30-49", "50-64", "65+"))],
    }
    responses = answer_survey([record], tmp_path, settings=settings)
    answer = responses[0]["answers"][0]
    assert answer["ok"] is False
    assert "no content" in answer["error"]


def test_other_model_errors_still_abort_the_survey(tmp_path: Path, monkeypatch):
    settings = _settings()

    def fake_chat(_settings, _messages):
        raise ModelError("model request failed (400): bad", retryable=False)

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-bad",
        "persona": {"text": "Age: 40"},
        "turns": [_turn(1, "QID13", "How old are you?", instrument="")],
    }
    with pytest.raises(InterviewError, match="400"):
        answer_survey([record], tmp_path, settings=settings)


def test_age_band_option_is_not_stripped_as_an_index(tmp_path: Path, monkeypatch):
    settings = _settings()

    def fake_chat(_settings, _messages):
        return json.dumps({"answers": ["30-49"]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-age",
        "persona": {"text": "Age: 40"},
        "turns": [
            _turn(
                3,
                "QID13",
                "How old are you?",
                instrument="",
                options=("18-29", "30-49", "50-64", "65+"),
            )
        ],
    }
    responses = answer_survey([record], tmp_path, settings=settings)
    answer = responses[0]["answers"][0]
    assert answer["ok"] is True
    assert answer["value"] == 2


def test_catalog_typo_still_matches_agree_strongly(tmp_path: Path, monkeypatch):
    settings = _settings()

    def fake_chat(_settings, _messages):
        return json.dumps({"answers": ["Agree strongly"]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-4",
        "persona": {"text": "Age: 40"},
        "turns": [
            _turn(
                1,
                "QID233_1",
                "I rely on myself most of the time.",
                instrument="Individualism vs. collectivism",
                options=("Disagree strongly", "Agree strongy"),
            )
        ],
    }
    responses = answer_survey([record], tmp_path, settings=settings)
    assert responses[0]["answers"][0]["ok"] is True
    assert responses[0]["answers"][0]["value"] == 2


def test_short_reply_is_split_into_smaller_requests(tmp_path: Path, monkeypatch):
    settings = _settings()

    def fake_chat(_settings, messages):
        user = messages[1]["content"]
        if "Questions 1-2" in user:
            return json.dumps({"answers": ["Agree a little"]})
        if "Question 1 of" in user:
            return json.dumps({"answers": ["Agree a little"]})
        return json.dumps({"answers": ["Disagree strongly"]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-5",
        "persona": {"text": "Age: 40"},
        "turns": [
            _turn(1, "QID25_1", STEM + "Is talkative"),
            _turn(2, "QID25_2", STEM + "Is reserved"),
        ],
    }
    responses = answer_survey([record], tmp_path, settings=settings)
    assert [row["value"] for row in responses[0]["answers"]] == [3, 1]


def test_one_instrument_is_one_request_and_a_bad_reply_is_not_cached(tmp_path: Path, monkeypatch):
    settings = _settings()
    turns = [
        _turn(1, "QID25_1", STEM + "Is talkative"),
        _turn(2, "QID25_2", STEM + "Is reserved"),
    ]
    replies = iter(
        [
            "not json",
            json.dumps({"answers": ["Agree a little", "Disagree a little"]}),
        ]
    )

    def fake_chat(_settings, messages):
        assert messages[1]["content"].count("Options:") == 1
        return next(replies)

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    record = {
        "id": "visitor-atus-2",
        "persona": {"text": "Age: 22"},
        "turns": turns,
    }
    responses = answer_survey([record], tmp_path, settings=settings)
    cached = list((tmp_path / "cache").glob("*.json"))
    assert len(cached) == 2
    assert [row["value"] for row in responses[0]["answers"]] == [3, 2]


def test_resume_skips_people_already_in_the_response_file(tmp_path: Path, monkeypatch):
    settings = _settings()
    calls = {"n": 0}

    def fake_chat(_settings, _messages):
        calls["n"] += 1
        return json.dumps({"answers": ["Agree a little"]})

    monkeypatch.setattr("popbench.simulate.survey.chat_with_retry", fake_chat)
    first = {
        "id": "visitor-atus-a",
        "persona": {"text": "Age: 30"},
        "turns": [_turn(1, "QID25_1", STEM + "Is talkative")],
    }
    second = {
        "id": "visitor-atus-b",
        "persona": {"text": "Age: 31"},
        "turns": [_turn(1, "QID25_1", STEM + "Is talkative")],
    }
    answer_survey([first], tmp_path, settings=settings, resume=True)
    answer_survey([first, second], tmp_path, settings=settings, resume=True)
    assert calls["n"] == 2
    rows = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["visitor-atus-a", "visitor-atus-b"]
    assert all(row["answers"][0]["ok"] for row in rows)
