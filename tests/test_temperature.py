"""Temperature sweep scoring and the three higher-temperature calls."""

import json
from pathlib import Path

from popbench.simulate.client import ModelSettings
from recent_db import FLASHES
from popbench.simulate.recent import FROZEN_ITEMS
from popbench.simulate.recent import UNDILUTED_EVENT_SEED
from popbench.simulate.recent_store import upsert_flashes
from popbench.simulate.temperature import (
    human_extreme_share,
    run_sweep,
    run_t08_recent,
    tier_metrics,
)


def _settings() -> ModelSettings:
    return ModelSettings(
        base_url="https://example.invalid",
        api_key="test",
        model="deepseek-flash",
        temperature=0,
        max_tokens=None,
        transport="deepseek",
    )


def _human(item_ids: list[str]) -> dict[str, dict]:
    return {
        item_id: {
            "item_id": item_id,
            "shares": [0.25, 0.25, 0.25, 0.25, 0.0],
            "mean_code": 3.0,
        }
        for item_id in item_ids
    }


def _answers(item_ids: list[str], value: int, *, ok: bool = True) -> list[dict]:
    return [
        {"item_id": item_id, "value": value, "raw": "{}", "ok": ok}
        for item_id in item_ids
    ]


def test_tier_metrics_average_the_frozen_items_and_the_extreme_share():
    item_ids = list(FROZEN_ITEMS)
    human = _human(item_ids)
    responses = [
        {"id": "a", "answers": _answers(item_ids, 5)},
        {"id": "b", "answers": _answers(item_ids, 5)},
    ]
    responses[1]["answers"][0] = {
        "item_id": item_ids[0],
        "value": None,
        "raw": "no",
        "ok": False,
    }
    metrics = tier_metrics(responses, human, item_ids)
    assert metrics["mean_tvd"] == 1.0
    assert metrics["far_mean_tvd"] == 1.0
    assert metrics["near_mean_tvd"] == 1.0
    assert metrics["near_mean_abs"] == 2.0
    assert metrics["extreme_share"] == 1.0
    assert metrics["n_invalid"] == 1
    assert metrics["n_answers"] == 32
    assert human_extreme_share(human, item_ids) == 0.25


def test_run_sweep_reuses_temperature_zero_and_writes_the_summary(tmp_path, monkeypatch):
    item_ids = [f"QID25_{number}" for number in range(1, 45)]
    sim = tmp_path / "runs" / "simulation"
    sim.mkdir(parents=True)
    (sim / "human_baseline.json").write_text(
        json.dumps(
            {
                "n_respondents": 10,
                "items": [
                    {
                        "item_id": item_id,
                        "kind": "choice",
                        "family": "personality",
                        "n": 10,
                        "shares": [0.25, 0.25, 0.25, 0.25, 0.0],
                        "mean_code": 3.0,
                    }
                    for item_id in item_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    records = []
    rows = []
    for index in range(200):
        persona_id = f"p{index}"
        records.append({"id": persona_id, "persona": {"text": f"card {index}"}, "turns": []})
        rows.append({"id": persona_id, "answers": _answers(item_ids, 3)})
    _write_jsonl(sim / "records.jsonl", records)
    _write_jsonl(sim / "responses.jsonl", rows)
    before = (sim / "responses.jsonl").read_text(encoding="utf-8")
    calls: list[dict] = []

    def fake_answer(passed, run_dir, workers=4, settings=None, **kwargs):
        calls.append(
            {
                "temperature": settings.temperature,
                "run_dir": Path(run_dir),
                "texts": [record["persona"]["text"] for record in passed],
                "workers": workers,
            }
        )
        _write_jsonl(Path(run_dir) / "responses.jsonl", rows)

    monkeypatch.setattr("popbench.simulate.temperature.answer_survey", fake_answer)
    monkeypatch.setattr("popbench.simulate.temperature.model_settings_from_env", _settings)
    summary = run_sweep(workers=3, root=tmp_path)
    assert [call["temperature"] for call in calls] == [0.4, 0.8, 1.2]
    assert [call["run_dir"].name for call in calls] == ["t0.4", "t0.8", "t1.2"]
    assert all(call["texts"][0] == "card 0" for call in calls)
    assert (sim / "responses.jsonl").read_text(encoding="utf-8") == before
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "| 0 | 0.7500 | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0 / 8800 |" in report
    assert "人类基线的两端份额是 0.2500" in report
    again: list[dict] = []
    monkeypatch.setattr(
        "popbench.simulate.temperature.answer_survey",
        lambda *args, **kwargs: again.append(kwargs),
    )
    run_sweep(workers=3, root=tmp_path)
    assert again == []


def test_run_t08_recent_appends_one_event_and_compares_with_plain(tmp_path, monkeypatch):
    item_ids = [f"QID25_{number}" for number in range(1, 45)]
    sim = tmp_path / "runs" / "simulation"
    sim.mkdir(parents=True)
    (sim / "human_baseline.json").write_text(
        json.dumps(
            {
                "n_respondents": 10,
                "items": [
                    {
                        "item_id": item_id,
                        "kind": "choice",
                        "family": "personality",
                        "n": 10,
                        "shares": [0.25, 0.25, 0.25, 0.25, 0.0],
                        "mean_code": 3.0,
                    }
                    for item_id in item_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    records = []
    plain_rows = []
    recent_rows = []
    for index in range(200):
        persona_id = f"p{index}"
        records.append(
            {
                "id": persona_id,
                "persona": {"id": persona_id, "age_band": "30-49", "text": f"card {index}\n"},
                "turns": [],
            }
        )
        plain_rows.append({"id": persona_id, "answers": _answers(item_ids, 3)})
        recent_rows.append({"id": persona_id, "answers": _answers(item_ids, 5)})
    _write_jsonl(sim / "records.jsonl", records)
    plain_path = sim / "temperature" / "t0.8" / "responses.jsonl"
    _write_jsonl(plain_path, plain_rows)
    records_before = (sim / "records.jsonl").read_text(encoding="utf-8")
    plain_before = plain_path.read_text(encoding="utf-8")
    calls: list[dict] = []

    def fake_answer(passed, run_dir, workers=4, settings=None, **kwargs):
        calls.append(
            {
                "temperature": settings.temperature,
                "run_dir": Path(run_dir),
                "texts": [record["persona"]["text"] for record in passed],
                "workers": workers,
            }
        )
        _write_jsonl(Path(run_dir) / "responses.jsonl", recent_rows)

    monkeypatch.setattr("popbench.simulate.temperature.answer_survey", fake_answer)
    monkeypatch.setattr("popbench.simulate.temperature.model_settings_from_env", _settings)
    summary = run_t08_recent(workers=3, root=tmp_path)
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.8
    assert calls[0]["run_dir"].name == "t0.8-recent"
    assert calls[0]["workers"] == 3
    assert all("## Recent event" in text for text in calls[0]["texts"])
    assert all(text.startswith("card ") for text in calls[0]["texts"])
    assert (sim / "records.jsonl").read_text(encoding="utf-8") == records_before
    assert plain_path.read_text(encoding="utf-8") == plain_before
    assert summary["seed"] == UNDILUTED_EVENT_SEED
    assert summary["delta"]["mean_tvd"] == 0.25
    assert summary["delta"]["extreme_share"] == 1.0
    assert sum(summary["counts"].values()) == 200
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "| 0.8 无 recent | 0.7500 | 0.7500 | 0.7500 | 0.0000 | 0.0000 | 0 / 8800 |" in report
    assert "| 0.8 加 recent | 1.0000 | 1.0000 | 1.0000 | 2.0000 | 1.0000 | 0 / 8800 |" in report
    assert "| 差值 | 0.2500 | 0.2500 | 0.2500 | 2.0000 | 1.0000 | — |" in report
    assert "人类基线的两端份额是 0.2500" in report
    assert "`runs/simulation/temperature/t0.8-recent`" in report
    assert "横财应酬" not in report
    assert "核爆炸" not in report
    assert FLASHES[0] in report
    again: list[dict] = []
    monkeypatch.setattr(
        "popbench.simulate.temperature.answer_survey",
        lambda *args, **kwargs: again.append(kwargs),
    )
    upsert_flashes(
        [
            {
                "id": "later",
                "time": "2026-10-19 23:00:00",
                "channel": [1],
                "important": 0,
                "type": 0,
                "content": "A later headline that must stay off the reused card.",
                "vip_locked": False,
                "vip_title": "",
                "vip_level": 0,
                "site_tags": [],
                "topics": ["股指市场"],
            }
        ]
    )
    reused = run_t08_recent(workers=3, root=tmp_path)
    assert again == []
    reused_report = Path(reused["report"]).read_text(encoding="utf-8")
    assert "A later headline" not in reused_report
    assert FLASHES[0] in reused_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
