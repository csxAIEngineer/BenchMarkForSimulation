"""Fixed economy-benchmark run: one flash from the full recent store per card."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recent_db import FLASHES
from popbench.cli import main
from popbench.evaluate.economy_benchmark import evaluate_economy_benchmark, render_economy_benchmark
from popbench.simulate.client import ModelSettings
from popbench.simulate.interview import system_message
from popbench.simulate.economy_benchmark_recent import (
    DRAW,
    MODEL,
    SEED,
    TEMPERATURE,
    EconomyBenchmarkRecentError,
    attach_economy_benchmark_recent,
    default_run_dir,
    economy_benchmark_recent_kind,
    economy_benchmark_recent_payload,
    economy_benchmark_recent_text,
    run_economy_benchmark_recent,
)


def _index(person_id: str, size: int) -> int:
    digest = hashlib.sha256(f"economy-benchmark-recent:{person_id}".encode()).digest()
    return int.from_bytes(digest, "big") % size


def _record(person_id: str, age_band: str, card: str = "card\n") -> dict:
    return {
        "id": person_id,
        "persona": {"id": person_id, "age_band": age_band, "text": card, "sex": "Female"},
        "turns": [{"item_id": "QID229"}],
    }


def test_each_card_gets_one_flash_from_the_full_store():
    person_id = "visitor-atus-20230807230256"
    card = "## Basics\nAge: 24\n"
    record = _record(person_id, "18-29", card)
    index = _index(person_id, len(FLASHES))
    kind = f"flash-{index:02d}"

    assert DRAW == 'sha256("economy-benchmark-recent:" + person_id) % free_flash_count'
    assert economy_benchmark_recent_kind(person_id) == kind
    assert economy_benchmark_recent_text("18-29", person_id) == FLASHES[index]
    assert economy_benchmark_recent_text("65+", person_id) == FLASHES[index]

    updated = attach_economy_benchmark_recent(record)
    assert record["persona"]["text"] == card
    assert record["turns"] == [{"item_id": "QID229"}]
    assert updated["persona"] is not record["persona"]
    assert updated["turns"] == record["turns"]
    assert updated["persona"]["text"] == f"{card.rstrip()}\n\n## Recent event\n{FLASHES[index]}\n"
    for other in FLASHES:
        if other != FLASHES[index]:
            assert other not in updated["persona"]["text"]

    payload = economy_benchmark_recent_payload(
        [
            {"id": person_id, "demographics": {"age_band": "18-29"}},
            {"id": "visitor-atus-other", "age_band": "30-49"},
        ]
    )
    assert payload["draw"] == DRAW
    assert payload["kinds"] == [f"flash-{i:02d}" for i in range(len(FLASHES))]
    assert payload["people"][0]["kind"] == kind
    assert payload["people"][0]["text"] == FLASHES[index]
    kinds = [economy_benchmark_recent_kind(f"person-{i}") for i in range(12)]
    assert len(set(kinds)) > 1


EMPHASIS = (
    "你是一个人类，具有懒的性格，随机性，不会有同一锚点，还有库存、习惯、冲动、随便点一下的坏习惯。"
)


def test_default_directory_locks_model_conditions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    assert default_run_dir(50) == (
        tmp_path / "runs" / "compare" / "twin2K-economy-bechmark-n50-seed0-t0-recent"
    )
    assert default_run_dir(50, splice_recent=False, emphasis=EMPHASIS) == (
        tmp_path / "runs" / "compare" / "twin2K-economy-bechmark-n50-seed0-t0-emphasis"
    )


def test_emphasis_replaces_the_consistency_instruction():
    text = system_message("card one\n", emphasis=EMPHASIS)
    assert text.index(f"强调：{EMPHASIS}") < text.index("card one")
    assert "stay consistent" not in text
    assert "Choose exactly one of the listed options." in text
    plain = system_message("card one\n")
    assert "stay consistent across questions" in plain
    assert "强调：" not in plain


def test_report_names_the_full_recent_store():
    report = _scored_report()
    plain = render_economy_benchmark({key: value for key, value in report.items() if key != "recent"})
    assert "recent：" not in plain
    text = render_economy_benchmark(report)
    assert (
        "- recent：金十免费快讯全量 5 条。每人在人物卡末尾拼接其中一条，按 "
        '`sha256("economy-benchmark-recent:" + person_id) % free_flash_count`。'
    ) in text


def test_evaluation_writes_the_recent_draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "popbench.evaluate.economy_benchmark.evaluate_category_run",
        lambda *_args, **_kwargs: {
            "family": "economic",
            "label": "经济偏好",
            "model": MODEL,
            "n_people": 1,
            "n_items": 1,
            "n_invalid": 0,
            "n_answers": 1,
            "mean_tvd": 0.2,
            "run_dir": "saved",
        },
    )
    monkeypatch.setattr("popbench.evaluate.economy_benchmark.read_jsonl", lambda _path: [])
    monkeypatch.setattr(
        "popbench.evaluate.economy_benchmark.score_full",
        lambda *_args, **_kwargs: {"items": []},
    )
    recent = {
        "draw": DRAW,
        "n_flashes": 5,
        "kinds": [f"flash-{i:02d}" for i in range(5)],
        "by_id": {"p1": "flash-01"},
    }
    report = evaluate_economy_benchmark(
        tmp_path,
        [{"family": "economic", "items": [], "baseline": {}}],
        slug=MODEL,
        seed=SEED,
        temperature=TEMPERATURE,
        recent=recent,
    )
    saved = json.loads((tmp_path / "compare.json").read_text())
    assert saved["model"] == MODEL
    assert saved["seed"] == 0
    assert saved["temperature"] == 0.0
    assert saved["recent"]["n_flashes"] == 5
    assert saved["recent"]["by_id"]["p1"] == "flash-01"
    assert report["recent"]["kinds"] == recent["kinds"]
    assert "全量 5 条" in (tmp_path / "compare.md").read_text()


def test_run_appends_one_full_store_flash_then_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []
    def build(_data_dir, *, n, seed, out_dir):
        assert n == 2
        assert seed == 0
        return _built(out_dir)

    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        build,
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.model_settings_from_env",
        lambda: ModelSettings(
            base_url="https://example.test",
            api_key="test",
            model="ignored",
            temperature=None,
            transport="cr",
        ),
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.responses_cover",
        lambda *_args, **_kwargs: False,
    )

    def answer(records, run_dir, workers, settings, *, panel, seed, family):
        calls.append(
            {
                "records": records,
                "workers": workers,
                "model": settings.model,
                "temperature": settings.temperature,
                "transport": settings.transport,
                "base_url": settings.base_url,
                "thinking": settings.extra_body["thinking"]["type"],
                "panel": panel,
                "seed": seed,
                "family": family,
                "run_dir": run_dir,
            }
        )

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", answer)
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.evaluate_economy_benchmark",
        lambda root, panels, *, slug, seed, temperature, recent: {
            "model": slug,
            "seed": seed,
            "temperature": temperature,
            "mean_tvd": 0.2,
            "families": [{"label": panel["family"], "mean_tvd": 0.2} for panel in panels],
            "recent": recent,
            "out_dir": str(root),
        },
    )

    report = run_economy_benchmark_recent(tmp_path, n=2, workers=3, run_dir=tmp_path / "out")

    assert report["model"] == MODEL
    assert report["seed"] == SEED
    assert report["temperature"] == TEMPERATURE
    assert [call["family"] for call in calls] == ["economic", "experiments", "pricing"]
    assert {call["model"] for call in calls} == {MODEL}
    assert {call["temperature"] for call in calls} == {0.0}
    assert {call["transport"] for call in calls} == {"deepseek"}
    assert {call["base_url"] for call in calls} == {"https://api.deepseek.com"}
    assert {call["thinking"] for call in calls} == {"disabled"}
    assert {call["seed"] for call in calls} == {0}
    assert {call["panel"] for call in calls} == {"atus"}
    assert {call["workers"] for call in calls} == {3}
    seen: set[str] = set()
    for call in calls:
        assert [record["id"] for record in call["records"]] == ["p1", "p2"]
        for record in call["records"]:
            body = record["persona"]["text"].split("## Recent event\n", 1)[1].strip()
            assert body in FLASHES
            seen.add(body)
            assert record["persona"]["text"].count("## Recent event") == 1
    assert len(seen) >= 1
    saved = json.loads((tmp_path / "out" / "recent.json").read_text())
    assert saved["dataset"] == "twin2K-economy-bechmark"
    assert saved["model"] == MODEL
    assert saved["temperature"] == 0.0
    assert saved["seed"] == 0
    assert saved["kinds"] == [f"flash-{index:02d}" for index in range(len(FLASHES))]
    assert saved["people"][0]["text"] in FLASHES
    assert [row["id"] for row in saved["people"]] == ["p1", "p2"]
    assert saved["answered"] is True


def test_changed_store_does_not_reuse_saved_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("must not answer against a different recent splice")

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", forbid)
    root = tmp_path / "out"
    cache = root / MODEL / "economic" / "cache"
    cache.mkdir(parents=True)
    (cache / "old.json").write_text("{}\n")
    (root / "recent.json").write_text(
        json.dumps({"draw": DRAW, "kinds": ["other"], "people": []}) + "\n"
    )
    with pytest.raises(EconomyBenchmarkRecentError, match="recent store no longer matches"):
        run_economy_benchmark_recent(tmp_path, n=2, run_dir=root)
    assert "other" in (root / "recent.json").read_text()


def test_saved_answers_without_recent_json_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("must not answer from a cache with no recorded splice")

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", forbid)
    root = tmp_path / "out"
    cache = root / MODEL / "pricing" / "cache"
    cache.mkdir(parents=True)
    (cache / "old.json").write_text("{}\n")
    with pytest.raises(EconomyBenchmarkRecentError, match="recent store no longer matches"):
        run_economy_benchmark_recent(tmp_path, n=2, run_dir=root)
    assert not (root / "recent.json").exists()


def test_unfinished_recent_file_still_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "out"
    root.mkdir()
    payload = economy_benchmark_recent_payload(_built(tmp_path)["personas"])
    payload["answered"] = False
    (root / "recent.json").write_text(json.dumps(payload) + "\n")
    calls: list[str] = []
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.model_settings_from_env",
        lambda: ModelSettings(
            base_url="https://example.test",
            api_key="test",
            model="ignored",
            temperature=None,
            transport="cr",
        ),
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.responses_cover",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.answer_survey",
        lambda *_args, **kwargs: calls.append(kwargs["family"]),
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.evaluate_economy_benchmark",
        lambda root, panels, *, slug, seed, temperature, recent: {
            "families": [],
            "mean_tvd": None,
            "out_dir": str(root),
            "recent": recent,
        },
    )
    run_economy_benchmark_recent(tmp_path, n=2, run_dir=root)
    assert calls == ["economic", "experiments", "pricing"]
    assert json.loads((root / "recent.json").read_text())["answered"] is True


def test_finished_recent_file_skips_saved_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "out"
    root.mkdir()
    payload = economy_benchmark_recent_payload(_built(tmp_path)["personas"])
    payload["answered"] = True
    (root / "recent.json").write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.responses_cover",
        lambda *_args, **_kwargs: True,
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("a finished recent run must not call the model again")

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", forbid)
    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.model_settings_from_env", forbid)
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.evaluate_economy_benchmark",
        lambda root, panels, *, slug, seed, temperature, recent: {
            "families": [],
            "mean_tvd": None,
            "out_dir": str(root),
            "recent": recent,
        },
    )
    run_economy_benchmark_recent(tmp_path, n=2, run_dir=root)


def test_score_only_rejects_a_different_person_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "out"
    root.mkdir()
    (root / "recent.json").write_text(
        json.dumps(
            {
                "draw": DRAW,
                "kinds": ["flash-saved"],
                "people": [
                    {"id": "p1", "kind": "flash-saved"},
                    {"id": "p2", "kind": "flash-saved"},
                ],
            }
        )
        + "\n"
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("a different --n must not rewrite the run directory")

    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        forbid,
    )
    with pytest.raises(EconomyBenchmarkRecentError, match="pass --n 2"):
        run_economy_benchmark_recent(tmp_path, n=4, run_dir=root, score_only=True)


def test_score_only_reads_the_saved_recent_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("score-only must not call the model")

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.model_settings_from_env", forbid)
    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", forbid)
    seen: dict = {}

    def evaluate(root, panels, *, slug, seed, temperature, recent):
        seen["recent"] = recent
        return {
            "model": slug,
            "seed": seed,
            "temperature": temperature,
            "mean_tvd": None,
            "families": [],
            "recent": recent,
            "out_dir": str(root),
        }

    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.evaluate_economy_benchmark",
        evaluate,
    )
    root = tmp_path / "out"
    root.mkdir()
    (root / "recent.json").write_text(
        json.dumps(
            {
                "draw": DRAW,
                "kinds": ["flash-saved"],
                "people": [
                    {"id": "p1", "kind": "flash-saved", "text": "saved paragraph"},
                    {"id": "p2", "kind": "flash-saved", "text": "saved paragraph"},
                ],
            }
        )
        + "\n"
    )
    run_economy_benchmark_recent(tmp_path, n=2, run_dir=root, score_only=True)
    assert seen["recent"]["kinds"] == ["flash-saved"]
    assert seen["recent"]["by_id"]["p1"] == "flash-saved"


def test_score_only_requires_recent_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )
    with pytest.raises(EconomyBenchmarkRecentError, match="recent.json"):
        run_economy_benchmark_recent(tmp_path, n=2, run_dir=tmp_path / "out", score_only=True)


def test_cli_locks_the_recent_economy_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    seen: dict = {}

    def fake(data_dir, *, n, workers, run_dir, score_only):
        seen.update(
            {
                "data_dir": data_dir,
                "n": n,
                "workers": workers,
                "run_dir": run_dir,
                "score_only": score_only,
            }
        )
        return {
            "mean_tvd": 0.25,
            "families": [{"label": "定价", "mean_tvd": 0.25}],
            "out_dir": str(run_dir),
        }

    monkeypatch.setattr("popbench.cli.run_economy_benchmark_recent", fake)
    code = main(
        [
            "answer-economy-benchmark-recent",
            "--data-dir",
            str(tmp_path),
            "--n",
            "4",
            "--workers",
            "2",
            "--run",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    assert seen == {
        "data_dir": tmp_path,
        "n": 4,
        "workers": 2,
        "run_dir": tmp_path / "out",
        "score_only": False,
    }
    captured = capsys.readouterr()
    assert "定价" in captured.out
    assert "0.25" in captured.out


def test_plain_run_keeps_the_card_and_passes_emphasis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []

    def build(_data_dir, *, n, seed, out_dir):
        assert n == 2
        assert seed == 0
        return _built(out_dir)

    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        build,
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.model_settings_from_env",
        lambda: ModelSettings(
            base_url="https://example.test",
            api_key="test",
            model="ignored",
            temperature=None,
            transport="cr",
        ),
    )
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.responses_cover",
        lambda *_args, **_kwargs: False,
    )

    def answer(records, run_dir, workers, settings, *, panel, seed, family, emphasis):
        calls.append(
            {
                "records": records,
                "model": settings.model,
                "temperature": settings.temperature,
                "transport": settings.transport,
                "base_url": settings.base_url,
                "thinking": settings.extra_body["thinking"]["type"],
                "panel": panel,
                "seed": seed,
                "family": family,
                "emphasis": emphasis,
            }
        )

    seen: dict = {}

    def evaluate(root, panels, *, slug, seed, temperature, recent, emphasis, splice_recent):
        seen.update(
            {
                "slug": slug,
                "seed": seed,
                "temperature": temperature,
                "recent": recent,
                "emphasis": emphasis,
                "splice_recent": splice_recent,
            }
        )
        return {
            "model": slug,
            "seed": seed,
            "temperature": temperature,
            "mean_tvd": 0.2,
            "families": [{"label": panel["family"], "mean_tvd": 0.2} for panel in panels],
            "out_dir": str(root),
        }

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", answer)
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.evaluate_economy_benchmark",
        evaluate,
    )

    report = run_economy_benchmark_recent(
        tmp_path,
        n=2,
        workers=3,
        run_dir=tmp_path / "plain",
        splice_recent=False,
        emphasis=EMPHASIS,
    )

    assert report["model"] == MODEL
    assert seen == {
        "slug": MODEL,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "recent": None,
        "emphasis": EMPHASIS,
        "splice_recent": False,
    }
    assert [call["family"] for call in calls] == ["economic", "experiments", "pricing"]
    assert {call["emphasis"] for call in calls} == {EMPHASIS}
    assert {call["model"] for call in calls} == {MODEL}
    assert {call["temperature"] for call in calls} == {0.0}
    assert {call["transport"] for call in calls} == {"deepseek"}
    assert {call["base_url"] for call in calls} == {"https://api.deepseek.com"}
    assert {call["thinking"] for call in calls} == {"disabled"}
    assert {call["seed"] for call in calls} == {0}
    assert {call["panel"] for call in calls} == {"atus"}
    for call in calls:
        for record in call["records"]:
            assert "## Recent event" not in record["persona"]["text"]
            assert record["persona"]["text"] in {"card one\n", "card two\n"}
    note = json.loads((tmp_path / "plain" / "run.json").read_text())
    assert note["splice_recent"] is False
    assert note["emphasis"] == EMPHASIS
    assert note["model"] == MODEL
    assert note["temperature"] == 0.0
    assert note["seed"] == 0
    assert note["transport"] == "deepseek"
    assert note["base_url"] == "https://api.deepseek.com"
    assert note["answered"] is True
    assert not (tmp_path / "plain" / "recent.json").exists()


def test_plain_run_refuses_a_directory_that_already_spliced_recent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "popbench.simulate.economy_benchmark_recent.build_economy_benchmark_panels",
        lambda *_args, **_kwargs: _built(tmp_path),
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("must not answer a spliced directory without recent")

    monkeypatch.setattr("popbench.simulate.economy_benchmark_recent.answer_survey", forbid)
    root = tmp_path / "out"
    root.mkdir()
    (root / "recent.json").write_text("{}\n")
    with pytest.raises(EconomyBenchmarkRecentError, match="splices recent flashes"):
        run_economy_benchmark_recent(
            tmp_path,
            n=2,
            run_dir=root,
            splice_recent=False,
            emphasis=EMPHASIS,
        )


def test_report_names_an_unspliced_emphasis():
    report = _scored_report()
    report.pop("recent")
    report["splice_recent"] = False
    report["emphasis"] = EMPHASIS
    text = render_economy_benchmark(report)
    assert "- recent：不拼接。人物卡保持原样。" in text
    assert f"- 强调：{EMPHASIS}" in text


def test_cli_can_skip_recent_and_set_emphasis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict = {}

    def fake(data_dir, *, n, workers, run_dir, score_only, splice_recent, emphasis):
        seen.update(
            {
                "data_dir": data_dir,
                "n": n,
                "workers": workers,
                "run_dir": run_dir,
                "score_only": score_only,
                "splice_recent": splice_recent,
                "emphasis": emphasis,
            }
        )
        return {
            "mean_tvd": 0.25,
            "families": [{"label": "定价", "mean_tvd": 0.25}],
            "out_dir": str(run_dir),
        }

    monkeypatch.setattr("popbench.cli.run_economy_benchmark_recent", fake)
    code = main(
        [
            "answer-economy-benchmark-recent",
            "--data-dir",
            str(tmp_path),
            "--no-recent",
            "--emphasis",
            EMPHASIS,
            "--run",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    assert seen["splice_recent"] is False
    assert seen["emphasis"] == EMPHASIS
    assert seen["score_only"] is False
    assert seen["n"] == 50
    assert seen["run_dir"] == tmp_path / "out"


def test_cli_rejects_model_temperature_and_seed_overrides():
    for extra in (
        ["--model", "gpt-5.5"],
        ["--temperature", "0.8"],
        ["--seed", "1"],
    ):
        with pytest.raises(SystemExit) as caught:
            main(["answer-economy-benchmark-recent", *extra])
        assert caught.value.code == 2


def _built(root: Path) -> dict:
    personas = [
        {"id": "p1", "text": "card one\n", "demographics": {"age_band": "18-29", "sex": "Female"}},
        {"id": "p2", "text": "card two\n", "demographics": {"age_band": "65+", "sex": "Male"}},
    ]
    panels = []
    for family in ("economic", "experiments", "pricing"):
        records = [
            _record(person["id"], person["demographics"]["age_band"], person["text"])
            for person in personas
        ]
        panels.append(
            {
                "family": family,
                "label": family,
                "items": [{"item_id": "QID229"}],
                "baseline": {},
                "records": records,
            }
        )
    return {"personas": personas, "panels": panels, "n": 2, "seed": 0, "out_dir": root}


def _scored_report() -> dict:
    return {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "n_people": 2,
        "n_items": 1,
        "n_invalid": 0,
        "n_answers": 2,
        "mean_tvd": 0.2,
        "families": [
            {
                "label": "经济偏好",
                "n_items": 1,
                "mean_tvd": 0.2,
                "n_invalid": 0,
                "n_answers": 2,
            }
        ],
        "farthest": [],
        "recent": {
            "draw": DRAW,
            "n_flashes": 5,
            "kinds": [f"flash-{index:02d}" for index in range(5)],
            "by_id": {"p1": "flash-01"},
        },
    }
