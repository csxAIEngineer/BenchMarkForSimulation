"""Temperature-0 recent contrast from saved answers. No model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from popbench.simulate.recent_store import upsert_flashes
from popbench.dao.io import write_jsonl
from popbench.evaluate.economy_recent import (
    render_t0_economy_recent,
    score_t0_economy_recent,
    write_t0_economy_recent,
)
from popbench.simulate.recent import (
    education_recent_kind,
    education_recent_payload,
)

MODEL = "deepseek-v4-pro"


def test_recent_minus_plain_splits_by_event_kind(tmp_path: Path):
    by_kind = _ids_covering_kinds()
    first_kind, second_kind, third_kind = list(by_kind)
    first = by_kind[first_kind]
    second = by_kind[second_kind]
    third = by_kind[third_kind]
    personas = [_persona(person_id) for person_id in (first, second, third)]
    economic_items = [_choice("Q1", "economic", 1), _numeric("N1", "economic", 2)]
    pricing_items = [_choice("P1", "pricing", 1)]
    families = {
        "economic": {
            "items": economic_items,
            "baseline": {
                "n_respondents": 8,
                "items": [_gold_choice("Q1", "economic"), _gold_numeric("N1", "economic", 10.0)],
            },
        },
        "pricing": {
            "items": pricing_items,
            "baseline": {"n_respondents": 8, "items": [_gold_choice("P1", "pricing")]},
        },
    }
    # Choice codes: plain all 1. Recent is 2, 1, 2. Numeric plain 10, 12, 14; recent 10, 12, 20.
    control_answers = {
        "economic": {first: [1, 10], second: [1, 12], third: [1, 14]},
        "pricing": {first: [1], second: [1], third: [1]},
    }
    treatment_answers = {
        "economic": {first: [2, 10], second: [1, 12], third: [2, 20]},
        "pricing": {first: [1], second: [1], third: [1]},
    }
    control = tmp_path / "plain"
    treatment = tmp_path / "recent"
    _write_arm(control, personas, families, control_answers, recent=False)
    _write_arm(treatment, personas, families, treatment_answers, recent=True)

    report = score_t0_economy_recent(
        control,
        treatment,
        families=("economic", "pricing"),
        n_people=3,
    )
    economic = report["families"][0]
    pricing = report["families"][1]
    assert economic["family"] == "economic"
    assert economic["distances"]["control"]["mean_tvd"] == pytest.approx(0.5)
    assert economic["distances"]["treatment"]["mean_tvd"] == pytest.approx(1 / 6)
    assert economic["distances"]["mean_tvd_diff"] == pytest.approx(1 / 6 - 0.5)
    assert economic["distances"]["control"]["mean_mse"] == pytest.approx(0.25)
    assert economic["distances"]["treatment"]["mean_mse"] == pytest.approx(1 / 36)
    assert economic["distances"]["mean_mse_diff"] == pytest.approx(1 / 36 - 0.25)
    assert economic["distances"]["control"]["mean_mae"] == pytest.approx(2)
    assert economic["distances"]["treatment"]["mean_mae"] == pytest.approx(4)
    assert economic["distances"]["mean_mae_diff"] == pytest.approx(2)
    assert economic["paired"]["change_rate"] == pytest.approx(0.5)
    assert economic["paired"]["n_changed"] == 3
    assert economic["paired"]["n_comparable"] == 6
    assert economic["paired"]["choice_mean_diff"] == pytest.approx(2 / 3)
    assert economic["paired"]["numeric_mean_diff"] == pytest.approx(2)

    first_row = _kind(economic, first_kind)
    second_row = _kind(economic, second_kind)
    third_row = _kind(economic, third_kind)
    assert first_row["n_people"] == 1
    assert first_row["distances"]["mean_tvd_diff"] == pytest.approx(0.0)
    assert first_row["distances"]["mean_mae_diff"] == pytest.approx(0.0)
    assert first_row["paired"]["choice_mean_diff"] == pytest.approx(1)
    assert first_row["paired"]["numeric_mean_diff"] == pytest.approx(0)
    assert first_row["paired"]["change_rate"] == pytest.approx(0.5)
    assert second_row["distances"]["mean_mae_diff"] == pytest.approx(0.0)
    assert second_row["paired"]["change_rate"] == pytest.approx(0.0)
    assert second_row["paired"]["choice_mean_diff"] == pytest.approx(0)
    assert third_row["distances"]["mean_mae_diff"] == pytest.approx(6)
    assert third_row["paired"]["choice_mean_diff"] == pytest.approx(1)
    assert third_row["paired"]["numeric_mean_diff"] == pytest.approx(6)
    assert third_row["paired"]["change_rate"] == pytest.approx(1)

    assert pricing["distances"]["mean_tvd_diff"] == pytest.approx(0.0)
    assert pricing["paired"]["change_rate"] == pytest.approx(0.0)
    assert pricing["paired"]["choice_mean_diff"] == pytest.approx(0.0)
    assert pricing["paired"]["numeric_mean_diff"] is None
    assert pricing["distances"]["mean_mae_diff"] is None

    text = render_t0_economy_recent(report)
    assert "经济偏好" in text
    assert "定价" in text
    assert "不做跨种子的显著性检验" in text
    assert first_kind in text
    assert "-0.3333" in text
    assert "-0.2222" in text
    assert "MSE" in text
    assert "改动比例" not in text
    assert "绝对差" not in text
    assert "选项均值差" not in text
    written = write_t0_economy_recent(report, tmp_path / "reports" / "t0-economy-recent")
    saved = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert saved["families"][0]["distances"]["mean_tvd_diff"] == pytest.approx(1 / 6 - 0.5)
    assert Path(written["markdown"]).name == "summary.md"
    with pytest.raises(ValueError, match="expected 50"):
        score_t0_economy_recent(control, treatment, families=("economic", "pricing"), n_people=50)


def test_choice_mean_diff_weights_items_equally(tmp_path: Path):
    person_ids = list(_ids_covering_kinds().values())
    first, second, third = person_ids
    personas = [_persona(person_id) for person_id in person_ids]
    items = [_choice("Q1", "economic", 1), _choice("Q2", "economic", 2)]
    families = {
        "economic": {
            "items": items,
            "baseline": {
                "n_respondents": 8,
                "items": [_gold_choice("Q1", "economic"), _gold_choice("Q2", "economic")],
            },
        }
    }
    control_answers = {"economic": {person_id: [1, 1] for person_id in person_ids}}
    treatment_answers = {
        "economic": {
            first: [2, 1],
            second: [2, None],
            third: [2, None],
        }
    }
    control = tmp_path / "plain"
    treatment = tmp_path / "recent"
    _write_arm(control, personas, families, control_answers, recent=False)
    _write_arm(treatment, personas, families, treatment_answers, recent=True)
    report = score_t0_economy_recent(control, treatment, families=("economic",))
    paired = report["families"][0]["paired"]
    # Item means are 1 and 0. A pooled pair mean would be 0.75.
    assert paired["choice_mean_diff"] == pytest.approx(0.5)
    assert paired["n_choice_paired"] == 2
    assert paired["change_rate"] == pytest.approx(0.75)
    assert paired["n_changed"] == 3
    assert paired["n_comparable"] == 4


def test_control_arm_rejects_recent_json(tmp_path: Path):
    control, treatment = _tiny_pair(tmp_path)
    (control / "recent.json").write_text((treatment / "recent.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="should not have"):
        score_t0_economy_recent(control, treatment, families=("economic",))


def test_scoring_keeps_the_saved_flashes_when_the_store_changes(tmp_path: Path):
    control, treatment = _tiny_pair(tmp_path)
    saved = json.loads((treatment / "recent.json").read_text(encoding="utf-8"))
    upsert_flashes(
        [
            {
                "id": "later",
                "time": "2026-10-19 23:00:00",
                "channel": [1],
                "important": 0,
                "type": 0,
                "content": "A later headline.",
                "vip_locked": False,
                "vip_title": "",
                "vip_level": 0,
                "site_tags": [],
                "topics": ["股指市场"],
            }
        ]
    )
    report = score_t0_economy_recent(control, treatment, families=("economic",))
    assert report["recent"]["kinds"] == saved["kinds"]
    assert "later" not in report["recent"]["kinds"]


def test_treatment_arm_requires_recent_json(tmp_path: Path):
    control, treatment = _tiny_pair(tmp_path)
    (treatment / "recent.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        score_t0_economy_recent(control, treatment, families=("economic",))


def test_temperature_must_be_zero(tmp_path: Path):
    control, treatment = _tiny_pair(tmp_path, treatment_temperature=0.8)
    with pytest.raises(ValueError, match="temperature"):
        score_t0_economy_recent(control, treatment, families=("economic",))


def test_reasoning_effort_stays_off_the_request(tmp_path: Path):
    control, treatment = _tiny_pair(tmp_path, reasoning_effort="none")
    with pytest.raises(ValueError, match="reasoning_effort"):
        score_t0_economy_recent(control, treatment, families=("economic",))


def test_person_ids_must_match(tmp_path: Path):
    person_ids = list(_ids_covering_kinds().values())
    control, treatment = _tiny_pair(tmp_path, person_ids=person_ids, skip_treatment_ids=(person_ids[0],))
    with pytest.raises(ValueError, match="person ids differ"):
        score_t0_economy_recent(control, treatment, families=("economic",))


def test_item_ids_must_match(tmp_path: Path):
    control, treatment = _tiny_pair(tmp_path)
    path = treatment / "families" / "economic" / "items.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    items[0]["options"] = ["A", "C"]
    path.write_text(json.dumps(items) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="items differ"):
        score_t0_economy_recent(control, treatment, families=("economic",))


def _tiny_pair(
    tmp_path: Path,
    *,
    person_ids: list[str] | None = None,
    treatment_temperature: float = 0.0,
    reasoning_effort: str | None = None,
    skip_treatment_ids: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    if person_ids is None:
        person_ids = list(_ids_covering_kinds().values())
    personas = [_persona(person_id) for person_id in person_ids]
    items = [_choice("Q1", "economic", 1)]
    families = {
        "economic": {
            "items": items,
            "baseline": {"n_respondents": 4, "items": [_gold_choice("Q1", "economic")]},
        }
    }
    answers = {"economic": {person_id: [1] for person_id in person_ids}}
    control = tmp_path / "plain"
    treatment = tmp_path / "recent"
    _write_arm(control, personas, families, answers, recent=False)
    _write_arm(
        treatment,
        personas,
        families,
        answers,
        recent=True,
        temperature=treatment_temperature,
        reasoning_effort=reasoning_effort,
        skip_response_ids=skip_treatment_ids,
    )
    return control, treatment


def _ids_covering_kinds() -> dict[str, str]:
    found: dict[str, str] = {}
    number = 0
    while len(found) < 3:
        person_id = f"visitor-atus-{number}"
        found.setdefault(education_recent_kind(person_id), person_id)
        number += 1
    return found


def _persona(person_id: str) -> dict:
    return {
        "id": person_id,
        "text": f"card {person_id}",
        "demographics": {"age": 40, "age_band": "30-49", "sex": "Female"},
    }


def _choice(item_id: str, family: str, turn: int) -> dict:
    return {
        "item_id": item_id,
        "kind": "choice",
        "family": family,
        "block": "Economic preferences",
        "instrument": "Risk",
        "prompt": f"Question {item_id}",
        "options": ["A", "B"],
        "turn": turn,
        "encoding": "single",
        "column": item_id,
    }


def _numeric(item_id: str, family: str, turn: int) -> dict:
    return {
        "item_id": item_id,
        "kind": "numeric",
        "family": family,
        "block": "Economic preferences",
        "instrument": "Risk",
        "prompt": f"Question {item_id}",
        "options": [],
        "turn": turn,
        "encoding": "number",
        "column": item_id,
    }


def _gold_choice(item_id: str, family: str) -> dict:
    return {
        "item_id": item_id,
        "kind": "choice",
        "family": family,
        "n": 8,
        "shares": [0.5, 0.5],
        "mean_code": 1.5,
    }


def _gold_numeric(item_id: str, family: str, mean: float) -> dict:
    return {
        "item_id": item_id,
        "kind": "numeric",
        "family": family,
        "n": 8,
        "mean": mean,
    }


def _write_arm(
    root: Path,
    personas: list[dict],
    families: dict[str, dict],
    answers: dict[str, dict[str, list[float | None]]],
    *,
    recent: bool,
    temperature: float = 0.0,
    seed: int = 0,
    reasoning_effort: str | None = None,
    skip_response_ids: tuple[str, ...] = (),
    model: str = MODEL,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "personas.jsonl", personas)
    if recent:
        payload = education_recent_payload(personas)
        (root / "recent.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    skipped = set(skip_response_ids)
    for family, spec in families.items():
        folder = root / "families" / family
        folder.mkdir(parents=True, exist_ok=True)
        items = spec["items"]
        (folder / "items.json").write_text(json.dumps(items, ensure_ascii=False) + "\n", encoding="utf-8")
        (folder / "human_baseline.json").write_text(json.dumps(spec["baseline"]) + "\n", encoding="utf-8")
        run_dir = root / model / family
        run_dir.mkdir(parents=True, exist_ok=True)
        responses = []
        for persona in personas:
            person_id = persona["id"]
            if person_id in skipped:
                continue
            values = answers[family][person_id]
            responses.append(
                {
                    "id": person_id,
                    "answers": [
                        _stored(item["item_id"], value) for item, value in zip(items, values, strict=True)
                    ],
                }
            )
        write_jsonl(run_dir / "responses.jsonl", responses)
        extra: dict[str, str] = {}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        meta = {
            "model": model,
            "panel": "atus",
            "n_people": len(personas),
            "n_turns": len(items),
            "seed": seed,
            "family": family,
            "request": {
                "transport": "cr",
                "temperature": temperature,
                "max_tokens": None,
                "extra_body": extra,
            },
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _stored(item_id: str, value: float | None) -> dict:
    if value is None:
        return {"item_id": item_id, "value": None, "ok": False, "raw": ""}
    return {"item_id": item_id, "value": value, "ok": True, "raw": "{}"}


def _kind(family: dict, kind: str) -> dict:
    return next(row for row in family["by_kind"] if row["kind"] == kind)
