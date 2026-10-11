"""Distribution and age tests for persona answers on personality items."""

from __future__ import annotations

import numpy as np
import pandas as pd

from popbench.evaluate.significance import (
    _score_item,
    build_personality_age_report,
    holm,
    kruskal_p,
)


def test_holm_keeps_the_running_maximum():
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted[0] == 0.03
    assert adjusted[1] == 0.06
    assert adjusted[2] == 0.06


def test_kruskal_is_one_when_groups_match_and_small_when_they_do_not():
    same = [np.array([1, 2, 3, 2]), np.array([1, 2, 3, 2]), np.array([1, 2, 3, 2])]
    assert kruskal_p(same) == 1.0
    split = [np.full(12, code) for code in (1, 2, 3, 4)]
    assert kruskal_p(split) < 1e-6


def test_matching_answers_are_not_a_significant_departure():
    report = _report(sim_code=_human_code, seed=0)
    assert report["overall"]["mean_tvd"] == 0
    assert report["overall"]["n_gof_holm"] == 0
    assert report["age_effects"]["gradient_r"] == 1
    assert report["age_effects"]["gradient_p"] < 0.05
    assert any("不能拒绝" in line for line in report["verdict"])


def test_missing_human_answer_drops_that_age_band_row():
    human = pd.DataFrame(
        {
            "pid": ["a", "b", "c"],
            "age_band": ["18-29", "18-29", "65+"],
            "Q1": [1, None, 5],
        }
    )
    item = {
        "item_id": "Q1",
        "kind": "choice",
        "instrument": "Anxiety",
        "prompt": "Item",
        "options": ["1", "2", "3", "4", "5"],
    }
    people = [
        {"id": "s1", "age_band": band, "answers": {"Q1": 1}}
        for band in ("18-29", "30-49", "50-64", "65+")
    ]
    row = _score_item(item, people, human, np.random.default_rng(0), 99)
    assert row["n_human"] == 2
    assert row["n_human_18-29"] == 1
    assert row["n_human_65+"] == 1
    assert row["tvd"] is not None


def test_shifted_answers_differ_from_the_human_shares():
    report = _report(sim_code=lambda *_args: 5, seed=1)
    assert report["overall"]["n_gof_holm"] == report["overall"]["n_tested"]
    assert report["overall"]["mean_tvd"] > 0.5
    text = "\n".join(report["verdict"])
    assert "统计学意义" in text


def _report(sim_code, seed: int) -> dict:
    bands = ("18-29", "30-49", "50-64", "65+")
    items = []
    human_rows = []
    responses = []
    personas = []
    for band_index, band in enumerate(bands):
        for person in range(6):
            human_rows.append({"pid": f"h-{band}-{person}", "age_band": band})
        for person in range(6):
            person_id = f"s-{band}-{person}"
            personas.append({"id": person_id, "demographics": {"age_band": band, "sex": "Female"}})
            responses.append({"id": person_id, "answers": []})
    for item_index in range(6):
        item_id = f"Q{item_index}"
        items.append(
            {
                "item_id": item_id,
                "kind": "choice",
                "instrument": "Anxiety",
                "prompt": f"Item {item_index}",
                "options": ["1", "2", "3", "4", "5"],
            }
        )
        for row in human_rows:
            row[item_id] = _human_code(bands.index(row["age_band"]), item_index)
        for response, persona in zip(responses, personas):
            band_index = bands.index(persona["demographics"]["age_band"])
            response["answers"].append(
                {
                    "item_id": item_id,
                    "value": sim_code(band_index, item_index),
                    "ok": True,
                }
            )
    human = pd.DataFrame(human_rows)
    return build_personality_age_report(
        responses,
        personas,
        items,
        human,
        model="gpt-5.5",
        seed=seed,
        n_per_age_band=6,
        reps=199,
        mean_reps=499,
        rng=np.random.default_rng(seed),
    )


def _human_code(band_index: int, item_index: int) -> int:
    """Younger bands sit lower, and later items stretch that gap."""
    return min(5, 1 + min(band_index, 1) + (item_index if band_index == 3 else 0))
