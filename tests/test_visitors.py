from pathlib import Path

import json
import pandas as pd
import pytest

from popbench.dao.visitors import (
    VisitorError,
    build_visitors,
    cache_is_complete,
    fetch_visitors,
)
from popbench.simulate.visitors import (
    VISIT_TURNS,
    system_message,
    visit_user_message,
    visitor_panel_path,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_visitor_cache_is_present():
    assert cache_is_complete(DATA)


def test_visit_instrument_has_eight_atus_aligned_turns():
    assert len(VISIT_TURNS) == 8
    ids = [turn.item_id for turn in VISIT_TURNS]
    assert "visit_work" in ids
    assert "visit_leisure" in ids
    text = visit_user_message(VISIT_TURNS[0])
    assert "Visit question 1 of 8" in text
    assert "Options:" in text


def test_visitor_system_prompt_embeds_persona():
    prompt = system_message("Age: 40\nSex: Female\nState: Ohio")
    assert "daily life" in prompt
    assert "Age: 40" in prompt
    assert "ATUS-style" in prompt


def test_build_visitors_writes_panel_and_baselines(tmp_path):
    src = DATA / "visitors"
    dest = tmp_path / "data" / "visitors"
    dest.mkdir(parents=True)
    for name in ("acs", "atus", "census"):
        (dest / name).symlink_to(src / name, target_is_directory=True)

    out = build_visitors(tmp_path / "data", n=8, seed=1)
    assert (out / "visitors_n8.jsonl").is_file()
    assert (out / "routine_baseline.json").is_file()
    assert (out / "vacsim_margins.json").is_file()
    assert (out / "manifest.json").is_file()

    lines = (out / "visitors_n8.jsonl").read_text().strip().splitlines()
    assert len(lines) == 8
    row = json.loads(lines[0])
    assert row["source"] == "acs+atus+census"
    assert row["persona"]["age"] >= 18
    assert row["routine"]
    assert row["state_population"] > 0
    assert visitor_panel_path(tmp_path / "data", n=8).name == "visitors_n8.jsonl"


def test_fetch_visitors_is_idempotent_when_cache_exists():
    paths = fetch_visitors(DATA)
    assert paths["acs"].is_file()
    assert paths["census"].is_file()
    assert paths["atus"].is_dir()
    assert "choosing_wisely" not in paths
    assert (DATA / "visitors" / "acs" / "manifest.json").is_file()


def test_census_state_rows_are_usable():
    frame = pd.read_csv(DATA / "visitors" / "census" / "NST-EST2023-ALLDATA.csv")
    states = frame.loc[frame["SUMLEV"] == 40]
    assert len(states) >= 50
    assert int(states["POPESTIMATE2023"].sum()) > 300_000_000


def test_build_visitors_rejects_empty_n(tmp_path):
    with pytest.raises(VisitorError):
        build_visitors(tmp_path, n=0)
