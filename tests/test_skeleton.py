import os
from pathlib import Path

import pytest

from popbench.config import ConfigError, load_config
from popbench.dao.schema import Item
from popbench.evaluate.report import render_markdown
from popbench.simulate.client import configured_model, model_from_run, model_settings_from_env, response_cache_key
import popbench.simulate.client as simulate

ROOT = Path(__file__).resolve().parents[1]


def test_v1_config_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs" / "v1.yaml")
    assert config.suite == "both"
    assert config.persona_condition == "full_twin2k"
    assert config.n_people == 50
    assert config.effective_n_people == 50
    assert config.full_n is False
    assert config.item_cap is None
    assert config.seed == 0
    assert config.data_dir == tmp_path / "data"


def test_full_n_uses_the_published_sample(tmp_path):
    path = tmp_path / "full.yaml"
    path.write_text(
        "suite: decision\n"
        "persona_condition: demographics_only\n"
        "n_people: 50\n"
        "full_n: true\n"
        "seed: 1\n"
        "data_dir: data\n"
    )
    config = load_config(path)
    assert config.effective_n_people == 2058


def test_twin2k_sample_cannot_exceed_the_panel(tmp_path):
    path = tmp_path / "too-many.yaml"
    path.write_text("persona_condition: full_twin2k\nn_people: 3000\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_config_field_is_rejected(tmp_path):
    path = tmp_path / "extra.yaml"
    path.write_text("n_people: 10\nmodel: gpt\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_item_shares_must_match_options():
    with pytest.raises(ValueError):
        Item(
            id="q",
            prompt="Pick one",
            options=("a", "b"),
            kind="multiple_choice",
            wave="4",
            suite="decision",
            human_shares=(1.0,),
        )


def test_report_states_sample_and_ceiling():
    text = render_markdown(
        {
            "n_people": 50,
            "persona_condition": "full_twin2k",
            "suite": "both",
            "test_retest_ceiling": 0.8,
            "distance_to_ceiling": 0.1,
        }
    )
    assert "Sample size: 50" in text
    assert "full_twin2k" in text
    assert "not evidence" in text
    assert "0.8" in text


def test_deepseek_settings_come_from_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=test-key\n"
        "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
        "DEEPSEEK_MODEL=deepseek-flash\n"
    )
    monkeypatch.setattr(simulate, "_default_env_path", lambda: env_file)
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = model_settings_from_env()
    assert settings.api_key == "test-key"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        os.environ.pop(name, None)


def test_configured_model_does_not_require_an_api_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_MODEL=deepseek-flash\n")
    monkeypatch.setattr(simulate, "_default_env_path", lambda: env_file)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    assert configured_model() == "deepseek-flash"
    assert model_from_run({}) == "deepseek-flash"
    assert model_from_run({"model": "unknown"}) == "deepseek-flash"
    assert model_from_run({"model": "other-model"}) == "other-model"
    os.environ.pop("DEEPSEEK_MODEL", None)


def test_missing_deepseek_key_is_reported(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_BASE_URL=https://api.deepseek.com\n")
    monkeypatch.setattr(simulate, "_default_env_path", lambda: env_file)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        model_settings_from_env()


def test_response_cache_key_depends_on_model_persona_and_item():
    first = response_cache_key("model-a", "p1", "q1")
    assert first == response_cache_key("model-a", "p1", "q1")
    assert first != response_cache_key("model-b", "p1", "q1")
    assert first != response_cache_key("model-a", "p2", "q1")
    assert first != response_cache_key("model-a", "p1", "q2")
