"""Provider settings and visitor-model ranking. No network calls."""

import os
from pathlib import Path

import pytest

from popbench.cli import main
from popbench.evaluate.compare import build_visitor_comparison, render_visitor_comparison
from popbench.simulate.client import (
    ModelError,
    SettingsError,
    assistant_text,
    canonical_compare_model,
    chat,
    chat_with_retry,
    completion_payload,
    model_settings_for,
    model_settings_from_env,
    parse_choice,
    run_slug,
)
from cr_api.client import BASE_URL as CR_BASE_URL
from cr_api.client import CRError, ChatResult
import popbench.simulate.client as simulate

KEY_NAMES = (
    "CR_API_KEY",
    "CR_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
)


def _isolate_env(tmp_path, monkeypatch, text: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(text)
    monkeypatch.setattr(simulate, "_default_env_path", lambda: env_file)
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def _summary(model: str, correlation: float, share_mse: float, item_corr: float) -> dict:
    return {
        "model": model,
        "run_dir": f"runs/{model}",
        "n_people": 50,
        "seed": 0,
        "baseline": {"source": "ATUS", "n_respondents": 10},
        "request": {"temperature": 0, "max_tokens": 128, "token_field": "max_tokens", "extra_body": {}},
        "behavioral_fit": {
            "distribution_correlation": correlation,
            "share_mse": share_mse,
            "mean_code_mse": share_mse,
            "mean_code_correlation": correlation,
            "response_amplitude_coverage": 1.0,
            "response_amplitude_ratio": 1.0,
            "per_item": [
                {"item_id": "visit_work", "distribution_correlation": item_corr},
            ],
        },
    }


def test_compare_models_share_the_cr_api(tmp_path, monkeypatch):
    _isolate_env(
        tmp_path,
        monkeypatch,
        "\n".join(
            [
                "CR_API_KEY=cr-key",
                "DEEPSEEK_API_KEY=deepseek-key",
                "DEEPSEEK_MODEL=deepseek-flash",
                "",
            ]
        ),
    )
    gpt = model_settings_for("GPT5.5")
    glm = model_settings_for("glm-5.3")
    deepseek = model_settings_for("deepseek-v4-pro")
    assert (gpt.model, glm.model, deepseek.model) == (
        "gpt-5.5",
        "glm-5.3",
        "deepseek-v4-pro",
    )
    assert gpt.api_key == glm.api_key == deepseek.api_key == "cr-key"
    assert gpt.transport == glm.transport == deepseek.transport == "cr"
    assert gpt.base_url == glm.base_url == deepseek.base_url == CR_BASE_URL
    assert run_slug("GPT5.5") == "gpt-5.5"

    messages = [{"role": "user", "content": "hi"}]
    for settings in (gpt, glm, deepseek):
        assert settings.max_tokens is None
        assert settings.temperature is None
        assert settings.extra_body == {}

    legacy = completion_payload(model_settings_from_env(), messages)
    assert legacy["model"] == "deepseek-flash"
    assert legacy["thinking"] == {"type": "disabled"}
    for name in KEY_NAMES:
        os.environ.pop(name, None)


def test_missing_provider_key_names_the_variable(tmp_path, monkeypatch):
    _isolate_env(tmp_path, monkeypatch, "DEEPSEEK_API_KEY=deepseek-key\n")
    with pytest.raises(SettingsError, match="CR_API_KEY"):
        model_settings_for("gpt-5.5")
    assert canonical_compare_model("deepseek-flash") is None
    other = model_settings_for("deepseek-flash")
    assert other.model == "deepseek-flash"
    assert other.base_url == "https://api.deepseek.com"


def test_parse_choice_ignores_a_think_block():
    raw = '<think>draft</think>\n{"answer": "A little", "rationale": "ok"}'
    parsed = parse_choice(raw, ("Almost never / very little", "A little"))
    assert parsed.answer == "A little"


def test_assistant_text_joins_content_parts():
    text = assistant_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": '{"answer": "A little"}'}],
                    }
                }
            ]
        }
    )
    assert "A little" in text


def test_ranking_prefers_higher_correlation_then_lower_share_mse():
    report = build_visitor_comparison(
        [
            _summary("gpt-5.5", 0.2, 0.01, 0.2),
            _summary("glm-5.3", 0.9, 0.05, 0.9),
            _summary("deepseek-v4-pro", 0.9, 0.01, 0.4),
        ]
    )
    assert [row["model"] for row in report["ranked"]] == [
        "deepseek-v4-pro",
        "glm-5.3",
        "gpt-5.5",
    ]
    text = render_visitor_comparison(report)
    assert text.index("deepseek-v4-pro") < text.index("glm-5.3")
    assert "| `visit_work` | 0.400 | 0.900 | 0.200 |" in text


def test_compare_chat_uses_cr_client(tmp_path, monkeypatch):
    _isolate_env(tmp_path, monkeypatch, "CR_API_KEY=cr-key\n")
    settings = model_settings_for("glm-5.3")
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, api_key=None, env_path=None):
            seen["api_key"] = api_key

        def chat(self, messages, model="deepseek-v4-pro", max_tokens=None, stream=False, timeout=300, response_format=None):
            seen["model"] = model
            seen["max_tokens"] = max_tokens
            seen["messages"] = messages
            return ChatResult('{"answer": "A little"}', "thought", model, {})

    monkeypatch.setattr(simulate, "CRClient", FakeClient)
    assert chat(settings, [{"role": "user", "content": "hi"}]) == '{"answer": "A little"}'
    assert seen["api_key"] == "cr-key"
    assert seen["model"] == "glm-5.3"
    assert seen["max_tokens"] is None


def test_cr_client_rejects_a_bad_request_without_retry(tmp_path, monkeypatch):
    _isolate_env(tmp_path, monkeypatch, "CR_API_KEY=cr-key\n")
    settings = model_settings_for("gpt-5.5")

    class FakeClient:
        def __init__(self, api_key=None, env_path=None):
            pass

        def chat(self, messages, model="deepseek-v4-pro", max_tokens=None, stream=False, timeout=300, response_format=None):
            raise CRError("CR API 返回 400：bad", status=400, body="bad")

    monkeypatch.setattr(simulate, "CRClient", FakeClient)
    with pytest.raises(ModelError, match="400") as caught:
        chat_with_retry(settings, [{"role": "user", "content": "hi"}], attempts=3)
    assert caught.value.retryable is False


def test_compare_cli_rejects_a_single_model(capsys):
    assert main(["compare", "--models", "gpt-5.5"]) == 1
    assert "at least two" in capsys.readouterr().err


def test_compare_cli_reports_the_shared_cr_key(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=deepseek-key\n")
    monkeypatch.setattr(simulate, "_default_env_path", lambda: env_file)
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["compare", "--models", "gpt-5.5", "glm-5.3", "--n", "1"]) == 1
    err = capsys.readouterr().err
    assert err.count("CR_API_KEY") == 1
