"""Communication layer: mock provider, retry, cache, and JSON parsing."""

import json

import pytest

from popbench.comms.cache import cache_key
from popbench.comms.client import LLMClient
from popbench.comms.errors import CommsError, CommsTransportError
from popbench.comms.parse import parse_json_object
from popbench.comms.providers.cr import CRProvider
from popbench.comms.providers.mock import MockProvider
from popbench.comms.providers.openai_compat import OpenAICompatProvider
from popbench.comms.retry import call_with_retry
from popbench.models import build_prompt
from popbench.models._prompt import build_package
from popbench.schema import PromptPackage, Question, Subject, Case
from popbench.datasets.twin2k_500.loader import bundle_from_records


def _package(mode="answer"):
    bundle = bundle_from_records(
        [
            {
                "id": "7",
                "demographic_text": "Age: 34",
                "persona_text": "Full",
                "persona_summary": "Sum",
                "questions": [
                    {"id": "Q1", "prompt": "Pick one", "options": ["red", "blue"], "kind": "multiple_choice"}
                ],
            }
        ]
    )
    if mode == "answer":
        return build_prompt("atus_expanded", bundle.cases[0], seed=0)
    subject = Subject(
        id="ESS:population",
        dataset="simbench",
        unit="group",
        demographics={},
        group_prompt="You are an ESS respondent.",
    )
    case = Case(
        dataset="simbench",
        subject=subject,
        question=Question(id="trust", prompt="Trust most people?", options=("A", "B"), kind="distribution"),
    )
    return build_package(
        case,
        method="distribution",
        persona="You are an ESS respondent.",
        response_mode="distribution",
    )


def test_parse_json_object_accepts_fences_and_prose():
    assert parse_json_object('```json\n{"answer": "red"}\n```') == {"answer": "red"}
    assert parse_json_object('Sure: {"answer": "red"} thanks') == {"answer": "red"}
    with pytest.raises(Exception):
        parse_json_object("no json here")


def test_retry_then_succeeds_without_sleeping_the_suite():
    calls = {"n": 0}

    def action():
        calls["n"] += 1
        if calls["n"] < 3:
            raise CommsTransportError("down")
        return "ok"

    slept: list[float] = []
    assert call_with_retry(action, attempts=3, sleep=slept.append) == "ok"
    assert slept == [0.2, 0.4]


def test_mock_client_caches_and_parses(tmp_path):
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", cache_dir=tmp_path, temperature=0)
    package = _package()
    first = client.complete(package)
    second = client.complete(package)
    assert first.parsed == {"answer": "mock", "rationale": "mock reply from mock-llm"}
    assert first.cached is False
    assert second.cached is True
    assert provider.calls == 1
    assert "SECRET" not in package.user


def test_mock_distribution_uses_listed_options():
    provider = MockProvider()
    client = LLMClient(provider, model="mock-llm", temperature=None)
    result = client.complete(_package("distribution"))
    assert result.parsed["distribution"] == {"A": 0.5, "B": 0.5}
    assert result.parse_error is None


def test_retry_is_used_for_transport_failures(tmp_path):
    provider = MockProvider(fail_times=2)
    client = LLMClient(provider, model="mock-llm", cache_dir=tmp_path, attempts=3, temperature=0)
    result = client.complete(_package())
    assert result.parsed["answer"] == "mock"
    assert provider.calls == 3


def test_cache_key_changes_with_temperature():
    package = _package()
    cold = cache_key(package, provider="mock", model="mock-llm", temperature=0)
    hot = cache_key(package, provider="mock", model="mock-llm", temperature=0.8)
    assert cold != hot


def test_openai_provider_posts_chat_completions(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "{\"answer\": \"red\"}"}}]}

    def post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return Response()

    import httpx

    monkeypatch.setattr(httpx, "post", post)
    provider = OpenAICompatProvider(api_key="test-key", base_url="https://example.test/v1")
    raw = provider.complete(
        [{"role": "user", "content": "hi"}],
        model="gpt-test",
        temperature=0,
    )
    assert json.loads(raw)["answer"] == "red"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "gpt-test"


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(CommsError):
        OpenAICompatProvider(api_key="")


def test_cr_provider_delegates_to_cr_client(monkeypatch):
    class Result:
        content = "{\"answer\": \"blue\"}"

    class FakeClient:
        def __init__(self, api_key=None):
            assert api_key == "cr-test"

        def chat(self, messages, model="deepseek-v4-pro", temperature=None):
            assert model == "deepseek-v4-pro"
            assert temperature == 0
            return Result()

    import cr_api.client as cr_client

    monkeypatch.setattr(cr_client, "CRClient", FakeClient)
    provider = CRProvider(api_key="cr-test")
    assert json.loads(provider.complete([{"role": "user", "content": "hi"}], model="deepseek-v4-pro", temperature=0))[
        "answer"
    ] == "blue"


def test_cr_provider_wraps_cr_errors(monkeypatch):
    import cr_api.client as cr_client

    class FakeClient:
        def __init__(self, api_key=None):
            return None

        def chat(self, *args, **kwargs):
            raise cr_client.CRError("nope")

    monkeypatch.setattr(cr_client, "CRClient", FakeClient)
    provider = CRProvider(api_key="cr-test")
    with pytest.raises(CommsTransportError, match="nope"):
        provider.complete([], model="deepseek-v4-pro")


def test_package_round_trip_shape():
    package = _package()
    assert isinstance(package, PromptPackage)
    assert package.messages()[0]["role"] == "system"
