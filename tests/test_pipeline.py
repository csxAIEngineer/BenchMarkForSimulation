"""Minimal path: cached dataset, one persona method, mock LLM."""

import json
from pathlib import Path

import pytest

from popbench.cli import main
from popbench.comms.client import LLMClient
from popbench.comms.providers.mock import MockProvider
from popbench.datasets.io import write_jsonl
from popbench.datasets.twin2k_500.loader import bundle_from_records
from popbench.datasets.twin2k_mega_study.loader import bundle_from_records as mega_bundle
from popbench.models.errors import PersonaMethodError
from popbench.pipeline import prepare_bundle, simulate


def _write_twin(data_dir: Path) -> None:
    write_jsonl(
        data_dir / "twin2k_500" / "subjects.jsonl",
        [
            {
                "id": "7",
                "demographics": {"age": "34", "sex": "Female"},
                "demographic_text": "Age: 34\nSex: Female",
                "persona_text": "Full card",
                "persona_summary": "Short summary",
                "questions": [
                    {
                        "id": "Q9",
                        "prompt": "Pick one",
                        "options": ["red", "blue"],
                        "kind": "multiple_choice",
                        "gold": "SECRET_GOLD",
                    }
                ],
            }
        ],
    )


def test_simulate_with_mock_does_not_leak_gold(tmp_path):
    bundle = bundle_from_records(
        [
            {
                "id": "7",
                "demographic_text": "Age: 34",
                "questions": [
                    {
                        "id": "Q9",
                        "prompt": "Pick one",
                        "options": ["red", "blue"],
                        "kind": "multiple_choice",
                        "gold": "SECRET_GOLD",
                    }
                ],
            }
        ]
    )
    client = LLMClient(MockProvider(), model="mock-llm", cache_dir=tmp_path / "cache")
    results = simulate(bundle, method="atus_expanded", client=client, limit=1)
    assert results[0].parsed["answer"] == "mock"
    assert results[0].parse_error is None


def test_cli_simulate_writes_a_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_twin(tmp_path / "data")
    code = main(
        [
            "simulate",
            "--dataset",
            "twin2k_500",
            "--method",
            "atus_expanded",
            "--provider",
            "mock",
            "--limit",
            "1",
            "--out",
            "runs/minimal",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "runs" / "minimal" / "responses.jsonl").read_text())
    assert payload["method"] == "atus_expanded"
    assert payload["provider"] == "mock"
    assert payload["parsed"]["answer"] == "mock"
    manifest = json.loads((tmp_path / "runs" / "minimal" / "manifest.json").read_text())
    assert manifest["dataset"] == "twin2k_500"
    assert "SECRET_GOLD" not in (tmp_path / "runs" / "minimal" / "responses.jsonl").read_text()


def test_mega_study_requires_a_twin2k_bundle_before_an_individual_method():
    mega = mega_bundle(
        [
            {
                "id": "7",
                "study": "privacy",
                "survey_text": "New study question",
                "gold_payload": "{\"kept\": true}",
            }
        ]
    )
    with pytest.raises(PersonaMethodError, match="attach"):
        prepare_bundle(mega, method="atus_expanded", twin=None)


def test_help_names_the_layers(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "fetch" in text
    assert "simulate" in text
