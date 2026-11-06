"""Independent-process builds, lock fail-closed behavior, and tamper checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_pricing_basic_info.py"
OUTPUT = ROOT / "experiments" / "pricing-basic-info"
RAW = ROOT / "data" / "pricing_basic_info" / "raw"


def _config(tmp: Path, *, output: Path, cache: Path) -> Path:
    payload = {
        "cache_dir": str(cache),
        "output_dir": str(output),
        "sample_size": None,
        "sample_size_rule": {"margin_of_error": "0.10", "proportion": "0.5", "z": "1.96"},
        "seed": 42,
    }
    path = tmp / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_lock_fails_and_does_not_create_one(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    config = _config(tmp_path, output=output, cache=cache)
    result = _run([str(config)])
    assert result.returncode != 0
    assert "source-lock.json is missing" in result.stderr
    assert not (output / "source-lock.json").exists()


def test_mismatched_cache_fails_without_updating_the_lock(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    lock = {
        "bdm_doi": "10.5683/SP3/CDLUEZ",
        "bdm_version": "V1",
        "files": {
            "question_catalog_and_human_response_csv/wave1_3_response_label.csv": {
                "bytes": 7,
                "sha256": "0" * 64,
            }
        },
        "twin2k_revision": "f883165a3026fde855dfd448e0cd16443ab257b6",
    }
    lock_path = output / "source-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    before = lock_path.read_bytes()
    cache = tmp_path / "cache" / "raw" / "twin2k" / "question_catalog_and_human_response_csv"
    cache.mkdir(parents=True)
    (cache / "wave1_3_response_label.csv").write_text("corrupt", encoding="utf-8")
    config = _config(tmp_path, output=output, cache=tmp_path / "cache")
    result = _run([str(config)])
    assert result.returncode != 0
    assert "does not match source-lock.json" in result.stderr
    assert lock_path.read_bytes() == before


def test_verify_accepts_committed_artifacts_and_rejects_one_byte(tmp_path: Path):
    config = _config(tmp_path, output=OUTPUT, cache=tmp_path / "unused")
    ok = _run(["--verify", str(config)])
    assert ok.returncode == 0, ok.stderr
    copied = tmp_path / "tree"
    copied.mkdir()
    for path in OUTPUT.rglob("*"):
        if path.is_file():
            target = copied / path.relative_to(OUTPUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    bad_config = _config(tmp_path, output=copied, cache=tmp_path / "unused2")
    intact = _run(["--verify", str(bad_config)])
    assert intact.returncode == 0, intact.stderr
    cards = copied / "twin2k_500" / "cards.jsonl"
    original = cards.read_bytes()
    cards.write_bytes(original[:-1] + (b"X" if original[-1:] != b"X" else b"Y"))
    tampered = _run(["--verify", str(bad_config)])
    assert tampered.returncode != 0
    assert "hash mismatch" in tampered.stderr
    cards.write_bytes(original)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    broken_manifest = _run(["--verify", str(bad_config)])
    assert broken_manifest.returncode != 0
    assert "manifest_sha256" in broken_manifest.stderr


@pytest.mark.skipif(not (RAW / "bdm" / "20240723_VFExperiment_01.csv").is_file(), reason="pinned raw cache is not present")
def test_two_independent_builds_match(tmp_path: Path):
    runs = []
    for index in (1, 2):
        output = tmp_path / f"out{index}"
        output.mkdir()
        (output / "source-lock.json").write_bytes((OUTPUT / "source-lock.json").read_bytes())
        cache = tmp_path / f"cache{index}"
        raw_link = cache / "raw"
        raw_link.parent.mkdir(parents=True, exist_ok=True)
        raw_link.symlink_to(RAW)
        config = _config(tmp_path, output=output, cache=cache)
        result = _run([str(config)])
        assert result.returncode == 0, result.stderr
        runs.append((output, cache / "outcomes"))
    left, right = runs
    names = [
        "field_map.json",
        "manifest.json",
        "source-lock.json",
        "twin2k_500/cards.jsonl",
        "twin2k_500/sample.json",
        "twin2k_500/distribution.json",
        "twin2k_500/exclusions.json",
        "twin2k_500/wave_audit.json",
        "bdm_lettuce/cards.jsonl",
        "bdm_lettuce/sample.json",
        "bdm_lettuce/distribution.json",
        "bdm_lettuce/exclusions.json",
    ]
    for name in names:
        assert (left[0] / name).read_bytes() == (right[0] / name).read_bytes()
    for name in ("twin2k_500/outcomes.jsonl", "bdm_lettuce/outcomes.jsonl"):
        assert (left[1] / name).read_bytes() == (right[1] / name).read_bytes()
    manifest = json.loads((left[0] / "manifest.json").read_text(encoding="utf-8"))
    assert "output_dir" not in manifest
    from popbench.pricing_basic_info.pipeline import verify_outputs

    verify_outputs(left[0])
