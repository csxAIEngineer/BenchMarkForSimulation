"""Persona prompts. Twin-2K rendering and Nemotron sampling are later steps.

Fetching stores one Nemotron-USA parquet shard on disk. Callers must not load
every persona text in that shard into memory just to draw a small sample.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

NEMOTRON_REPO = "nvidia/Nemotron-Personas-USA"
# Hugging Face dataset revision used for the v1 cache.
NEMOTRON_REVISION = "5b4cd35ab46490c1da1bd2b5a2324d6f871be180"
NEMOTRON_SHARD_COUNT = 11
# Strata for the sample the simulator will draw. `state` is the geography margin.
NEMOTRON_STRATA = ("age", "sex", "state")


class PersonaError(RuntimeError):
    """The Nemotron shard cache is missing or the shard request is out of range."""


@dataclass(frozen=True)
class Persona:
    id: str
    condition: str
    prompt: str


def nemotron_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "nemotron_usa"


def nemotron_shard_name(index: int) -> str:
    if not 0 <= index < NEMOTRON_SHARD_COUNT:
        raise PersonaError(
            f"Nemotron shard index must be in 0..{NEMOTRON_SHARD_COUNT - 1}"
        )
    return f"data/train-{index:05d}-of-{NEMOTRON_SHARD_COUNT:05d}.parquet"


def cache_nemotron_shards(data_dir: Path, n_shards: int = 1) -> list[Path]:
    """Download the first `n_shards` Nemotron-USA parquet files into the cache.

    The public dump is about 1M records and 2.7GB. v1 keeps a shard cache and
    samples from it instead of reading the full corpus.
    """
    from huggingface_hub import snapshot_download

    if not 1 <= n_shards <= NEMOTRON_SHARD_COUNT:
        raise PersonaError(f"n_shards must be in 1..{NEMOTRON_SHARD_COUNT}")
    hub = Path(data_dir) / ".hf-home"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hub))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    dest = nemotron_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    patterns = [nemotron_shard_name(index) for index in range(n_shards)]
    try:
        snapshot_download(
            repo_id=NEMOTRON_REPO,
            repo_type="dataset",
            revision=NEMOTRON_REVISION,
            allow_patterns=patterns,
            local_dir=dest,
        )
    except Exception as exc:
        raise PersonaError(
            f"could not download {NEMOTRON_REPO}@{NEMOTRON_REVISION}: {exc}"
        ) from exc
    paths = [dest / name for name in patterns]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PersonaError("Nemotron download did not produce: " + ", ".join(missing))
    manifest = {
        "repo_id": NEMOTRON_REPO,
        "revision": NEMOTRON_REVISION,
        "shards": [
            {"path": name, "bytes": (dest / name).stat().st_size, "rows": _parquet_rows(dest / name)}
            for name in patterns
        ],
        "shards_total": NEMOTRON_SHARD_COUNT,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def render_twin2k_profile(respondent_id: str, condition: str) -> str:
    """Render one Twin-2K respondent as the persona prompt for `condition`."""
    raise NotImplementedError("Twin-2K persona rendering is not implemented yet")


def sample_nemotron(n: int, seed: int, data_dir: Path) -> list[Persona]:
    """Draw a seeded sample stratified on age, sex, and state."""
    raise NotImplementedError("Nemotron stratified sampling is not implemented yet")
