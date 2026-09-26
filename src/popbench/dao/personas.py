"""Nemotron-USA persona cache and the adult sample the interview is built from.

Fetching stores parquet shards on disk. Callers draw a small sample instead of
reading every persona text into memory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NEMOTRON_REPO = "nvidia/Nemotron-Personas-USA"
# Hugging Face dataset revision used for the v1 cache.
NEMOTRON_REVISION = "5b4cd35ab46490c1da1bd2b5a2324d6f871be180"
NEMOTRON_SHARD_COUNT = 11
# Strata for the sample the simulator will draw. `state` is the geography margin.
NEMOTRON_STRATA = ("age", "sex", "state")
NARRATIVE_FIELDS = (
    "persona",
    "professional_persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
)
AGE_BANDS = (
    (18, 29, "18-29"),
    (30, 49, "30-49"),
    (50, 64, "50-64"),
    (65, 200, "65+"),
)


class PersonaError(RuntimeError):
    """The Nemotron shard cache is missing or the shard request is out of range."""


@dataclass(frozen=True)
class NemotronAdult:
    """One sampled Nemotron adult, already rendered as persona text."""

    id: str
    age: int
    age_band: str
    sex: str
    state: str
    education_level: str
    occupation: str
    marital_status: str
    text: str


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


def allocate_quotas(weights: dict[str, int], n: int) -> dict[str, int]:
    """Spread `n` seats across keys in proportion to positive weights."""
    usable = {key: weight for key, weight in weights.items() if weight > 0}
    total = sum(usable.values())
    if n < 1 or total < n:
        raise ValueError("not enough weighted rows to fill the quota")
    raw = {key: n * weight / total for key, weight in usable.items()}
    base = {key: int(value) for key, value in raw.items()}
    leftover = n - sum(base.values())
    ranked = sorted(usable, key=lambda key: (raw[key] - base[key], usable[key]), reverse=True)
    for key in ranked:
        if leftover == 0:
            break
        if base[key] < usable[key]:
            base[key] += 1
            leftover -= 1
    if leftover:
        raise ValueError("quota allocation could not fill every seat")
    return {key: count for key, count in base.items() if count}


def age_band(age: int) -> str | None:
    for low, high, name in AGE_BANDS:
        if low <= int(age) <= high:
            return name
    return None


def persona_text(row: Any) -> str:
    """Render the demographic line and the narrative fields as one persona card."""
    lines = [
        f"Age: {int(row.age)}",
        f"Sex: {row.sex}",
        f"State: {row.state}",
        f"Education: {row.education_level}",
        f"Occupation: {row.occupation}",
        f"Marital status: {row.marital_status}",
        "",
    ]
    for field in NARRATIVE_FIELDS:
        text = getattr(row, field, None)
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
            lines.append("")
    return "\n".join(lines).strip()


def sample_usa_adults(data_dir: Path, n: int, seed: int) -> list[NemotronAdult]:
    """Adults, quota-sampled on age band and sex. State follows the shard."""
    import pyarrow.parquet as pq

    path = nemotron_dir(data_dir) / nemotron_shard_name(0)
    if not path.is_file():
        raise PersonaError(f"Nemotron shard is missing: {path}")
    columns = [
        "uuid",
        "age",
        "sex",
        "state",
        "education_level",
        "occupation",
        "marital_status",
        *NARRATIVE_FIELDS,
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    adults = frame.loc[frame["age"] >= 18].copy()
    adults["age_band"] = adults["age"].map(age_band)
    adults = adults.loc[adults["age_band"].notna() & adults["sex"].isin(["Female", "Male"])]
    adults["stratum"] = adults["age_band"] + "|" + adults["sex"]
    quotas = allocate_quotas(adults["stratum"].value_counts().to_dict(), n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum, count in quotas.items():
        pool = adults.index[adults["stratum"] == stratum].to_numpy()
        picked.extend(rng.choice(pool, size=count, replace=False).tolist())
    panel = adults.loc[picked].sort_values(["stratum", "uuid"])
    people = []
    for row in panel.itertuples(index=False):
        people.append(
            NemotronAdult(
                id=str(row.uuid),
                age=int(row.age),
                age_band=str(row.age_band),
                sex=str(row.sex),
                state=str(row.state),
                education_level=str(row.education_level),
                occupation=str(row.occupation),
                marital_status=str(row.marital_status),
                text=persona_text(row),
            )
        )
    return people


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)
