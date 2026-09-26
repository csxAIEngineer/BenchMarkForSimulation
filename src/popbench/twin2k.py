"""Download and read the Twin-2K-500 question catalog and response tables.

Split construction (decision versus sentiment items) is a later step. This
module only caches the files and loads them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from popbench.schema import TWIN2K_N

TWIN2K_REPO = "LLM-Digital-Twin/Twin-2K-500"
# Hugging Face `main` as of the dataset card used for v1.
TWIN2K_REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
WAVE1_3_COLUMNS = 761
WAVE4_COLUMNS = 127
CATALOG_QUESTIONS = 256

_CATALOG_DIR = "question_catalog_and_human_response_csv"
_FILES = (
    "question_catalog.json",
    "question_catalog_README.md",
    "wave1_3_response.csv",
    "wave1_3_response_label.csv",
    "wave4_response.csv",
    "wave4_response_label.csv",
)


class DatasetError(RuntimeError):
    """The Twin-2K cache is missing, incomplete, or does not match the published shapes."""


@dataclass(frozen=True)
class Twin2KData:
    """Catalog plus the numeric and text-label response tables."""

    root: Path
    revision: str
    catalog: list[dict[str, Any]]
    wave1_3: pd.DataFrame
    wave1_3_labels: pd.DataFrame
    wave4: pd.DataFrame
    wave4_labels: pd.DataFrame

    @property
    def n_respondents(self) -> int:
        return len(self.wave1_3)


def twin2k_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "twin2k"


def catalog_dir(data_dir: Path) -> Path:
    return twin2k_dir(data_dir) / _CATALOG_DIR


def _file_path(data_dir: Path, name: str) -> Path:
    return catalog_dir(data_dir) / name


def cache_is_complete(data_dir: Path) -> bool:
    return all(_file_path(data_dir, name).is_file() for name in _FILES)


def download_twin2k(data_dir: Path) -> Path:
    """Download the catalog and wave response CSVs into `data_dir/twin2k`.

    Persona text, raw Qualtrics exports, and the published LLM simulation
    dumps are left on the Hub. v1 scores from these tables.
    """
    from huggingface_hub import snapshot_download

    _prepare_hub_cache(data_dir)
    dest = twin2k_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=TWIN2K_REPO,
            repo_type="dataset",
            revision=TWIN2K_REVISION,
            allow_patterns=[f"{_CATALOG_DIR}/{name}" for name in _FILES],
            local_dir=dest,
        )
    except Exception as exc:
        raise DatasetError(
            f"could not download {TWIN2K_REPO}@{TWIN2K_REVISION}: {exc}"
        ) from exc
    missing = [name for name in _FILES if not _file_path(data_dir, name).is_file()]
    if missing:
        raise DatasetError("Twin-2K download did not produce: " + ", ".join(missing))
    _write_manifest(data_dir)
    return dest


def load_twin2k(data_dir: Path, *, download: bool = True) -> Twin2KData:
    """Read the cached catalog and response CSVs, downloading them if needed."""
    if download and not cache_is_complete(data_dir):
        download_twin2k(data_dir)
    if not cache_is_complete(data_dir):
        raise DatasetError(
            f"Twin-2K cache is incomplete under {twin2k_dir(data_dir)}. "
            "Run `popbench fetch`."
        )
    catalog = _read_catalog(_file_path(data_dir, "question_catalog.json"))
    wave1_3 = _read_responses(_file_path(data_dir, "wave1_3_response.csv"), WAVE1_3_COLUMNS)
    wave1_3_labels = _read_responses(
        _file_path(data_dir, "wave1_3_response_label.csv"),
        WAVE1_3_COLUMNS,
    )
    wave4 = _read_responses(_file_path(data_dir, "wave4_response.csv"), WAVE4_COLUMNS)
    wave4_labels = _read_responses(
        _file_path(data_dir, "wave4_response_label.csv"),
        WAVE4_COLUMNS,
    )
    _same_respondents(wave1_3, wave1_3_labels, wave4, wave4_labels)
    if len(catalog) != CATALOG_QUESTIONS:
        raise DatasetError(
            f"question catalog has {len(catalog)} entries, expected {CATALOG_QUESTIONS}"
        )
    return Twin2KData(
        root=twin2k_dir(data_dir),
        revision=TWIN2K_REVISION,
        catalog=catalog,
        wave1_3=wave1_3,
        wave1_3_labels=wave1_3_labels,
        wave4=wave4,
        wave4_labels=wave4_labels,
    )


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise DatasetError(f"question catalog must be a list: {path}")
    for entry in payload:
        if not isinstance(entry, dict) or "QuestionID" not in entry:
            raise DatasetError(f"catalog entry is missing QuestionID in {path}")
    return payload


def _read_responses(path: Path, n_columns: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.shape != (TWIN2K_N, n_columns):
        raise DatasetError(
            f"{path.name} has shape {frame.shape}, expected {(TWIN2K_N, n_columns)}"
        )
    if frame.columns[0] != "pid":
        raise DatasetError(f"{path.name} must start with a pid column")
    if frame["pid"].nunique() != TWIN2K_N:
        raise DatasetError(f"{path.name} does not have {TWIN2K_N} distinct pids")
    return frame


def _same_respondents(*frames: pd.DataFrame) -> None:
    pid = frames[0]["pid"].astype(str).tolist()
    for frame in frames[1:]:
        if frame["pid"].astype(str).tolist() != pid:
            raise DatasetError("response files do not share the same pid order")


def _prepare_hub_cache(data_dir: Path) -> None:
    hub = Path(data_dir) / ".hf-home"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hub))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _write_manifest(data_dir: Path) -> None:
    files = {}
    for name in _FILES:
        path = _file_path(data_dir, name)
        files[name] = {"bytes": path.stat().st_size}
    manifest = {
        "repo_id": TWIN2K_REPO,
        "revision": TWIN2K_REVISION,
        "files": files,
    }
    (twin2k_dir(data_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
