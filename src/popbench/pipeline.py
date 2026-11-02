"""Minimal path: dataset case, persona method, then the communication layer."""

from __future__ import annotations

import json
from pathlib import Path

from popbench.schema import DatasetBundle, SimulationResult
from popbench.comms.client import LLMClient
from popbench.datasets.twin2k_mega_study.loader import attach_twin2k
from popbench.models import build_prompt
from popbench.models.errors import PersonaMethodError


def prepare_bundle(
    bundle: DatasetBundle,
    *,
    method: str,
    twin: DatasetBundle | None = None,
) -> DatasetBundle:
    """Attach Twin-2K personas when a Mega-Study run asks for an individual card."""
    if bundle.name == "twin2k_mega_study":
        if twin is None:
            raise PersonaMethodError(
                f"{method} needs Twin-2K demographics on Mega-Study rows. "
                "Pass a Twin-2K-500 bundle or fetch twin2k_500 and use --attach-twin2k."
            )
        return attach_twin2k(bundle, twin)
    return bundle


def simulate(
    bundle: DatasetBundle,
    *,
    method: str,
    client: LLMClient,
    limit: int | None = None,
    seed: int = 0,
    batch_size: int = 15,
) -> tuple[SimulationResult, ...]:
    """Ask each subject's questions in batches. The persona card is built once per batch."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise PersonaMethodError("batch_size must be a positive integer")
    chosen = bundle.limited(limit)
    if not chosen.cases:
        raise PersonaMethodError(f"{bundle.name} has no question cases to simulate")
    packages = [build_prompt(method, case, seed=seed) for case in chosen.cases]
    results: list[SimulationResult] = []
    for group in _by_subject(packages):
        for start in range(0, len(group), batch_size):
            results.extend(client.complete_batch(group[start : start + batch_size]))
    return tuple(results)


def _by_subject(packages: list) -> list[list]:
    """Keep each subject's questions together, in their original order."""
    order: list[str] = []
    grouped: dict[str, list] = {}
    for package in packages:
        if package.subject_id not in grouped:
            order.append(package.subject_id)
            grouped[package.subject_id] = []
        grouped[package.subject_id].append(package)
    return [grouped[subject_id] for subject_id in order]


def write_results(directory: Path, results: tuple[SimulationResult, ...], *, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(result.to_dict(), ensure_ascii=False) for result in results]
    target = directory / "responses.jsonl"
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
