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
    if bundle.name == "twin2k_mega_study" and method != "group_prompt":
        if twin is None:
            raise PersonaMethodError(
                "Mega-Study rows do not include persona text. "
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
) -> tuple[SimulationResult, ...]:
    chosen = bundle.limited(limit)
    if not chosen.cases:
        raise PersonaMethodError(f"{bundle.name} has no question cases to simulate")
    results: list[SimulationResult] = []
    for case in chosen.cases:
        kwargs = {"seed": seed} if method in {"atus_expanded", "bdm_survey_persona"} else {}
        package = build_prompt(method, case, **kwargs)
        results.append(client.complete(package))
    return tuple(results)


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
