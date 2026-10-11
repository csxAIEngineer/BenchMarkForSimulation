"""ATUS persona cards answering one Twin-2K paper category at a time.

People come from the same ATUS draw as the six-item probe. Each card's text is
the written adult persona for that TUCASEID. Each category is a separate
interview: later categories do not see answers from earlier ones. The survey
system prompt is that card. The human baseline is the category's wave 1–3
response columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from popbench.dao.atus_twin2k import build_atus_personas
from popbench.dao.interview import InterviewError
from popbench.dao.io import write_jsonl
from popbench.dao.survey import (
    CATEGORY_LABELS,
    PAPER_FAMILIES,
    build_closed_items,
    human_baseline,
    items_by_family,
)


def category_families(names: list[str] | None) -> list[str]:
    """Paper categories in catalog order. ``None`` keeps all seven."""
    if not names:
        return list(PAPER_FAMILIES)
    unknown = [name for name in names if name not in PAPER_FAMILIES]
    if unknown:
        known = ", ".join(PAPER_FAMILIES)
        raise InterviewError("unknown paper category: " + ", ".join(unknown) + f". Expected {known}")
    return [name for name in PAPER_FAMILIES if name in names]


def category_records(
    personas: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One survey record per persona. Turns are only this category's closed items."""
    turns = [_turn(item) for item in items]
    records = []
    for person in personas:
        demographics = person.get("demographics") or {}
        records.append(
            {
                "id": person["id"],
                "persona": {
                    "id": person["id"],
                    "age_band": demographics.get("age_band"),
                    "sex": demographics.get("sex"),
                    "text": person["text"],
                },
                "turns": turns,
            }
        )
    return records


def build_atus_category_panels(
    data_dir: Path,
    *,
    n: int,
    seed: int,
    out_dir: Path,
    families: list[str] | None = None,
    n_per_age_band: int | None = None,
) -> dict[str, Any]:
    """Write shared ATUS cards and one items/baseline folder per category.

    ``n_per_age_band`` draws that many adults from each age band and ignores ``n``.
    """
    from popbench.dao.twin2k import DatasetError, load_twin2k

    selected = category_families(families)
    try:
        dataset = load_twin2k(data_dir, download=False)
    except DatasetError as exc:
        raise InterviewError(str(exc)) from exc
    grouped = items_by_family(build_closed_items(dataset.catalog))
    personas = build_atus_personas(
        data_dir,
        n=n,
        seed=seed,
        n_per_age_band=n_per_age_band,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "personas.jsonl", personas)
    panels = []
    for family in selected:
        items = grouped[family]
        if not items:
            raise InterviewError(f"{family} has no closed items")
        baseline = human_baseline(dataset.wave1_3, items)
        records = category_records(personas, items)
        folder = out_dir / "families" / family
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "items.json").write_text(json.dumps(items, ensure_ascii=False) + "\n")
        (folder / "human_baseline.json").write_text(json.dumps(baseline) + "\n")
        manifest = {
            "family": family,
            "label": CATEGORY_LABELS[family],
            "n_items": len(items),
            "n_people": len(records),
            "seed": seed,
            "n_per_age_band": n_per_age_band,
            "sampling": "age_band" if n_per_age_band else "age_band_x_sex",
            "baseline": "Twin-2K wave1_3 shares for this category",
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        panels.append(
            {
                "family": family,
                "label": CATEGORY_LABELS[family],
                "items": items,
                "baseline": baseline,
                "records": records,
                "dir": folder,
            }
        )
    return {
        "personas": personas,
        "panels": panels,
        "n": len(personas),
        "seed": seed,
        "n_per_age_band": n_per_age_band,
        "out_dir": out_dir,
    }


def _turn(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": item["turn"],
        "item_id": item["item_id"],
        "kind": item["kind"],
        "family": item["family"],
        "block": item["block"],
        "instrument": item["instrument"],
        "prompt": item["prompt"],
        "options": list(item["options"]),
    }
