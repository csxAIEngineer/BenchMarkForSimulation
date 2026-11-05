#!/usr/bin/env python3
"""Score pricing-only CR model runs. Does not call a model API.

Writes ``compare_models.json`` only when every model/dataset job has
responses.jsonl. The table columns are TVD, Top-1, individual accuracy,
Fisher-z correlation, Glass Δ, and SD ratio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from popbench.datasets.io import read_jsonl  # noqa: E402
from pricing_models_cr import (  # noqa: E402
    COLUMN_LABELS,
    DATASETS,
    METHOD,
    MODEL_IDS,
    OMITTED_FROM_TABLE,
    OUT_ROOT,
    SEED,
    TABLE_COLUMNS,
    job_dir,
    public_table_row,
)


def score_rows(dataset: str, rows: list[dict]) -> dict:
    from score_cr_skill_compare_metrics import score_bdm, score_twin2k

    if dataset == "twin2k_500":
        return score_twin2k(rows)
    if dataset == "bdm_lettuce":
        return score_bdm(rows)
    raise ValueError(f"unsupported dataset {dataset}")


def main() -> int:
    present: list[tuple[str, str, Path]] = []
    missing: list[str] = []
    for model in MODEL_IDS:
        for dataset in DATASETS:
            path = job_dir(model, dataset) / "responses.jsonl"
            if path.is_file() and path.stat().st_size > 0:
                present.append((model, dataset, path))
            else:
                missing.append(str(path))

    if not present:
        print("NOT_READY", flush=True)
        for path in missing:
            print(path, flush=True)
        print("compare table not written", flush=True)
        return 2

    table = []
    for model, dataset, path in present:
        metrics = score_rows(dataset, read_jsonl(path))
        slim = {key: value for key, value in metrics.items() if key != "items"}
        row = public_table_row(dataset, model, slim)
        table.append(row)
        job_dir(model, dataset).mkdir(parents=True, exist_ok=True)
        (job_dir(model, dataset) / "run_summary_metrics.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": METHOD,
                    "partial_experiment": bool(missing),
                    "table": row,
                    "detail": slim,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if missing:
        print("NOT_READY", flush=True)
        for path in missing:
            print(path, flush=True)
        print("compare table not written", flush=True)
        return 2

    payload = {
        "method": METHOD,
        "seed": SEED,
        "n_definition": "subjects, income-stratified",
        "models": list(MODEL_IDS),
        "column_labels": COLUMN_LABELS,
        "columns": list(TABLE_COLUMNS),
        "excluded_from_table": list(OMITTED_FROM_TABLE),
        "table": table,
        "run_dir": str(OUT_ROOT),
        "no_api_calls": True,
    }
    (OUT_ROOT / "compare_models.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
