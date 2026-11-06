"""Shared constants for the pricing-only CR multi-model benchmark.

``n`` is a subject count, matching ``scripts/run_skill_compare_cr.py``.
The persona method is the project default base card, ``atus_expanded``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from popbench.datasets.bdm_lettuce.fields import OUTCOME_COLUMNS
from popbench.schema import DatasetBundle

ROOT = Path(__file__).resolve().parents[1]
PRICING_DIR = ROOT / "data" / "pricing_only"
OUT_ROOT = ROOT / "runs" / "cr-pricing-models-n50"
LOG_PATH = ROOT / "logs" / "cr-pricing-models-n50.log"

METHOD = "atus_expanded"
PROVIDER = "cr"
SEED = 42
N_SUBJECTS = 50
BATCH_SIZE = 15
REQUEST_TIMEOUT = 600

TWIN_PRICING_PATH = PRICING_DIR / "twin2k_500_pricing_subjects.jsonl"
BDM_PRICING_PATH = PRICING_DIR / "bdm_lettuce_pricing_subjects.jsonl"
TWIN_QUESTIONS_PER_SUBJECT = 40
BDM_QUESTIONS_PER_SUBJECT = 7

# Slugs are the strings in cr_api.client.TEXT_MODELS. GLM has no prior run
# in this repo; glm-5.3 is the first listed non-flash GLM id.
MODELS: tuple[dict[str, str], ...] = (
    {
        "id": "deepseek-v4-pro",
        "label": "deepseek v4 pro",
        "source": "cr_api.TEXT_MODELS['deepseek'][0]; popbench.cli default for provider cr",
    },
    {
        "id": "gpt-5.5",
        "label": "gpt5.5",
        "source": "cr_api.TEXT_MODELS['gpt'][0]",
    },
    {
        "id": "glm-5.3",
        "label": "glm",
        "source": (
            "cr_api.TEXT_MODELS['glm'][0]; no runner or log pinned a GLM id; "
            "skipped glm-5.3-flash and glm-5.2"
        ),
    },
)

MODEL_IDS: tuple[str, ...] = tuple(spec["id"] for spec in MODELS)
DATASETS: tuple[str, ...] = ("twin2k_500", "bdm_lettuce")
BDM_PRICING_IDS = frozenset(OUTCOME_COLUMNS)

TABLE_COLUMNS: tuple[str, ...] = (
    "mean_tvd",
    "mean_top1_acc",
    "indiv_acc",
    "fisher_z_corr",
    "glass_delta",
    "sd_ratio",
)

COLUMN_LABELS: dict[str, str] = {
    "mean_tvd": "TVD",
    "mean_top1_acc": "Top-1 Accuracy",
    "indiv_acc": "个体准确率（1−MAD/range）",
    "fisher_z_corr": "跨人相关（Fisher z 平均）",
    "glass_delta": "Glass Δ",
    "sd_ratio": "SD 比",
}

OMITTED_FROM_TABLE: tuple[str, ...] = ("accuracy", "n_scored", "parse_errors")


def documented_model_ids() -> set[str]:
    from cr_api.client import TEXT_MODELS

    return {model for family in TEXT_MODELS.values() for model in family}


def assert_models_documented(model_ids: tuple[str, ...] = MODEL_IDS) -> None:
    known = documented_model_ids()
    unknown = [model for model in model_ids if model not in known]
    if unknown:
        raise ValueError(f"model ids not in cr_api.TEXT_MODELS: {unknown}")


def is_twin_pricing(question_id: str) -> bool:
    return str(question_id).startswith("QID9_")


def is_bdm_pricing(question_id: str) -> bool:
    return str(question_id) in BDM_PRICING_IDS


def filter_questions(bundle: DatasetBundle, keep: Callable[[str], bool]) -> DatasetBundle:
    cases = tuple(case for case in bundle.cases if keep(case.question.id))
    kept_ids = {case.subject.id for case in cases}
    subjects = tuple(subject for subject in bundle.subjects if subject.id in kept_ids)
    return DatasetBundle(
        name=bundle.name,
        unit=bundle.unit,
        source=bundle.source,
        revision=bundle.revision,
        subjects=subjects,
        cases=cases,
    )


def question_counts(bundle: DatasetBundle) -> dict[str, int]:
    counts = {subject.id: 0 for subject in bundle.subjects}
    for case in bundle.cases:
        counts[case.subject.id] = counts.get(case.subject.id, 0) + 1
    return counts


def assert_uniform_question_count(bundle: DatasetBundle, expected: int) -> None:
    counts = question_counts(bundle)
    bad = {subject_id: n for subject_id, n in counts.items() if n != expected}
    if bad:
        sample = list(bad.items())[:5]
        raise ValueError(
            f"{bundle.name} expected {expected} pricing questions per subject; "
            f"{len(bad)} subjects differ, e.g. {sample}"
        )


def planned_batches(n_subjects: int, questions_per_subject: int, batch_size: int) -> int:
    if n_subjects < 1 or questions_per_subject < 1 or batch_size < 1:
        raise ValueError("batch plan needs positive sizes")
    return n_subjects * math.ceil(questions_per_subject / batch_size)


def job_dir(model: str, dataset: str, root: Path = OUT_ROOT) -> Path:
    return root / model / dataset / METHOD


def job_is_complete(out_dir: Path) -> bool:
    responses = out_dir / "responses.jsonl"
    manifest = out_dir / "manifest.json"
    return responses.is_file() and responses.stat().st_size > 0 and manifest.is_file()


def public_table_row(dataset: str, model: str, metrics: dict) -> dict:
    row: dict[str, object] = {"dataset": dataset, "model": model}
    for key in TABLE_COLUMNS:
        row[key] = metrics.get(key)
    return row
