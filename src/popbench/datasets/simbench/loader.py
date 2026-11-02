"""Download pitehu/SimBench and keep every row a group distribution.

SimBench does not publish individual respondents. This loader refuses rows
that look like person-level microdata and never invents member ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from popbench.schema import Case, DatasetBundle, Gold, Question, Subject
from popbench.datasets.io import (
    DatasetError,
    parse_jsonish,
    read_jsonl,
    require_positive_limit,
    write_jsonl,
    write_manifest,
)

REPO_ID = "pitehu/SimBench"
DISPLAY_NAME = "SimBench"
SPLITS = ("SimBenchPop", "SimBenchGrouped")


class SimBench:
    name = "simbench"
    display_name = DISPLAY_NAME
    unit = "group"
    source = REPO_ID

    def cache_dir(self, data_dir: Path) -> Path:
        return Path(data_dir) / "simbench"

    def cache_is_complete(self, data_dir: Path, *, split: str | None = None) -> bool:
        root = self.cache_dir(data_dir)
        names = (split,) if split else SPLITS
        return all((root / f"{name}.jsonl").is_file() for name in names)

    def download(
        self,
        data_dir: Path,
        *,
        limit: int | None = None,
        split: str | None = None,
        **_ignored: Any,
    ) -> Path:
        require_positive_limit(limit)
        splits = _splits(split)
        from datasets import load_dataset

        root = self.cache_dir(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        try:
            hub = load_dataset(REPO_ID)
        except Exception as exc:
            raise DatasetError(f"could not download {REPO_ID}: {exc}") from exc
        written: list[str] = []
        for name in splits:
            if name not in hub:
                raise DatasetError(f"{REPO_ID} has no split {name}")
            rows = []
            for index, row in enumerate(hub[name]):
                if limit is not None and index >= limit:
                    break
                rows.append(_record_from_hub(name, index, row))
            if not rows:
                raise DatasetError(f"{name} returned no rows")
            write_jsonl(root / f"{name}.jsonl", rows)
            written.append(name)
        write_manifest(root / "manifest.json", {"repo_id": REPO_ID, "splits": written, "limit": limit})
        return root

    def load(
        self,
        data_dir: Path,
        *,
        download: bool = False,
        limit: int | None = None,
        split: str | None = None,
        **_ignored: Any,
    ) -> DatasetBundle:
        require_positive_limit(limit)
        splits = _splits(split)
        if download and not self.cache_is_complete(data_dir, split=split):
            self.download(data_dir, limit=limit, split=split)
        records: list[dict[str, Any]] = []
        root = self.cache_dir(data_dir)
        for name in splits:
            path = root / f"{name}.jsonl"
            if not path.is_file():
                raise DatasetError(
                    f"SimBench cache is missing {path.name}. Run `benchmark fetch simbench`."
                )
            remaining = None if limit is None else limit - len(records)
            if remaining is not None and remaining <= 0:
                break
            records.extend(read_jsonl(path, limit=remaining))
        return bundle_from_records(records, source=self.source)


SIMBENCH = SimBench()


def download(data_dir: Path, **kwargs: Any) -> Path:
    return SIMBENCH.download(data_dir, **kwargs)


def load(data_dir: Path, **kwargs: Any) -> DatasetBundle:
    return SIMBENCH.load(data_dir, **kwargs)


def bundle_from_records(records: list[dict[str, Any]], *, source: str = REPO_ID) -> DatasetBundle:
    subjects: list[Subject] = []
    cases: list[Case] = []
    seen: set[str] = set()
    for record in records:
        _refuse_individual(record)
        subject, case = _subject_and_case(record)
        if subject.id not in seen:
            subjects.append(subject)
            seen.add(subject.id)
        cases.append(case)
    return DatasetBundle(
        name="simbench",
        unit="group",
        source=source,
        revision=None,
        subjects=tuple(subjects),
        cases=tuple(cases),
    )


def _record_from_hub(split: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    _refuse_individual(row)
    dataset_name = str(row.get("dataset_name") or split)
    variables = _mapping(parse_jsonish(row.get("group_prompt_variable_map")))
    template = str(row.get("group_prompt_template") or "")
    prompt = _format_template(template, variables)
    distribution = _distribution(parse_jsonish(row.get("human_answer")))
    auxiliary = parse_jsonish(row.get("auxiliary"))
    question_id = _question_id(auxiliary, index)
    options = tuple(label for label, _weight in distribution)
    return {
        "split": split,
        "dataset_name": dataset_name,
        "group_key": _group_key(dataset_name, variables),
        "group_prompt": prompt,
        "variables": variables,
        "question_id": question_id,
        "prompt": str(row.get("input_template") or "").strip(),
        "options": list(options),
        "distribution": {label: weight for label, weight in distribution},
        "group_size": row.get("group_size"),
    }


def _subject_and_case(record: dict[str, Any]) -> tuple[Subject, Case]:
    group_key = str(record.get("group_key") or "").strip()
    group_prompt = str(record.get("group_prompt") or "").strip()
    prompt = str(record.get("prompt") or "").strip()
    if not group_key or not group_prompt or not prompt:
        raise DatasetError("SimBench record needs group_key, group_prompt, and prompt")
    variables = _mapping(record.get("variables"))
    subject = Subject(
        id=group_key,
        dataset="simbench",
        unit="group",
        demographics={str(key): str(value) for key, value in variables.items()},
        group_prompt=group_prompt,
    )
    options = tuple(str(option) for option in record.get("options") or [])
    question = Question(
        id=str(record.get("question_id") or ""),
        prompt=prompt,
        options=options,
        kind="distribution",
    )
    distribution = _distribution(record.get("distribution"))
    group_size = record.get("group_size")
    gold = Gold(
        distribution=distribution,
        group_size=None if group_size is None else int(group_size),
    )
    return subject, Case(dataset="simbench", subject=subject, question=question, gold=gold)


def _refuse_individual(record: dict[str, Any]) -> None:
    """SimBench is published as aggregates. Person-level rows are out of scope."""
    if "pid" in record or "persona_text" in record or "respondent_id" in record:
        raise DatasetError(
            "SimBench loader refuses individual respondent rows. "
            "Keep the published group distribution."
        )
    answer = record.get("human_answer", record.get("distribution"))
    if answer is None:
        return
    parsed = parse_jsonish(answer)
    if isinstance(parsed, str):
        raise DatasetError("SimBench human_answer must be a group distribution, not one person's answer")


def _distribution(value: Any) -> tuple[tuple[str, float], ...]:
    parsed = parse_jsonish(value)
    if not isinstance(parsed, dict) or not parsed:
        raise DatasetError("SimBench human_answer must be a non-empty option-to-weight mapping")
    pairs: list[tuple[str, float]] = []
    for label, weight in parsed.items():
        number = float(weight)
        if number < 0:
            raise DatasetError("SimBench distribution weights cannot be negative")
        pairs.append((str(label), number))
    return tuple(pairs)


def _format_template(template: str, variables: dict[str, Any]) -> str:
    text = template
    for key, value in variables.items():
        text = text.replace("{" + str(key) + "}", str(value))
    return text.strip()


def _group_key(dataset_name: str, variables: dict[str, Any]) -> str:
    if not variables:
        return f"{dataset_name}:population"
    parts = [f"{key}={variables[key]}" for key in sorted(variables)]
    return f"{dataset_name}:" + "|".join(parts)


def _question_id(auxiliary: Any, index: int) -> str:
    if isinstance(auxiliary, dict):
        for key in ("task_id", "question_id_original", "question_id"):
            if auxiliary.get(key) not in (None, ""):
                return str(auxiliary[key])
    return f"item-{index}"


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        parsed = parse_jsonish(value)
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, dict):
        return value
    raise DatasetError("group_prompt_variable_map must be an object")


def _splits(split: str | None) -> tuple[str, ...]:
    if split is None:
        return SPLITS
    if split not in SPLITS:
        known = ", ".join(SPLITS)
        raise DatasetError(f"unknown SimBench split {split}. Use one of: {known}")
    return (split,)

