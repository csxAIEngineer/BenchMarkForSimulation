"""Download and load LLM-Digital-Twin/Twin-2K-500 into the shared schema.

The cache stores real respondent ids from the Hub. This loader does not
synthesize people.
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
from popbench.datasets.twin2k_500.render import demographic_fields, held_out_questions

REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
DISPLAY_NAME = "Twin-2K-500"
CACHE_NAME = "subjects.jsonl"


class Twin2K500:
    name = "twin2k_500"
    display_name = DISPLAY_NAME
    unit = "individual"
    source = REPO_ID

    def cache_dir(self, data_dir: Path) -> Path:
        return Path(data_dir) / self.name

    def cache_file(self, data_dir: Path) -> Path:
        return self.cache_dir(data_dir) / CACHE_NAME

    def cache_is_complete(self, data_dir: Path) -> bool:
        return self.cache_file(data_dir).is_file()

    def download(self, data_dir: Path, *, limit: int | None = None, **_ignored: Any) -> Path:
        require_positive_limit(limit)
        from datasets import load_dataset

        root = self.cache_dir(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        try:
            full = load_dataset(REPO_ID, "full_persona", split="data")
            wave = load_dataset(REPO_ID, "wave_split", split="data")
        except Exception as exc:
            raise DatasetError(f"could not download {REPO_ID}: {exc}") from exc
        wave_by_pid = {_pid(row): row for row in wave}
        records = []
        for index, row in enumerate(full):
            if limit is not None and index >= limit:
                break
            pid = _pid(row)
            records.append(_record_from_hub(row, wave_by_pid.get(pid)))
        if not records:
            raise DatasetError(f"{REPO_ID} returned no respondents")
        path = self.cache_file(data_dir)
        write_jsonl(path, records)
        write_manifest(
            root / "manifest.json",
            {"repo_id": REPO_ID, "configs": ["full_persona", "wave_split"], "n_subjects": len(records)},
        )
        return path

    def load(
        self,
        data_dir: Path,
        *,
        download: bool = False,
        limit: int | None = None,
        **_ignored: Any,
    ) -> DatasetBundle:
        require_positive_limit(limit)
        if download and not self.cache_is_complete(data_dir):
            self.download(data_dir, limit=limit)
        path = self.cache_file(data_dir)
        if not path.is_file():
            raise DatasetError(
                f"Twin-2K-500 cache is missing under {self.cache_dir(data_dir)}. "
                "Run `benchmark fetch twin2k_500`."
            )
        return bundle_from_records(read_jsonl(path, limit=limit), source=self.source)


TWIN2K_500 = Twin2K500()


def download(data_dir: Path, **kwargs: Any) -> Path:
    return TWIN2K_500.download(data_dir, **kwargs)


def load(data_dir: Path, **kwargs: Any) -> DatasetBundle:
    return TWIN2K_500.load(data_dir, **kwargs)


def bundle_from_records(records: list[dict[str, Any]], *, source: str = REPO_ID) -> DatasetBundle:
    subjects: list[Subject] = []
    cases: list[Case] = []
    for record in records:
        subject, subject_cases = _subject_and_cases(record, source=source)
        subjects.append(subject)
        cases.extend(subject_cases)
    return DatasetBundle(
        name="twin2k_500",
        unit="individual",
        source=source,
        revision=None,
        subjects=tuple(subjects),
        cases=tuple(cases),
    )


def _record_from_hub(full_row: dict[str, Any], wave_row: dict[str, Any] | None) -> dict[str, Any]:
    persona_json = parse_jsonish(_field(full_row, "persona_json"))
    wave_persona = None if wave_row is None else parse_jsonish(_field(wave_row, "wave1_3_persona_json"))
    demographic_source = wave_persona if wave_persona is not None else persona_json
    demographic_text, demographics = demographic_fields(demographic_source)
    wave4 = None if wave_row is None else parse_jsonish(_field(wave_row, "wave4_Q_wave4_A"))
    questions = held_out_questions(wave4) if wave4 is not None else []
    summary = _field(full_row, "persona_summary")
    text = _field(full_row, "persona_text")
    return {
        "id": _pid(full_row),
        "demographics": demographics,
        "demographic_text": demographic_text,
        "persona_text": "" if text is None else str(text),
        "persona_summary": "" if summary is None else str(summary),
        "questions": questions,
    }


def _subject_and_cases(record: dict[str, Any], *, source: str) -> tuple[Subject, list[Case]]:
    subject_id = str(record.get("id") or "").strip()
    if not subject_id:
        raise DatasetError("Twin-2K-500 record is missing id")
    demographics = _string_map(record.get("demographics"))
    subject = Subject(
        id=subject_id,
        dataset="twin2k_500",
        unit="individual",
        demographics=demographics,
        persona_text=_optional_text(record.get("persona_text")),
        persona_summary=_optional_text(record.get("persona_summary")),
        demographic_text=_optional_text(record.get("demographic_text")),
        routine=_routine(record.get("routine")),
    )
    cases: list[Case] = []
    raw_questions = record.get("questions") or []
    if not isinstance(raw_questions, list):
        raise DatasetError(f"{subject_id} questions must be a list")
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise DatasetError(f"{subject_id} has a question that is not an object")
        question = Question(
            id=str(raw.get("id") or ""),
            prompt=str(raw.get("prompt") or ""),
            options=tuple(str(option) for option in raw.get("options") or []),
            kind=_question_kind(raw.get("kind")),
        )
        gold_answer = raw.get("gold")
        gold = None if gold_answer in (None, "") else Gold(answer=str(gold_answer))
        _reject_gold_in_prompt(question, gold)
        cases.append(Case(dataset="twin2k_500", subject=subject, question=question, gold=gold))
    return subject, cases


def _reject_gold_in_prompt(question: Question, gold: Gold | None) -> None:
    """Reject an explicit answer line. Short labels such as Yes can appear in stems."""
    if gold is None or not gold.answer:
        return
    if f"Answer: {gold.answer}" in question.prompt:
        raise DatasetError(
            f"question {question.id} contains its held-out answer. "
            "Prompts must not include gold."
        )


def _question_kind(value: Any) -> Any:
    kind = str(value or "text")
    if kind not in {"multiple_choice", "likert", "numeric", "distribution", "text"}:
        return "text"
    return kind


def _pid(row: dict[str, Any]) -> str:
    raw = _field(row, "pid", "PID", "id")
    if raw is None:
        raise DatasetError("Twin-2K row is missing pid")
    text = str(raw).strip()
    if text.lower().startswith("pid_"):
        return text[4:]
    return text


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _routine(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    routine: dict[str, float] = {}
    for key, item in value.items():
        routine[str(key)] = float(item)
    return routine

