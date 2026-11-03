"""Download LLM-Digital-Twin/Twin-2K-500-Mega-Study study splits.

Each row is a real Twin-2K participant in one outcome study. Persona text is
not in this dataset; ``attach_twin2k`` copies it from a Twin-2K-500 bundle.
Human answers stay on ``Gold.payload`` and are not copied into the question.
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

REPO_ID = "LLM-Digital-Twin/Twin-2K-500-Mega-Study"
DISPLAY_NAME = "Twin-2K-500-Mega-Study"

STUDIES = (
    "accuracy_nudges",
    "affective_priming",
    "consumer_minimalism",
    "context_effects",
    "default_eric",
    "digital_certification",
    "hiring_algorithms",
    "idea_evaluation",
    "idea_generation",
    "infotainment",
    "junk_fees",
    "obedient_twins",
    "preference_redistribution",
    "privacy",
    "promiscuous_donors",
    "quantitative_intuition",
    "recommendation_algorithms",
    "story_beliefs",
    "targeting_fairness",
)


class Twin2KMegaStudy:
    name = "twin2k_mega_study"
    display_name = DISPLAY_NAME
    unit = "individual"
    source = REPO_ID

    def cache_dir(self, data_dir: Path) -> Path:
        return Path(data_dir) / self.name

    def cache_is_complete(self, data_dir: Path, *, study: str | None = None) -> bool:
        root = self.cache_dir(data_dir)
        names = (study,) if study else STUDIES
        return all((root / f"{name}.jsonl").is_file() for name in names)

    def download(
        self,
        data_dir: Path,
        *,
        limit: int | None = None,
        study: str | None = None,
        **_ignored: Any,
    ) -> Path:
        require_positive_limit(limit)
        studies = _studies(study)
        from datasets import load_dataset

        root = self.cache_dir(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        try:
            hub = load_dataset(REPO_ID)
        except Exception as exc:
            raise DatasetError(f"could not download {REPO_ID}: {exc}") from exc
        written: list[str] = []
        for name in studies:
            if name not in hub:
                raise DatasetError(f"{REPO_ID} has no study split {name}")
            split = hub[name]
            rows = []
            for index, row in enumerate(split):
                if limit is not None and index >= limit:
                    break
                rows.append(_record_from_hub(name, row))
            if not rows:
                raise DatasetError(f"{name} returned no rows")
            write_jsonl(root / f"{name}.jsonl", rows)
            written.append(name)
        write_manifest(
            root / "manifest.json",
            {"repo_id": REPO_ID, "studies": written, "limit": limit},
        )
        return root

    def load(
        self,
        data_dir: Path,
        *,
        download: bool = False,
        limit: int | None = None,
        study: str | None = None,
        **_ignored: Any,
    ) -> DatasetBundle:
        require_positive_limit(limit)
        studies = _studies(study)
        if download and not self.cache_is_complete(data_dir, study=study):
            self.download(data_dir, limit=limit, study=study)
        records: list[dict[str, Any]] = []
        root = self.cache_dir(data_dir)
        for name in studies:
            path = root / f"{name}.jsonl"
            if not path.is_file():
                raise DatasetError(
                    f"Mega-Study cache is missing {path.name}. "
                    "Run `benchmark fetch twin2k_mega_study`."
                )
            remaining = None if limit is None else limit - len(records)
            if remaining is not None and remaining <= 0:
                break
            records.extend(read_jsonl(path, limit=remaining))
        return bundle_from_records(records, source=self.source)


TWIN2K_MEGA_STUDY = Twin2KMegaStudy()


def download(data_dir: Path, **kwargs: Any) -> Path:
    return TWIN2K_MEGA_STUDY.download(data_dir, **kwargs)


def load(data_dir: Path, **kwargs: Any) -> DatasetBundle:
    return TWIN2K_MEGA_STUDY.load(data_dir, **kwargs)


def bundle_from_records(records: list[dict[str, Any]], *, source: str = REPO_ID) -> DatasetBundle:
    subjects: list[Subject] = []
    cases: list[Case] = []
    seen: set[str] = set()
    for record in records:
        subject, case = _subject_and_case(record)
        if subject.id not in seen:
            subjects.append(subject)
            seen.add(subject.id)
        cases.append(case)
    return DatasetBundle(
        name="twin2k_mega_study",
        unit="individual",
        source=source,
        revision=None,
        subjects=tuple(subjects),
        cases=tuple(cases),
    )


def attach_twin2k(bundle: DatasetBundle, twin: DatasetBundle) -> DatasetBundle:
    """Copy Twin-2K persona fields onto Mega-Study subjects with the same pid.

    Returns a new bundle. Subjects with no Twin-2K match keep empty persona fields.
    """
    if bundle.name != "twin2k_mega_study":
        raise DatasetError("attach_twin2k expects a Mega-Study bundle")
    if twin.name != "twin2k_500":
        raise DatasetError("attach_twin2k expects a Twin-2K-500 bundle")
    by_pid = {_canonical_pid(subject.id): subject for subject in twin.subjects}
    subjects: dict[str, Subject] = {}
    cases: list[Case] = []
    for case in bundle.cases:
        match = by_pid.get(_canonical_pid(case.subject.id))
        subject = case.subject if match is None else _with_persona(case.subject, match)
        subjects[subject.id] = subject
        cases.append(
            Case(dataset=case.dataset, subject=subject, question=case.question, gold=case.gold)
        )
    ordered = tuple(subjects[subject_id] for subject_id in dict.fromkeys(item.subject.id for item in cases))
    return DatasetBundle(
        name=bundle.name,
        unit=bundle.unit,
        source=bundle.source,
        revision=bundle.revision,
        subjects=ordered,
        cases=tuple(cases),
    )


def _record_from_hub(study: str, row: dict[str, Any]) -> dict[str, Any]:
    pid = _canonical_pid(str(_field(row, "PID", "pid", "id") or ""))
    if not pid:
        raise DatasetError(f"{study} row is missing PID")
    survey_text = str(_field(row, "survey_text") or "").strip()
    if not survey_text:
        raise DatasetError(f"{study} pid {pid} is missing survey_text")
    gold_raw = _field(row, "survey_json_with_human_response")
    return {
        "id": pid,
        "study": study,
        "survey_text": survey_text,
        "gold_payload": None if gold_raw is None else _payload_text(gold_raw),
    }


def _subject_and_case(record: dict[str, Any]) -> tuple[Subject, Case]:
    subject_id = _canonical_pid(str(record.get("id") or ""))
    study = str(record.get("study") or "")
    if not subject_id or not study:
        raise DatasetError("Mega-Study record needs id and study")
    if study not in STUDIES:
        raise DatasetError(f"unknown Mega-Study study: {study}")
    prompt = str(record.get("survey_text") or "").strip()
    if not prompt:
        raise DatasetError(f"{study} {subject_id} is missing survey_text")
    payload = record.get("gold_payload")
    if isinstance(payload, str) and payload and payload in prompt:
        raise DatasetError(f"{study} {subject_id} copied human answers into the question text")
    subject = Subject(
        id=subject_id,
        dataset="twin2k_mega_study",
        unit="individual",
        demographics={},
        persona_text=_optional(record.get("persona_text")),
        persona_summary=_optional(record.get("persona_summary")),
        demographic_text=_optional(record.get("demographic_text")),
    )
    question = Question(id=f"{study}:{subject_id}", prompt=prompt, kind="text")
    gold = None if not payload else Gold(payload=str(payload))
    return subject, Case(dataset="twin2k_mega_study", subject=subject, question=question, gold=gold)


def _with_persona(subject: Subject, twin: Subject) -> Subject:
    return Subject(
        id=subject.id,
        dataset=subject.dataset,
        unit=subject.unit,
        demographics=dict(twin.demographics),
        persona_text=twin.persona_text,
        persona_summary=twin.persona_summary,
        demographic_text=twin.demographic_text,
        routine=None if twin.routine is None else dict(twin.routine),
    )


def _studies(study: str | None) -> tuple[str, ...]:
    if study is None:
        return STUDIES
    if study not in STUDIES:
        raise DatasetError(f"unknown Mega-Study study: {study}")
    return (study,)


def _canonical_pid(value: str) -> str:
    text = value.strip()
    if text.lower().startswith("pid_"):
        return text[4:]
    return text


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _payload_text(value: Any) -> str:
    parsed = parse_jsonish(value)
    if isinstance(parsed, str):
        return parsed
    import json

    return json.dumps(parsed, ensure_ascii=False)


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
