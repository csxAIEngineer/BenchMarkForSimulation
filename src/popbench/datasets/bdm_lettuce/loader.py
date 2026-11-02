"""Download and load the Canadian BDM lettuce willingness-to-pay survey.

Source: Gibson and Li, Borealis, https://doi.org/10.5683/SP3/CDLUEZ (CC-BY 4.0).
Each row is one real participant. Bids stay on ``Gold`` and are not copied
into the persona.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from popbench.datasets.bdm_lettuce.fields import (
    OUTCOME_COLUMNS,
    SURVEY_FIELDS,
    WTP_COLUMNS,
    WTP_PRODUCTS,
    purchase_prompt,
    render_background,
    wtp_prompt,
)
from popbench.datasets.io import (
    DatasetError,
    read_jsonl,
    require_positive_limit,
    write_jsonl,
    write_manifest,
)
from popbench.schema import Case, DatasetBundle, Gold, Question, Subject

REPO_PAGE = "https://doi.org/10.5683/SP3/CDLUEZ"
DISPLAY_NAME = "Canadian BDM lettuce WTP"
PUBLISHED_N = 610
CACHE_NAME = "respondents.jsonl"

CSV_NAME = "20240723_VFExperiment_01.csv"
README_NAME = "100A_README.txt"
DOCX_NAME = "DemographicQuestionnaire.docx"

# Dataverse file ids from the dataset API. format=original keeps the CSV.
FILES: tuple[tuple[str, str], ...] = (
    (CSV_NAME, "https://borealisdata.ca/api/access/datafile/1035856?format=original"),
    (README_NAME, "https://borealisdata.ca/api/access/datafile/1051155?format=original"),
    (DOCX_NAME, "https://borealisdata.ca/api/access/datafile/1051154?format=original"),
)


class BdmLettuce:
    name = "bdm_lettuce"
    display_name = DISPLAY_NAME
    unit = "individual"
    source = REPO_PAGE

    def cache_dir(self, data_dir: Path) -> Path:
        return Path(data_dir) / self.name

    def cache_file(self, data_dir: Path) -> Path:
        return self.cache_dir(data_dir) / CACHE_NAME

    def cache_is_complete(self, data_dir: Path) -> bool:
        root = self.cache_dir(data_dir)
        return self.cache_file(data_dir).is_file() and (root / CSV_NAME).is_file()

    def download(self, data_dir: Path, *, limit: int | None = None, **_ignored: Any) -> Path:
        require_positive_limit(limit)
        root = self.cache_dir(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        saved: dict[str, int] = {}
        skipped: dict[str, str] = {}
        csv_bytes = b""
        for name, url in FILES:
            try:
                payload = _http_get(url)
            except DatasetError as exc:
                if name == CSV_NAME:
                    raise
                skipped[name] = str(exc)
                continue
            (root / name).write_bytes(payload)
            saved[name] = len(payload)
            if name == CSV_NAME:
                csv_bytes = payload
        if not csv_bytes:
            raise DatasetError(f"missing {CSV_NAME} after download")
        rows = parse_csv(csv_bytes)
        if limit is None and len(rows) != PUBLISHED_N:
            raise DatasetError(
                f"{CSV_NAME} has {len(rows)} respondents, expected {PUBLISHED_N}"
            )
        chosen = rows if limit is None else rows[:limit]
        if not chosen:
            raise DatasetError(f"{CSV_NAME} has no respondent rows")
        records = [record_from_row(row) for row in chosen]
        path = self.cache_file(data_dir)
        write_jsonl(path, records)
        write_manifest(
            root / "manifest.json",
            {
                "doi": "10.5683/SP3/CDLUEZ",
                "license": "CC-BY-4.0",
                "files": saved,
                "skipped_files": skipped,
                "n_csv_rows": len(rows),
                "n_subjects": len(records),
                "persona": "structured demographics and consumer habits, not a narrative persona",
            },
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
                f"Canadian BDM cache is missing under {self.cache_dir(data_dir)}. "
                "Run `benchmark fetch bdm_lettuce`."
            )
        return bundle_from_records(read_jsonl(path, limit=limit), source=self.source)


BDM_LETTUCE = BdmLettuce()


def download(data_dir: Path, **kwargs: Any) -> Path:
    return BDM_LETTUCE.download(data_dir, **kwargs)


def load(data_dir: Path, **kwargs: Any) -> DatasetBundle:
    return BDM_LETTUCE.load(data_dir, **kwargs)


def parse_csv(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "pid" not in reader.fieldnames:
        raise DatasetError(f"{CSV_NAME} is missing a pid column")
    required = (*WTP_COLUMNS, "survey_q01_age")
    missing = [name for name in required if name not in reader.fieldnames]
    if missing:
        raise DatasetError(f"{CSV_NAME} is missing columns: {', '.join(missing)}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in reader:
        pid = str(raw.get("pid") or "").strip()
        if not pid:
            raise DatasetError("BDM row is missing pid")
        if pid in seen:
            raise DatasetError(f"duplicate BDM pid: {pid}")
        seen.add(pid)
        rows.append({key: "" if value is None else str(value).strip() for key, value in raw.items()})
    return rows


def record_from_row(row: dict[str, str]) -> dict[str, Any]:
    demographics = {}
    for column, key, _label in SURVEY_FIELDS:
        value = row.get(column, "").strip()
        if value:
            demographics[key] = value
    for column in OUTCOME_COLUMNS:
        demographics.pop(column, None)
    questions = []
    for column, product in WTP_PRODUCTS:
        bid = row.get(column, "").strip()
        if not bid:
            continue
        questions.append(
            {
                "id": column,
                "prompt": wtp_prompt(product),
                "options": [],
                "kind": "numeric",
                "gold": bid,
            }
        )
    purchased = row.get("conclusion_purchased_product", "").strip()
    if purchased:
        questions.append(
            {
                "id": "conclusion_purchased_product",
                "prompt": purchase_prompt(
                    row.get("conclusion_selected_product", ""),
                    row.get("conclusion_generated_price", ""),
                ),
                "options": ["TRUE", "FALSE"],
                "kind": "multiple_choice",
                "gold": purchased,
            }
        )
    return {
        "id": row["pid"].strip(),
        "demographics": demographics,
        "demographic_text": render_background(demographics),
        "questions": questions,
    }


def bundle_from_records(records: list[dict[str, Any]], *, source: str = REPO_PAGE) -> DatasetBundle:
    subjects: list[Subject] = []
    cases: list[Case] = []
    for record in records:
        subject, subject_cases = _subject_and_cases(record)
        subjects.append(subject)
        cases.extend(subject_cases)
    return DatasetBundle(
        name="bdm_lettuce",
        unit="individual",
        source=source,
        revision="V1",
        subjects=tuple(subjects),
        cases=tuple(cases),
    )


def _subject_and_cases(record: dict[str, Any]) -> tuple[Subject, list[Case]]:
    subject_id = str(record.get("id") or "").strip()
    if not subject_id:
        raise DatasetError("BDM record is missing id")
    demographics = _string_map(record.get("demographics"))
    if any(key in OUTCOME_COLUMNS for key in demographics):
        raise DatasetError(f"{subject_id} stored an outcome column on the persona")
    text = render_background(demographics)
    for question in record.get("questions") or []:
        if isinstance(question, dict) and _gold_leaked(str(question.get("gold") or ""), text):
            raise DatasetError(f"{subject_id} copied a held-out answer into the persona")
    subject = Subject(
        id=subject_id,
        dataset="bdm_lettuce",
        unit="individual",
        demographics=demographics,
        demographic_text=text or None,
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
            kind=_kind(raw.get("kind")),
        )
        gold_answer = raw.get("gold")
        gold = None if gold_answer in (None, "") else Gold(answer=str(gold_answer))
        if gold and gold.answer and f"Answer: {gold.answer}" in question.prompt:
            raise DatasetError(f"question {question.id} contains its held-out answer")
        cases.append(Case(dataset="bdm_lettuce", subject=subject, question=question, gold=gold))
    return subject, cases


def _gold_leaked(gold: str, text: str) -> bool:
    """A short token such as 2 can appear in an age. Flag only a distinctive bid."""
    if not gold or not text:
        return False
    if any(mark in gold for mark in (".", "$")):
        return gold in text
    return False


def _kind(value: Any) -> Any:
    kind = str(value or "text")
    if kind not in {"multiple_choice", "likert", "numeric", "distribution", "text"}:
        return "text"
    return kind


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if str(item).strip()}


def _http_get(url: str) -> bytes:
    import httpx

    headers = {"User-Agent": "popbench/0.2 (dataset cache; +https://doi.org/10.5683/SP3/CDLUEZ)"}
    last: Exception | None = None
    for _attempt in range(3):
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last = exc
            continue
        if response.content:
            return response.content
        last = DatasetError(f"download was empty: {url}")
    raise DatasetError(f"could not download {url}: {last}")
