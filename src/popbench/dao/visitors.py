"""Fetch and assemble simulate-visitor sources into DAO records.

Sources:

- ACS (American Community Survey PUMS) — population portrait baselining
- ATUS (American Time Use Survey) — daily-routine baseline
- US Census population estimates — VacSim population initialization

Official Census/BLS hosts often block automated clients. Fetch tries the
published URL first, then a Wayback Machine capture, then (for ACS) the
Hugging Face ACS Income 2018 PUMS mirror.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from popbench.dao.io import write_jsonl
from popbench.dao.personas import allocate_quotas, age_band
from popbench.dao.schema import PersonaCard

USER_AGENT = "popbench/0.1 (research; visitor-dao)"

ACS_HF_REPO = "cmpatino/acs-income-2018"
ACS_HF_FILE = "data/train-00000-of-00001.parquet"
ACS_PARQUET = "acs_income_2018.parquet"

ATUS_RESP_URL = "https://www.bls.gov/tus/datafiles/atusresp-2023.zip"
ATUS_SUM_URL = "https://www.bls.gov/tus/datafiles/atussum-2023.zip"
ATUS_RESP_WAYBACK = (
    "https://web.archive.org/web/20250105023558id_/"
    "https://www.bls.gov/tus/datafiles/atusresp-2023.zip"
)
ATUS_SUM_WAYBACK = (
    "https://web.archive.org/web/20250105023621id_/"
    "https://www.bls.gov/tus/datafiles/atussum-2023.zip"
)

CENSUS_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/state/totals/NST-EST2023-ALLDATA.csv"
)
CENSUS_WAYBACK = (
    "https://web.archive.org/web/20240926184527id_/"
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/state/totals/NST-EST2023-ALLDATA.csv"
)
CENSUS_CSV = "NST-EST2023-ALLDATA.csv"

# ATUS major activity codes → routine labels (BLS activity lexicon).
ATUS_MAJOR = {
    "01": "personal_care",
    "02": "household",
    "03": "care_household",
    "05": "work",
    "06": "education",
    "11": "eating",
    "12": "leisure",
    "13": "sports",
    "14": "religious",
    "15": "volunteer",
    "18": "traveling",
}

SEX_LABEL = {1: "Male", 2: "Female"}
EDU_BANDS = (
    (1, 15, "less_than_high_school"),
    (16, 17, "high_school"),
    (18, 20, "some_college"),
    (21, 24, "bachelor_or_higher"),
)
MARITAL = {
    1: "Married",
    2: "Widowed",
    3: "Divorced",
    4: "Separated",
    5: "Never married",
}
RACE = {
    1: "White alone",
    2: "Black or African American alone",
    3: "American Indian alone",
    4: "Alaska Native alone",
    5: "American Indian and Alaska Native tribes specified",
    6: "Asian alone",
    7: "Native Hawaiian and Other Pacific Islander alone",
    8: "Some Other Race alone",
    9: "Two or More Races",
}


class VisitorError(RuntimeError):
    """Visitor source cache is missing, incomplete, or could not be built."""


@dataclass(frozen=True)
class VisitorRecord:
    """One simulate visitor assembled from ACS, ATUS, and Census margins."""

    id: str
    source: str
    persona: PersonaCard
    routine: dict[str, float]
    state: str
    state_population: int
    demographics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "persona": self.persona.to_dict(),
            "routine": self.routine,
            "state": self.state,
            "state_population": self.state_population,
            "demographics": self.demographics,
        }


def visitors_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "visitors"


def visitors_built_dir(data_dir: Path) -> Path:
    return visitors_dir(data_dir) / "v0"


def acs_dir(data_dir: Path) -> Path:
    return visitors_dir(data_dir) / "acs"


def atus_dir(data_dir: Path) -> Path:
    return visitors_dir(data_dir) / "atus"


def census_dir(data_dir: Path) -> Path:
    return visitors_dir(data_dir) / "census"


def cache_is_complete(data_dir: Path) -> bool:
    return (
        (acs_dir(data_dir) / ACS_PARQUET).is_file()
        and (atus_dir(data_dir) / "sum" / "atussum_2023.dat").is_file()
        and (census_dir(data_dir) / CENSUS_CSV).is_file()
    )


def fetch_visitors(data_dir: Path) -> dict[str, Path]:
    """Download ACS, ATUS, and Census into `data/visitors/`."""
    paths = {
        "acs": cache_acs(data_dir),
        "atus": cache_atus(data_dir),
        "census": cache_census(data_dir),
    }
    manifest = {
        "sources": {
            "acs": {
                "role": "population portrait baselining",
                "primary": "https://www.census.gov/programs-surveys/acs",
                "cache": ACS_PARQUET,
                "mirror": f"huggingface:{ACS_HF_REPO}",
                "people": 1_611_572,
                "adults_18_plus": 1_596_063,
            },
            "atus": {
                "role": "daily-routine baseline",
                "primary": "https://www.bls.gov/tus/",
                "files": ["atusresp-2023.zip", "atussum-2023.zip"],
                "respondents": 8548,
                "adults_18_plus": 8382,
            },
            "census": {
                "role": "VacSim population initialization",
                "primary": "https://www.census.gov",
                "file": CENSUS_CSV,
                "us_population_2023": 334_914_895,
                "state_rows": 52,
            },
        }
    }
    root = visitors_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def cache_acs(data_dir: Path) -> Path:
    """Cache ACS PUMS person rows used for population portraits."""
    dest = acs_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / ACS_PARQUET
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(
                repo_id=ACS_HF_REPO,
                repo_type="dataset",
                filename=ACS_HF_FILE,
                local_dir=str(dest),
            )
        except Exception as exc:
            raise VisitorError(
                f"could not download ACS mirror {ACS_HF_REPO}: {exc}. "
                "Primary ACS page: https://www.census.gov/programs-surveys/acs"
            ) from exc
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != path.resolve():
            path.write_bytes(downloaded_path.read_bytes())
    _write_source_manifest(
        dest,
        {
            "source": "American Community Survey (ACS)",
            "role": "人口画像基座化",
            "url": "https://www.census.gov/programs-surveys/acs",
            "mirror": f"{ACS_HF_REPO}/{ACS_HF_FILE}",
            "file": ACS_PARQUET,
            "bytes": path.stat().st_size,
        },
    )
    return path


def cache_atus(data_dir: Path) -> Path:
    """Cache ATUS 2023 respondent and activity-summary microdata."""
    dest = atus_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    resp_zip = dest / "atusresp-2023.zip"
    sum_zip = dest / "atussum-2023.zip"
    if not resp_zip.is_file():
        _download_first([ATUS_RESP_URL, ATUS_RESP_WAYBACK], resp_zip)
    if not sum_zip.is_file():
        _download_first([ATUS_SUM_URL, ATUS_SUM_WAYBACK], sum_zip)
    _unzip_if_needed(resp_zip, dest / "resp")
    _unzip_if_needed(sum_zip, dest / "sum")
    if not (dest / "sum" / "atussum_2023.dat").is_file():
        raise VisitorError("ATUS activity summary did not unpack atussum_2023.dat")
    _write_source_manifest(
        dest,
        {
            "source": "American Time Use Survey (ATUS)",
            "role": "作息基线",
            "url": "https://www.bls.gov/tus/",
            "files": {
                "atusresp-2023.zip": resp_zip.stat().st_size,
                "atussum-2023.zip": sum_zip.stat().st_size,
            },
        },
    )
    return dest


def cache_census(data_dir: Path) -> Path:
    """Cache Census Bureau state population estimates for VacSim init."""
    dest = census_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / CENSUS_CSV
    if not path.is_file():
        _download_first([CENSUS_URL, CENSUS_WAYBACK], path)
    frame = pd.read_csv(path)
    if "POPESTIMATE2023" not in frame.columns or "NAME" not in frame.columns:
        raise VisitorError(f"Census file is missing expected columns: {path}")
    _write_source_manifest(
        dest,
        {
            "source": "US Census population estimates",
            "role": "VacSim 人口初始化",
            "url": "https://www.census.gov",
            "file": CENSUS_CSV,
            "bytes": path.stat().st_size,
            "rows": int(len(frame)),
        },
    )
    return path


def build_visitors(data_dir: Path, n: int = 50, seed: int = 0) -> Path:
    """Assemble visitor personas, routine baselines, and VacSim margins."""
    if n < 1:
        raise VisitorError("n must be positive")
    if not cache_is_complete(data_dir):
        fetch_visitors(data_dir)
    if not cache_is_complete(data_dir):
        raise VisitorError(
            f"visitor cache incomplete under {visitors_dir(data_dir)}. "
            "Run `popbench fetch --visitors`."
        )

    acs = pd.read_parquet(acs_dir(data_dir) / ACS_PARQUET)
    atus = pd.read_csv(atus_dir(data_dir) / "sum" / "atussum_2023.dat")
    census = pd.read_csv(census_dir(data_dir) / CENSUS_CSV)
    routine_baseline = _routine_baseline(atus)
    margins = _vacsim_margins(acs, census)
    panel = _sample_visitors(acs, atus, census, n=n, seed=seed)

    out = visitors_built_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_name = f"visitors_n{n}.jsonl"
    write_jsonl(out / panel_name, [row.to_dict() for row in panel])
    (out / "routine_baseline.json").write_text(
        json.dumps(routine_baseline, indent=2) + "\n"
    )
    (out / "vacsim_margins.json").write_text(json.dumps(margins, indent=2) + "\n")
    (out / "example.md").write_text(_example_markdown(panel[0], routine_baseline, margins))
    manifest = {
        "n_people": n,
        "seed": seed,
        "region": "US",
        "language": "en",
        "panel": panel_name,
        "sources": {
            "acs": "population portrait baselining",
            "atus": "daily-routine baseline",
            "census": "VacSim population initialization",
        },
        "files": [
            panel_name,
            "routine_baseline.json",
            "vacsim_margins.json",
            "example.md",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def load_visitor_panel(path: Path) -> list[dict[str, Any]]:
    from popbench.dao.io import read_jsonl

    return read_jsonl(path)


def _sample_visitors(
    acs: pd.DataFrame,
    atus: pd.DataFrame,
    census: pd.DataFrame,
    *,
    n: int,
    seed: int,
) -> list[VisitorRecord]:
    adults = acs.loc[acs["AGEP"] >= 18].copy()
    adults["age"] = adults["AGEP"].astype(int)
    adults["sex_code"] = adults["SEX"].astype(int)
    adults["sex"] = adults["sex_code"].map(SEX_LABEL)
    adults["age_band"] = adults["age"].map(age_band)
    adults = adults.loc[adults["age_band"].notna() & adults["sex"].isin(["Female", "Male"])]
    adults["stratum"] = adults["age_band"] + "|" + adults["sex"]
    quotas = allocate_quotas(adults["stratum"].value_counts().to_dict(), n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum, count in quotas.items():
        pool = adults.index[adults["stratum"] == stratum].to_numpy()
        picked.extend(rng.choice(pool, size=count, replace=False).tolist())
    panel = adults.loc[picked].sort_values(["stratum", "AGEP"])

    states = _state_populations(census)
    state_names = list(states.keys())
    state_weights = np.array([states[name] for name in state_names], dtype=float)
    state_weights = state_weights / state_weights.sum()
    routines = _routines_by_stratum(atus)

    records: list[VisitorRecord] = []
    for index, (_, row) in enumerate(panel.iterrows()):
        state = str(rng.choice(state_names, p=state_weights))
        sex = str(row["sex"])
        band = str(row["age_band"])
        routine = routines.get(f"{band}|{sex}") or routines.get("all") or {}
        education = _edu_label(int(row["SCHL"]))
        marital = MARITAL.get(int(row["MAR"]), "Unknown")
        race = RACE.get(int(row["RAC1P"]), "Unknown")
        income_over_50k = bool(row["PINCP > 50k"])
        hours = None if pd.isna(row["WKHP"]) else float(row["WKHP"])
        text = _persona_text(
            age=int(row["age"]),
            sex=sex,
            state=state,
            education=education,
            marital=marital,
            race=race,
            income_over_50k=income_over_50k,
            work_hours=hours,
            routine=routine,
        )
        visitor_id = f"visitor-{seed}-{index:04d}"
        persona = PersonaCard(
            id=visitor_id,
            condition="acs_census_atus",
            text=text,
            age=int(row["age"]),
            age_band=band,
            sex=sex,
            state=state,
            education_level=education,
            occupation=None,
            marital_status=marital,
        )
        records.append(
            VisitorRecord(
                id=visitor_id,
                source="acs+atus+census",
                persona=persona,
                routine=routine,
                state=state,
                state_population=int(states[state]),
                demographics={
                    "age": int(row["age"]),
                    "age_band": band,
                    "sex": sex,
                    "education": education,
                    "marital_status": marital,
                    "race": race,
                    "income_over_50k": income_over_50k,
                    "work_hours": hours,
                },
            )
        )
    return records


def _routine_baseline(atus: pd.DataFrame) -> dict[str, Any]:
    overall = _mean_routine(atus)
    by_stratum: dict[str, dict[str, float]] = {}
    frame = atus.loc[atus["TEAGE"] >= 18].copy()
    frame["age_band"] = frame["TEAGE"].map(age_band)
    frame["sex"] = frame["TESEX"].map(SEX_LABEL)
    frame = frame.loc[frame["age_band"].notna() & frame["sex"].notna()]
    for (band, sex), group in frame.groupby(["age_band", "sex"]):
        by_stratum[f"{band}|{sex}"] = _mean_routine(group)
    return {
        "source": "American Time Use Survey (ATUS) 2023 activity summary",
        "url": "https://www.bls.gov/tus/",
        "unit": "mean_minutes_per_diary_day",
        "n_respondents": int(len(atus)),
        "overall": overall,
        "by_age_sex": by_stratum,
    }


def _routines_by_stratum(atus: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline = _routine_baseline(atus)
    out = dict(baseline["by_age_sex"])
    out["all"] = baseline["overall"]
    return out


def _mean_routine(frame: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for code, label in ATUS_MAJOR.items():
        cols = [
            column
            for column in frame.columns
            if column.startswith("t")
            and len(column) >= 3
            and column[1:3] == code
            and column[1:].isdigit()
        ]
        if not cols:
            continue
        values[label] = float(frame[cols].sum(axis=1).mean())
    return values


def _vacsim_margins(acs: pd.DataFrame, census: pd.DataFrame) -> dict[str, Any]:
    adults = acs.loc[acs["AGEP"] >= 18].copy()
    adults["age_band"] = adults["AGEP"].map(age_band)
    adults["sex"] = adults["SEX"].map(SEX_LABEL)
    adults["education"] = adults["SCHL"].map(lambda value: _edu_label(int(value)))
    states = _state_populations(census)
    return {
        "source": {
            "acs": "https://www.census.gov/programs-surveys/acs",
            "census": "https://www.census.gov",
        },
        "role": "VacSim population initialization",
        "n_acs_adults": int(len(adults)),
        "age_sex": _share_table(adults, ["age_band", "sex"]),
        "education": _share_table(adults, ["education"]),
        "sex": _share_table(adults, ["sex"]),
        "state_population_2023": states,
        "us_population_2023": int(
            census.loc[census["NAME"] == "United States", "POPESTIMATE2023"].iloc[0]
        ),
    }


def _state_populations(census: pd.DataFrame) -> dict[str, int]:
    states = census.loc[census["SUMLEV"] == 40, ["NAME", "POPESTIMATE2023"]]
    return {
        str(row.NAME): int(row.POPESTIMATE2023)
        for row in states.itertuples(index=False)
    }


def _share_table(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    counts = frame.groupby(columns, dropna=True).size()
    total = int(counts.sum())
    if total == 0:
        return {}
    out: dict[str, float] = {}
    for key, count in counts.items():
        label = "|".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        out[label] = float(count / total)
    return out


def _edu_label(code: int) -> str:
    for low, high, name in EDU_BANDS:
        if low <= code <= high:
            return name
    return "unknown"


def _persona_text(
    *,
    age: int,
    sex: str,
    state: str,
    education: str,
    marital: str,
    race: str,
    income_over_50k: bool,
    work_hours: float | None,
    routine: dict[str, float],
) -> str:
    income = "over $50,000" if income_over_50k else "$50,000 or less"
    hours = "unknown" if work_hours is None else f"{work_hours:.0f}"
    lines = [
        f"Age: {age}",
        f"Sex: {sex}",
        f"State: {state}",
        f"Education: {education}",
        f"Marital status: {marital}",
        f"Race: {race}",
        f"Personal income: {income}",
        f"Usual weekly work hours: {hours}",
        "",
        "Typical diary-day time use (ATUS minutes, age/sex stratum):",
    ]
    for key, minutes in sorted(routine.items(), key=lambda item: -item[1]):
        lines.append(f"- {key}: {minutes:.0f} minutes")
    return "\n".join(lines)


def _example_markdown(
    visitor: VisitorRecord,
    routine_baseline: dict[str, Any],
    margins: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Simulate visitor example",
            "",
            f"- Visitor id: `{visitor.id}`",
            f"- State (VacSim init weight): {visitor.state} "
            f"(pop {visitor.state_population:,})",
            f"- ACS demographics: age {visitor.demographics['age']}, "
            f"{visitor.demographics['sex']}, {visitor.demographics['education']}",
            f"- ATUS routine keys: {', '.join(sorted(visitor.routine))}",
            f"- US population 2023: {margins['us_population_2023']:,}",
            f"- ATUS respondents in baseline: {routine_baseline['n_respondents']}",
            "",
            "## Persona card",
            "",
            "```",
            visitor.persona.text,
            "```",
            "",
        ]
    )


def _download_first(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            payload = _http_get(url)
        except VisitorError as exc:
            errors.append(f"{url}: {exc}")
            continue
        dest.write_bytes(payload)
        return
    raise VisitorError("download failed for " + dest.name + "; " + " | ".join(errors))


def _http_get(url: str, timeout: int = 120) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise VisitorError(str(exc)) from exc
    if not payload:
        raise VisitorError(f"empty response from {url}")
    if payload[:15].lower().startswith(b"<!DOCTYPE html") or payload[:6].lower() == b"<html":
        raise VisitorError(f"HTML error page from {url}")
    return payload


def _unzip_if_needed(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)


def _write_source_manifest(dest: Path, payload: dict[str, Any]) -> None:
    (dest / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
