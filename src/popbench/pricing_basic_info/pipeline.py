"""Download pinned sources, build cards, and write a hash manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from popbench.pricing_basic_info.assemble import assemble_dataset, document_hashes, field_map
from popbench.pricing_basic_info.serialize import dumps, sha256_file, write_bytes
from popbench.pricing_basic_info.sources import load_bdm_table, load_twin_tables
from popbench.pricing_basic_info.spec import (
    BDM_BASIC_COLUMNS,
    BDM_DATASET,
    BDM_DESIGN_COLUMNS,
    BDM_DOI,
    BDM_OUTCOME_COLUMNS,
    BDM_SOURCE_FILE,
    BDM_STRATIFY_COLUMN,
    BDM_VERSION,
    TWIN_BASIC_COLUMNS,
    TWIN_CATALOG_FILE,
    TWIN_DATASET,
    TWIN_DEMOGRAPHIC_FILE,
    TWIN_PRICING_IDS,
    TWIN_REPO_ID,
    TWIN_REVISION,
    TWIN_STRATIFY_COLUMN,
    TWIN_WAVE4_FILE,
    TWIN_WAVE_CHUNKS,
)

BDM_CSV_URL = "https://borealisdata.ca/api/access/datafile/1035856?format=original"
BDM_README_URL = "https://borealisdata.ca/api/access/datafile/1051155?format=original"
BDM_README_NAME = "100A_README.txt"

TWIN_ELIGIBILITY = (
    "Included when pid is in the demographic label file and in wave4_Q_wave4_A "
    "with all 40 QID9 answers present. A blank demographic cell does not exclude "
    "the person. Wave 1-3 QID9 answers are an earlier administration and are not "
    "the held-out pricing outcome."
)
BDM_ELIGIBILITY = (
    "Included when pid is present and all six willingness-to-pay bids plus "
    "conclusion_purchased_product are non-blank. A blank survey cell does not "
    "exclude the person. Treatment, order, practice items, and timestamps are "
    "not basic info."
)


def run_build(root: Path, config: Mapping[str, object]) -> dict[str, object]:
    """Build both datasets. Returns the manifest object."""
    cache = root / str(config["cache_dir"])
    output = root / str(config["output_dir"])
    seed = int(config["seed"])  # type: ignore[arg-type]
    rule = config["sample_size_rule"]
    if not isinstance(rule, Mapping):
        raise ValueError("sample_size_rule must be an object")
    override = config.get("sample_size")
    if override is not None and not isinstance(override, int):
        raise ValueError("sample_size must be an integer or null")
    raw = _ensure_sources(cache / "raw")
    lock_path = output / "source-lock.json"
    lock = _source_lock(raw)
    if lock_path.is_file():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != lock:
            raise ValueError("pinned source files do not match source-lock.json")

    twin_rows, twin_outcomes, stems, audit = load_twin_tables(
        raw[TWIN_DEMOGRAPHIC_FILE],
        raw[TWIN_WAVE4_FILE],
        raw[TWIN_CATALOG_FILE],
        [raw[name] for name in TWIN_WAVE_CHUNKS],
    )
    bdm_rows, bdm_outcomes = load_bdm_table(raw[f"bdm/{BDM_SOURCE_FILE}"])
    sample_rule = {
        "margin_of_error": str(rule["margin_of_error"]),
        "proportion": str(rule["proportion"]),
        "z": str(rule["z"]),
    }
    twin_docs = assemble_dataset(
        twin_rows,
        dataset=TWIN_DATASET,
        source_file=TWIN_DEMOGRAPHIC_FILE,
        columns=TWIN_BASIC_COLUMNS,
        stratify_column=TWIN_STRATIFY_COLUMN,
        question_text=stems,
        outcomes=twin_outcomes,
        outcome_ids=TWIN_PRICING_IDS,
        forbidden_columns=TWIN_PRICING_IDS,
        seed=seed,
        sample_rule=sample_rule,
        sample_size_override=override if isinstance(override, int) else None,
        eligibility_note=TWIN_ELIGIBILITY,
    )
    bdm_docs = assemble_dataset(
        bdm_rows,
        dataset=BDM_DATASET,
        source_file=BDM_SOURCE_FILE,
        columns=BDM_BASIC_COLUMNS,
        stratify_column=BDM_STRATIFY_COLUMN,
        question_text=None,
        outcomes=bdm_outcomes,
        outcome_ids=BDM_OUTCOME_COLUMNS,
        forbidden_columns=BDM_OUTCOME_COLUMNS + BDM_DESIGN_COLUMNS,
        seed=seed,
        sample_rule=sample_rule,
        sample_size_override=override if isinstance(override, int) else None,
        eligibility_note=BDM_ELIGIBILITY,
    )
    # Assemble is pure. A second call must match before anything is written.
    twin_again = assemble_dataset(
        twin_rows,
        dataset=TWIN_DATASET,
        source_file=TWIN_DEMOGRAPHIC_FILE,
        columns=TWIN_BASIC_COLUMNS,
        stratify_column=TWIN_STRATIFY_COLUMN,
        question_text=stems,
        outcomes=twin_outcomes,
        outcome_ids=TWIN_PRICING_IDS,
        forbidden_columns=TWIN_PRICING_IDS,
        seed=seed,
        sample_rule=sample_rule,
        sample_size_override=override if isinstance(override, int) else None,
        eligibility_note=TWIN_ELIGIBILITY,
    )
    bdm_again = assemble_dataset(
        bdm_rows,
        dataset=BDM_DATASET,
        source_file=BDM_SOURCE_FILE,
        columns=BDM_BASIC_COLUMNS,
        stratify_column=BDM_STRATIFY_COLUMN,
        question_text=None,
        outcomes=bdm_outcomes,
        outcome_ids=BDM_OUTCOME_COLUMNS,
        forbidden_columns=BDM_OUTCOME_COLUMNS + BDM_DESIGN_COLUMNS,
        seed=seed,
        sample_rule=sample_rule,
        sample_size_override=override if isinstance(override, int) else None,
        eligibility_note=BDM_ELIGIBILITY,
    )
    if twin_docs != twin_again or bdm_docs != bdm_again:
        raise ValueError("assemble_dataset was not byte-stable")
    if not lock_path.is_file():
        write_bytes(lock_path, dumps(lock))

    crosswalk = {
        BDM_DATASET: field_map(
            dataset=BDM_DATASET,
            source_file=BDM_SOURCE_FILE,
            columns=BDM_BASIC_COLUMNS,
            question_text=None,
            blocked_columns=BDM_OUTCOME_COLUMNS + BDM_DESIGN_COLUMNS,
        ),
        TWIN_DATASET: field_map(
            dataset=TWIN_DATASET,
            source_file=TWIN_DEMOGRAPHIC_FILE,
            columns=TWIN_BASIC_COLUMNS,
            question_text=stems,
            blocked_columns=list(TWIN_PRICING_IDS),
        ),
    }
    audit_text = dumps(
        {
            "dataset": TWIN_DATASET,
            "held_out_outcome": "wave4_Q_wave4_A QID9_1 through QID9_40",
            "note": (
                "QID9_n is the nth pricing item shown to that respondent, not a "
                "fixed product. The product and price are the question_text on the "
                "outcome record. Wave 1-3 QID9 cells are an earlier administration."
            ),
            **audit,
        }
    )
    committed = _write_tree(output, twin_docs, bdm_docs, dumps(crosswalk), audit_text)
    cache_outcomes = _write_outcomes(cache / "outcomes", twin_docs["outcomes.jsonl"], bdm_docs["outcomes.jsonl"])
    manifest = {
        "card_version": "basic-info-v1",
        "cache_outputs": cache_outcomes,
        "committed_outputs": committed,
        "output_dir": str(config["output_dir"]),
        "seed": seed,
        "source_lock": "source-lock.json",
        "sources": {
            BDM_DATASET: {
                "doi": BDM_DOI,
                "files": [BDM_SOURCE_FILE, BDM_README_NAME],
                "license": "CC-BY-4.0",
                "version": BDM_VERSION,
            },
            TWIN_DATASET: {
                "files": [TWIN_DEMOGRAPHIC_FILE, TWIN_WAVE4_FILE, TWIN_CATALOG_FILE, *TWIN_WAVE_CHUNKS],
                "license": "cc-by-4.0",
                "repo_id": TWIN_REPO_ID,
                "revision": TWIN_REVISION,
            },
        },
    }
    manifest_text = dumps(manifest)
    write_bytes(output / "manifest.json", manifest_text)
    manifest["manifest_sha256"] = document_hashes({"manifest.json": manifest_text})["manifest.json"]
    return manifest


def _write_tree(
    output: Path,
    twin_docs: Mapping[str, str],
    bdm_docs: Mapping[str, str],
    field_map_text: str,
    audit_text: str,
) -> dict[str, dict[str, object]]:
    written: dict[str, str] = {"field_map.json": field_map_text}
    for name, text in twin_docs.items():
        if name == "outcomes.jsonl":
            continue
        written[f"{TWIN_DATASET}/{name}"] = text
    written[f"{TWIN_DATASET}/wave_audit.json"] = audit_text
    for name, text in bdm_docs.items():
        if name == "outcomes.jsonl":
            continue
        written[f"{BDM_DATASET}/{name}"] = text
    hashes: dict[str, dict[str, object]] = {}
    for name, text in sorted(written.items()):
        path = output / name
        write_bytes(path, text)
        hashes[name] = {"bytes": len(text.encode("utf-8")), "sha256": document_hashes({name: text})[name]}
    return hashes


def _write_outcomes(output: Path, twin_text: str, bdm_text: str) -> dict[str, dict[str, object]]:
    files = {
        f"{TWIN_DATASET}/outcomes.jsonl": twin_text,
        f"{BDM_DATASET}/outcomes.jsonl": bdm_text,
    }
    hashes: dict[str, dict[str, object]] = {}
    for name, text in files.items():
        path = output / name
        write_bytes(path, text)
        hashes[name] = {
            "bytes": len(text.encode("utf-8")),
            "committed": False,
            "sha256": document_hashes({name: text})[name],
        }
    return hashes


def _ensure_sources(raw: Path) -> dict[str, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in (TWIN_DEMOGRAPHIC_FILE, TWIN_WAVE4_FILE, TWIN_CATALOG_FILE, *TWIN_WAVE_CHUNKS):
        paths[name] = _hf(raw, name)
    bdm_dir = raw / "bdm"
    bdm_dir.mkdir(parents=True, exist_ok=True)
    paths[f"bdm/{BDM_SOURCE_FILE}"] = _http(BDM_CSV_URL, bdm_dir / BDM_SOURCE_FILE)
    paths[f"bdm/{BDM_README_NAME}"] = _http(BDM_README_URL, bdm_dir / BDM_README_NAME)
    return paths


def _hf(raw: Path, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=TWIN_REPO_ID,
        filename=name,
        repo_type="dataset",
        revision=TWIN_REVISION,
        local_dir=raw / "twin2k",
    )
    path = raw / "twin2k" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _http(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    import httpx

    headers = {"User-Agent": "popbench/0.2 (pricing basic-info; +https://doi.org/10.5683/SP3/CDLUEZ)"}
    last: Exception | None = None
    for _attempt in range(3):
        try:
            response = httpx.get(url, timeout=120, follow_redirects=True, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last = exc
            continue
        if response.content:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
            return dest
        last = ValueError(f"empty download: {url}")
    raise RuntimeError(f"could not download {url}: {last}")


def _source_lock(paths: Mapping[str, Path]) -> dict[str, object]:
    files = {}
    for name, path in sorted(paths.items()):
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "files": files,
        "twin2k_revision": TWIN_REVISION,
        "bdm_doi": BDM_DOI,
        "bdm_version": BDM_VERSION,
    }
