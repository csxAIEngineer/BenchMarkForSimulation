"""Download pinned sources, build cards, and write a hash manifest.

A normal build refuses to create or replace ``source-lock.json``. Only
``--init-source-lock`` writes that file. A cached file is hashed every time;
a non-empty cache is not trusted without that check.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from popbench.pricing_basic_info.assemble import assemble_dataset, field_map
from popbench.pricing_basic_info.sample import require_int
from popbench.pricing_basic_info.serialize import dumps, sha256_file, sha256_text, write_bytes
from popbench.pricing_basic_info.sources import load_bdm_table, load_twin_tables
from popbench.pricing_basic_info.spec import (
    BDM_ADMIN_COLUMNS,
    BDM_BASIC_COLUMNS,
    BDM_DATASET,
    BDM_DESIGN_COLUMNS,
    BDM_DOI,
    BDM_OUTCOME_COLUMNS,
    BDM_SOURCE_FILE,
    BDM_STRATIFY_COLUMNS,
    BDM_VERSION,
    TWIN_BASIC_COLUMNS,
    TWIN_CATALOG_FILE,
    TWIN_DATASET,
    TWIN_DEMOGRAPHIC_FILE,
    TWIN_PRICING_IDS,
    TWIN_REPO_ID,
    TWIN_REVISION,
    TWIN_STRATIFY_COLUMNS,
    TWIN_WAVE4_FILE,
    TWIN_WAVE_CHUNKS,
)

BDM_CSV_URL = "https://borealisdata.ca/api/access/datafile/1035856?format=original"
BDM_README_URL = "https://borealisdata.ca/api/access/datafile/1051155?format=original"
BDM_README_NAME = "100A_README.txt"

TWIN_ELIGIBILITY = (
    "Included when pid is in the demographic label file and in wave4_Q_wave4_A "
    "with all 40 QID9 answers present. A blank demographic cell does not exclude "
    "the person. A missing column does. Wave 1-3 QID9 answers are an earlier "
    "administration and are not the held-out pricing outcome."
)
BDM_ELIGIBILITY = (
    "Included when pid is present and all six willingness-to-pay bids plus "
    "conclusion_purchased_product are non-blank. A blank survey cell does not "
    "exclude the person. Treatment, order, practice items, timestamps, and "
    "last_page are classified administration or design fields and are not basic info."
)


class SourceIntegrityError(RuntimeError):
    """The pinned lock is missing, or a cached file does not match it."""


def run_build(root: Path, config: Mapping[str, object], *, init_source_lock: bool = False) -> dict[str, object]:
    """Build both datasets. Returns the manifest object. Does not rewrite an existing lock."""
    cache = root / str(config["cache_dir"])
    output = root / str(config["output_dir"])
    seed = require_int("seed", config["seed"])
    assert seed is not None
    rule = config["sample_size_rule"]
    if not isinstance(rule, Mapping):
        raise ValueError("sample_size_rule must be an object")
    override = require_int("sample_size", config.get("sample_size"), allow_none=True)
    raw = _ensure_sources(cache / "raw", output / "source-lock.json", init_source_lock=init_source_lock)
    sample_rule = {
        "margin_of_error": str(rule["margin_of_error"]),
        "proportion": str(rule["proportion"]),
        "z": str(rule["z"]),
    }
    twin_rows, twin_outcomes, stems, options, audit = load_twin_tables(
        raw[TWIN_DEMOGRAPHIC_FILE],
        raw[TWIN_WAVE4_FILE],
        raw[TWIN_CATALOG_FILE],
        [raw[name] for name in TWIN_WAVE_CHUNKS],
    )
    bdm_rows, bdm_outcomes = load_bdm_table(raw[f"bdm/{BDM_SOURCE_FILE}"])
    twin_docs = _assemble_twin(twin_rows, twin_outcomes, stems, options, seed, sample_rule, override)
    bdm_docs = _assemble_bdm(bdm_rows, bdm_outcomes, seed, sample_rule, override)
    twin_again = _assemble_twin(twin_rows, twin_outcomes, stems, options, seed, sample_rule, override)
    bdm_again = _assemble_bdm(bdm_rows, bdm_outcomes, seed, sample_rule, override)
    if twin_docs != twin_again or bdm_docs != bdm_again:
        raise ValueError("assemble_dataset was not byte-stable")

    crosswalk = {
        BDM_DATASET: field_map(
            dataset=BDM_DATASET,
            source_file=BDM_SOURCE_FILE,
            columns=BDM_BASIC_COLUMNS,
            question_text=None,
            question_options=None,
            blocked_columns=(*BDM_OUTCOME_COLUMNS, *BDM_DESIGN_COLUMNS, *BDM_ADMIN_COLUMNS),
        ),
        TWIN_DATASET: field_map(
            dataset=TWIN_DATASET,
            source_file=TWIN_DEMOGRAPHIC_FILE,
            columns=TWIN_BASIC_COLUMNS,
            question_text=stems,
            question_options=options,
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
    cache_outputs = _write_outcomes(cache / "outcomes", twin_docs["outcomes.jsonl"], bdm_docs["outcomes.jsonl"])
    manifest = {
        "cache_outputs": cache_outputs,
        "card_version": "basic-info-v1",
        "committed_outputs": committed,
        "scope": (
            "Hashes name logical artifacts. They do not include output_dir, "
            "the machine path, or a timestamp. A match does not say the sample "
            "statistically matches the population on unstratified fields."
        ),
        "seed": seed,
        "source_lock_sha256": sha256_file(output / "source-lock.json"),
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
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    write_bytes(output / "manifest.json", dumps(manifest))
    return manifest


ALLOWED_UNHASHED_FILES = frozenset({"README.md"})


def verify_outputs(output: Path, *, cache: Path | None = None) -> dict[str, object]:
    """Recompute recorded hashes. Do not download or rewrite.

    The deliverable directory rejects files that are not hashed outputs,
    ``manifest.json``, ``source-lock.json``, or ``README.md``. Raw caches
    live outside that directory and are not part of this scan.

    Recorded outcome files are hashed when they are present. A fresh clone
    with none of them returns ``cache_outputs: skipped_absent`` and does
    not claim those files were checked.
    """
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise SourceIntegrityError(f"manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SourceIntegrityError("manifest is not an object")
    if "output_dir" in manifest:
        raise SourceIntegrityError("manifest records output_dir; content checks must not depend on a path")
    recorded = manifest.get("manifest_sha256")
    if recorded != _manifest_sha256(manifest):
        raise SourceIntegrityError("manifest_sha256 does not match the manifest contents")
    lock_path = output / "source-lock.json"
    if not lock_path.is_file():
        raise SourceIntegrityError("source-lock.json is missing")
    if sha256_file(lock_path) != manifest.get("source_lock_sha256"):
        raise SourceIntegrityError("source-lock.json does not match manifest source_lock_sha256")
    committed = manifest.get("committed_outputs")
    _verify_group(output, committed, label="committed")
    _reject_unregistered_files(output, committed)
    cache_report = _verify_cache_outputs(cache, manifest.get("cache_outputs"))
    return {
        "cache_outputs": cache_report["status"],
        "cache_outputs_checked": cache_report["checked"],
        "cache_outputs_skipped": cache_report["skipped"],
        "committed_outputs": "verified",
        "extra_files": "none",
    }


def _verify_group(root: Path, group: object, *, label: str) -> None:
    if not isinstance(group, dict):
        raise SourceIntegrityError(f"{label} outputs are missing from the manifest")
    for name, meta in group.items():
        if not isinstance(meta, dict):
            raise SourceIntegrityError(f"{label} entry {name} is not an object")
        path = root / str(name)
        if not path.is_file():
            raise SourceIntegrityError(f"{label} file is missing: {name}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != meta.get("sha256") or size != meta.get("bytes"):
            raise SourceIntegrityError(f"{label} hash mismatch: {name}")


def _reject_unregistered_files(output: Path, committed: object) -> None:
    if not isinstance(committed, dict):
        raise SourceIntegrityError("committed outputs are missing from the manifest")
    allowed = set(committed) | {"manifest.json", "source-lock.json"} | set(ALLOWED_UNHASHED_FILES)
    extras = []
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(output).as_posix()
        if name not in allowed:
            extras.append(name)
    if extras:
        raise SourceIntegrityError("unregistered file in the deliverable directory: " + ", ".join(sorted(extras)))


def _verify_cache_outputs(cache: Path | None, group: object) -> dict[str, object]:
    if not isinstance(group, dict) or not group:
        return {"checked": [], "skipped": [], "status": "skipped_absent"}
    if cache is None:
        return {"checked": [], "skipped": sorted(str(name) for name in group), "status": "skipped_absent"}
    checked: list[str] = []
    skipped: list[str] = []
    for name, meta in sorted(group.items()):
        path = cache / "outcomes" / str(name)
        if not path.is_file():
            skipped.append(str(name))
            continue
        if not isinstance(meta, dict):
            raise SourceIntegrityError(f"cache entry {name} is not an object")
        digest = sha256_file(path)
        if digest != meta.get("sha256") or path.stat().st_size != meta.get("bytes"):
            raise SourceIntegrityError(f"cache hash mismatch: {name}")
        checked.append(str(name))
    if checked and skipped:
        status = "partial"
    elif checked:
        status = "verified"
    else:
        status = "skipped_absent"
    return {"checked": checked, "skipped": skipped, "status": status}


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_text(dumps(body))


def _assemble_twin(rows, outcomes, stems, options, seed, sample_rule, override):
    return assemble_dataset(
        rows,
        dataset=TWIN_DATASET,
        source_file=TWIN_DEMOGRAPHIC_FILE,
        columns=TWIN_BASIC_COLUMNS,
        stratify_columns=TWIN_STRATIFY_COLUMNS,
        question_text=stems,
        question_options=options,
        outcomes=outcomes,
        outcome_ids=TWIN_PRICING_IDS,
        forbidden_columns=TWIN_PRICING_IDS,
        allowed_columns=tuple(rows[0]),
        seed=seed,
        sample_rule=sample_rule,
        sample_size_override=override,
        eligibility_note=TWIN_ELIGIBILITY,
    )


def _assemble_bdm(rows, outcomes, seed, sample_rule, override):
    return assemble_dataset(
        rows,
        dataset=BDM_DATASET,
        source_file=BDM_SOURCE_FILE,
        columns=BDM_BASIC_COLUMNS,
        stratify_columns=BDM_STRATIFY_COLUMNS,
        question_text=None,
        outcomes=outcomes,
        outcome_ids=BDM_OUTCOME_COLUMNS,
        forbidden_columns=(*BDM_OUTCOME_COLUMNS, *BDM_DESIGN_COLUMNS, *BDM_ADMIN_COLUMNS),
        allowed_columns=tuple(rows[0]),
        seed=seed,
        sample_rule=sample_rule,
        sample_size_override=override,
        eligibility_note=BDM_ELIGIBILITY,
        reported_groupings=("survey_q01_age",),
    )


def _write_tree(output, twin_docs, bdm_docs, field_map_text, audit_text):
    written = {"field_map.json": field_map_text}
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
        payload = text.encode("utf-8")
        hashes[name] = {"bytes": len(payload), "sha256": sha256_text(text)}
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
        payload = text.encode("utf-8")
        hashes[name] = {"bytes": len(payload), "committed": False, "sha256": sha256_text(text)}
    return hashes


def _ensure_sources(raw: Path, lock_path: Path, *, init_source_lock: bool) -> dict[str, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    lock = None if init_source_lock or not lock_path.is_file() else json.loads(lock_path.read_text(encoding="utf-8"))
    if not init_source_lock and lock is None:
        raise SourceIntegrityError(
            "source-lock.json is missing. Refusing to create one during a normal build. "
            "A first clone uses the committed lock. --init-source-lock is the only writer."
        )
    paths: dict[str, Path] = {}
    twin_root = raw / "twin2k"
    for name in (TWIN_DEMOGRAPHIC_FILE, TWIN_WAVE4_FILE, TWIN_CATALOG_FILE, *TWIN_WAVE_CHUNKS):
        paths[name] = _materialize(
            twin_root / name,
            kind="hf",
            remote=name,
            lock=lock,
            lock_key=name,
            hf_root=twin_root,
        )
    bdm_dir = raw / "bdm"
    paths[f"bdm/{BDM_SOURCE_FILE}"] = _materialize(
        bdm_dir / BDM_SOURCE_FILE,
        kind="http",
        remote=BDM_CSV_URL,
        lock=lock,
        lock_key=f"bdm/{BDM_SOURCE_FILE}",
        hf_root=None,
    )
    paths[f"bdm/{BDM_README_NAME}"] = _materialize(
        bdm_dir / BDM_README_NAME,
        kind="http",
        remote=BDM_README_URL,
        lock=lock,
        lock_key=f"bdm/{BDM_README_NAME}",
        hf_root=None,
    )
    fresh = _source_lock(paths)
    if init_source_lock:
        write_bytes(lock_path, dumps(fresh))
        return paths
    if fresh != lock:
        raise SourceIntegrityError("pinned source files do not match source-lock.json. The lock was not updated.")
    return paths


def _materialize(
    path: Path,
    *,
    kind: str,
    remote: str,
    lock: Mapping[str, object] | None,
    lock_key: str,
    hf_root: Path | None,
) -> Path:
    if path.is_file() and path.stat().st_size == 0:
        raise SourceIntegrityError(f"cached file is empty: {lock_key}")
    if not path.is_file():
        if kind == "hf":
            if hf_root is None:
                raise SourceIntegrityError(f"missing download root for {lock_key}")
            _hf(hf_root, remote)
        else:
            _http(remote, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise SourceIntegrityError(f"source file is missing after fetch: {lock_key}")
    if lock is not None:
        files = lock.get("files")
        if not isinstance(files, dict) or lock_key not in files:
            raise SourceIntegrityError(f"source-lock.json has no entry for {lock_key}")
        expected = files[lock_key]
        digest = sha256_file(path)
        if (
            not isinstance(expected, dict)
            or digest != expected.get("sha256")
            or path.stat().st_size != expected.get("bytes")
        ):
            raise SourceIntegrityError(
                f"cached {lock_key} does not match source-lock.json. "
                "The cache was not replaced and the lock was not updated."
            )
    return path


def _hf(local_root: Path, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=TWIN_REPO_ID,
        filename=name,
        repo_type="dataset",
        revision=TWIN_REVISION,
        local_dir=local_root,
    )
    path = local_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _http(url: str, dest: Path) -> Path:
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
    raise SourceIntegrityError(f"could not download {url}: {last}")


def _source_lock(paths: Mapping[str, Path]) -> dict[str, object]:
    files = {}
    for name, path in sorted(paths.items()):
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "bdm_doi": BDM_DOI,
        "bdm_version": BDM_VERSION,
        "files": files,
        "twin2k_revision": TWIN_REVISION,
    }
