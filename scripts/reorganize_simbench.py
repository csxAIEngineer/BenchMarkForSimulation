#!/usr/bin/env python3
"""Reorganize SimBench into twin2K-analog layout under data/simbench."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path("/Users/shaneson/AI Agent")
SRC = ROOT / "SimBench" / "data"
DST = ROOT / "data" / "simbench"
TZ = ZoneInfo("Asia/Shanghai")

TASK_MAP = {
    "Choices13k": "decision_making",
    "MoralMachine": "decision_making",
    "MoralMachineClassic": "decision_making",
    "OpinionQA": "self_assessment",
    "OSPsychBig5": "self_assessment",
    "Afrobarometer": "self_assessment",
    "ESS": "self_assessment",
    "ISSP": "self_assessment",
    "LatinoBarometro": "self_assessment",
    "GlobalOpinionQA": "self_assessment",
    "TISP": "self_assessment",
    "ConspiracyCorr": "self_assessment",
    "DICES": "self_assessment",
    "OSPsychMACH": "self_assessment",
    "OSPsychRWAS": "self_assessment",
    "ChaosNLI": "judgment",
    "Jester": "judgment",
    "WisdomOfCrowds": "problem_solving",
    "OSPsychMGKT": "problem_solving",
    "NumberGame": "problem_solving",
}

OPTION_RE = re.compile(r"^\(([A-Za-z0-9]+)\)\s*:\s*(.*)$", re.MULTILINE)
OPTIONS_SPLIT_RE = re.compile(r"\n\s*Options:\s*\n", re.IGNORECASE)


def parse_jsonish(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return val
    return val


def sanitize_dataset(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name)


def normalize_template(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def make_qid(dataset_name: str, input_template: str) -> str:
    h = hashlib.sha1(
        f"{dataset_name}||{normalize_template(input_template)}".encode("utf-8")
    ).hexdigest()[:8]
    return f"SB_{sanitize_dataset(dataset_name)}_{h}"


def parse_options(input_template: str, answer_options=None, human_answer=None):
    stem = input_template or ""
    keys, texts = [], []

    ao = parse_jsonish(answer_options)
    if isinstance(ao, dict) and ao:
        def key_order(k):
            k = str(k)
            if len(k) == 1 and k.isalpha():
                return (0, k.upper())
            if k.isdigit():
                return (1, int(k))
            return (2, k)

        items = sorted(ao.items(), key=lambda kv: key_order(kv[0]))
        keys = [str(k) for k, _ in items]
        texts = [str(v) for _, v in items]
        parts = OPTIONS_SPLIT_RE.split(stem, maxsplit=1)
        if len(parts) == 2:
            stem = parts[0].strip()
        return stem, keys, texts, "from_answer_options"

    parts = OPTIONS_SPLIT_RE.split(stem, maxsplit=1)
    if len(parts) == 2:
        stem = parts[0].strip()
        matches = OPTION_RE.findall(parts[1])
        if matches:
            return stem, [m[0] for m in matches], [m[1].strip() for m in matches], "from_options_block"
        status = "options_block_no_match"
    else:
        matches = OPTION_RE.findall(stem)
        if matches:
            keys = [m[0] for m in matches]
            texts = [m[1].strip() for m in matches]
            m0 = re.search(r"\(([A-Za-z0-9]+)\)\s*:", stem)
            if m0:
                stem = stem[: m0.start()].strip()
            return stem, keys, texts, "from_inline_options"
        status = "no_options_block"

    ha = parse_jsonish(human_answer)
    if isinstance(ha, dict) and ha:
        keys = [str(k) for k in sorted(ha.keys(), key=lambda k: (0, str(k)) if len(str(k)) == 1 else (1, str(k)))]
        texts = list(keys)
        return stem.strip(), keys, texts, "fallback_human_answer_keys"

    return stem.strip(), [], [], f"fail_{status}"


def load_frames():
    pop = pd.read_parquet(SRC / "SimBenchPop.parquet").copy()
    grp = pd.read_parquet(SRC / "SimBenchGrouped.parquet").copy()
    pop["split"] = "pop"
    grp["split"] = "grouped"
    return pop, grp


def build_catalog(pop: pd.DataFrame, grp: pd.DataFrame):
    parse_stats = Counter()
    qid_to_entry = {}
    qid_sources = defaultdict(set)
    qid_datasets = defaultdict(set)

    for split_name, df in [("pop", pop), ("grouped", grp)]:
        for _, row in df.iterrows():
            ds = row["dataset_name"]
            tmpl = row["input_template"]
            qid = make_qid(ds, tmpl)
            ao = row["answer_options"] if "answer_options" in row.index else None
            stem, keys, texts, status = parse_options(tmpl, ao, row["human_answer"])
            parse_stats[status] += 1
            qid_sources[qid].add(split_name)
            qid_datasets[qid].add(ds)
            if qid not in qid_to_entry:
                if not keys:
                    parse_stats["empty_options"] += 1
                qid_to_entry[qid] = {
                    "QuestionID": qid,
                    "QuestionText": stem if stem else tmpl,
                    "QuestionType": "MC",
                    "Options": texts,
                    "OptionKeys": keys,
                    "Settings": {"Selector": "SAVR"},
                    "BlockName": ds,
                    "task_type": TASK_MAP.get(ds, "unknown"),
                    "source": split_name,
                    "csv_columns": [qid],
                    "n_options": len(keys),
                    "datasets": [ds],
                }
            else:
                prev = qid_to_entry[qid]
                if (not prev["Options"]) and texts:
                    prev["Options"] = texts
                    prev["OptionKeys"] = keys
                    prev["n_options"] = len(keys)

    catalog = []
    for qid, entry in qid_to_entry.items():
        srcs = qid_sources[qid]
        if srcs == {"pop", "grouped"}:
            entry["source"] = "both"
        elif "pop" in srcs:
            entry["source"] = "SimBenchPop"
        else:
            entry["source"] = "SimBenchGrouped"
        entry["datasets"] = sorted(qid_datasets[qid])
        catalog.append(entry)

    catalog.sort(key=lambda e: (e["BlockName"], e["QuestionID"]))
    return catalog, parse_stats, qid_to_entry


def row_to_case(row, split: str, index: int, qid: str):
    ds = row["dataset_name"]
    case = {
        "case_id": f"{split}:{ds}:{index:05d}",
        "question_id": qid,
        "split": split,
        "task_type": TASK_MAP.get(ds, "unknown"),
        "dataset_name": ds,
        "group_prompt_template": row.get("group_prompt_template"),
        "group_prompt_variable_map": parse_jsonish(row.get("group_prompt_variable_map", "{}")) or {},
        "input_template": row["input_template"],
        "human_answer": parse_jsonish(row["human_answer"]),
        "human_answer_unit": "percent_0_100",
        "group_size": int(row["group_size"]) if pd.notna(row["group_size"]) else None,
        "auxiliary": parse_jsonish(row.get("auxiliary", "{}")) or {},
    }
    if split == "grouped":
        for col in [
            "input_variable_map",
            "output_format",
            "answer_options",
            "wave",
            "group_prompt_variable_list",
            "num_grouping_vars",
            "grouping_keys",
            "is_allowed_vars",
        ]:
            if col not in row.index:
                continue
            val = row[col]
            if col in (
                "input_variable_map",
                "answer_options",
                "group_prompt_variable_list",
                "grouping_keys",
            ):
                case[col] = parse_jsonish(val)
            elif col == "wave":
                case[col] = None if pd.isna(val) else float(val)
            elif col == "num_grouping_vars":
                case[col] = None if pd.isna(val) else int(val)
            elif col == "is_allowed_vars":
                case[col] = bool(val) if pd.notna(val) else None
            else:
                case[col] = None if (isinstance(val, float) and pd.isna(val)) else val
    return case


def cases_and_shares(df: pd.DataFrame, split: str, qid_lookup):
    df = df.reset_index(drop=True)
    cases, share_rows = [], []
    for idx, row in df.iterrows():
        ds = row["dataset_name"]
        qid = make_qid(ds, row["input_template"])
        case = row_to_case(row, split, idx, qid)
        cases.append(case)
        ha = case["human_answer"] or {}
        if not isinstance(ha, dict):
            continue
        vals = [float(v) for v in ha.values()]
        total = sum(vals) if vals else 0.0
        is_percent = total > 1.5
        entry = qid_lookup.get(qid, {})
        keys = entry.get("OptionKeys") or []
        texts = entry.get("Options") or []
        for opt_key, raw in ha.items():
            raw_f = float(raw)
            share = raw_f / 100.0 if is_percent else raw_f
            opt_text = texts[keys.index(str(opt_key))] if str(opt_key) in keys else ""
            share_rows.append(
                {
                    "case_id": case["case_id"],
                    "question_id": qid,
                    "option_key": str(opt_key),
                    "option_text": opt_text,
                    "share": share,
                    "share_raw": raw_f,
                    "group_size": case["group_size"],
                    "dataset_name": ds,
                    "split": split,
                }
            )
    return cases, share_rows


def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_shares_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "question_id",
        "option_key",
        "option_text",
        "share",
        "share_raw",
        "group_size",
        "dataset_name",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def share_sum_issues(shares, tol=1e-3):
    by = defaultdict(float)
    for r in shares:
        by[r["case_id"]] += r["share"]
    bad = [(cid, s) for cid, s in by.items() if abs(s - 1.0) > tol]
    return len(by), bad[:10], len(bad)


def main():
    print("Loading SimBench...")
    pop, grp = load_frames()
    assert len(pop) == 7167, len(pop)
    assert len(grp) == 6343, len(grp)

    if DST.exists():
        print(f"Removing existing {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    raw_dir = DST / "raw"
    raw_dir.mkdir()
    link_modes = {}
    for name in [
        "SimBenchPop.parquet",
        "SimBenchGrouped.parquet",
        "stats.json",
        "meta_enriched.csv",
        "README.md",
    ]:
        src = SRC / name
        if src.exists():
            link_modes[name] = hardlink_or_copy(src, raw_dir / name)
    write_json(
        raw_dir / "source_pointer.json",
        {
            "original_path": str(SRC),
            "repo_id": "pitehu/SimBench",
            "paper": "https://arxiv.org/abs/2510.17516",
            "files": link_modes,
            "note": "Parquet/stats hardlinked or copied for a self-contained data/simbench; originals remain under SimBench/data.",
        },
    )
    (raw_dir / "README.md").write_text(
        "# SimBench raw snapshot\n\n"
        "Files hardlinked/copied from `/Users/shaneson/AI Agent/SimBench/data`.\n"
        "Upstream: Hugging Face `pitehu/SimBench`, paper arXiv:2510.17516.\n",
        encoding="utf-8",
    )

    print("Building catalog...")
    catalog, parse_stats, qid_lookup = build_catalog(pop, grp)
    print("Catalog size:", len(catalog))
    print("Parse stats:", dict(parse_stats))

    print("Building cases + shares...")
    pop_cases, pop_shares = cases_and_shares(pop, "pop", qid_lookup)
    grp_cases, grp_shares = cases_and_shares(grp, "grouped", qid_lookup)
    assert len(pop_cases) == 7167
    assert len(grp_cases) == 6343

    qcsv = DST / "question_catalog_and_human_response_csv"
    qcsv.mkdir(parents=True)
    write_json(qcsv / "question_catalog.json", catalog)
    write_jsonl(qcsv / "pop_cases.jsonl", pop_cases)
    write_jsonl(qcsv / "grouped_cases.jsonl", grp_cases)
    write_shares_csv(qcsv / "pop_human_shares.csv", pop_shares)
    write_shares_csv(qcsv / "grouped_human_shares.csv", grp_shares)

    n_pop_share_cases, pop_bad, n_pop_bad = share_sum_issues(pop_shares)
    n_grp_share_cases, grp_bad, n_grp_bad = share_sum_issues(grp_shares)
    catalog_ids = {e["QuestionID"] for e in catalog}
    missing_pop = [c["case_id"] for c in pop_cases if c["question_id"] not in catalog_ids]
    missing_grp = [c["case_id"] for c in grp_cases if c["question_id"] not in catalog_ids]
    empty_opts = [e["QuestionID"] for e in catalog if not e.get("Options")]

    print("Writing by_dataset...")
    by_ds = DST / "by_dataset"
    datasets = sorted(set(pop["dataset_name"]) | set(grp["dataset_name"]))
    ds_breakdown = {}
    for ds in datasets:
        ddir = by_ds / ds
        ddir.mkdir(parents=True)
        cat_ds = [e for e in catalog if e["BlockName"] == ds]
        write_json(ddir / "question_catalog.json", cat_ds)
        pop_ds = [c for c in pop_cases if c["dataset_name"] == ds]
        grp_ds = [c for c in grp_cases if c["dataset_name"] == ds]
        if pop_ds:
            write_jsonl(ddir / "pop_cases.jsonl", pop_ds)
        if grp_ds:
            write_jsonl(ddir / "grouped_cases.jsonl", grp_ds)
        man = {
            "dataset_name": ds,
            "task_type": TASK_MAP.get(ds, "unknown"),
            "n_catalog_questions": len(cat_ds),
            "n_pop_cases": len(pop_ds),
            "n_grouped_cases": len(grp_ds),
            "splits_present": [s for s, n in [("pop", len(pop_ds)), ("grouped", len(grp_ds))] if n],
        }
        write_json(ddir / "manifest.json", man)
        ds_breakdown[ds] = man

    print("Writing by_task...")
    by_task = DST / "by_task"
    task_breakdown = {}
    for task in sorted(set(TASK_MAP.values())):
        tdir = by_task / task
        tdir.mkdir(parents=True)
        cat_t = [e for e in catalog if e["task_type"] == task]
        write_json(tdir / "question_catalog.json", cat_t)
        pop_t = [c for c in pop_cases if c["task_type"] == task]
        grp_t = [c for c in grp_cases if c["task_type"] == task]
        write_jsonl(tdir / "pop_cases.jsonl", pop_t)
        write_jsonl(tdir / "grouped_cases.jsonl", grp_t)
        man = {
            "task_type": task,
            "n_catalog_questions": len(cat_t),
            "n_pop_cases": len(pop_t),
            "n_grouped_cases": len(grp_t),
            "datasets": sorted({e["BlockName"] for e in cat_t}),
            "splits_present": [s for s, n in [("pop", len(pop_t)), ("grouped", len(grp_t))] if n],
        }
        write_json(tdir / "manifest.json", man)
        task_breakdown[task] = man

    catalog_readme = f"""# SimBench question catalog（twin2K 对齐说明）

## 与 twin2K 的目录类比

| twin2K | SimBench（本目录） | 说明 |
|--------|-------------------|------|
| 个体答卷 `pid × question` | **群体分布** case（无个体） | **不会**伪造个体受访者 |
| `wave1_3_response*.csv` / `wave4_response*.csv` | `pop_cases.jsonl` + `grouped_cases.jsonl` 及对应 `*_human_shares.csv` | 两个并列数据分区，不是固定训练/测试语义 |
| `question_catalog.json` | 同名 | 每道唯一题干一条；`QuestionID` 在 Pop/Grouped 间稳定共享 |
| `by_category/<family>/` | `by_dataset/` 与 `by_task/` | 按来源数据集或任务类型切片 |

`SimBenchPop` 与 `SimBenchGrouped` 都是官方发布的评测分区：前者覆盖 20 个来源的总体人群分布；后者来自 5 个大型调查并带人口统计分组字段。二者在本包中作为**并列 sibling splits** 存放。

## human_answer 单位

- 源数据 `human_answer` 为 **0–100 的百分比**（各选项合计约 100）。
- JSONL 保留原始百分比，字段 `human_answer_unit: "percent_0_100"`。
- CSV `*_human_shares.csv` 中 `share` 已归一到 **0–1**；`share_raw` 为原始百分比。

## QuestionID

`SB_{{DatasetSanitized}}_{{sha1(dataset||normalized_input_template)[:8]}}`，跨运行稳定。

## 质量摘要（生成时）

- Pop cases: {len(pop_cases)}（期望 7167）
- Grouped cases: {len(grp_cases)}（期望 6343）
- Catalog questions: {len(catalog)}
- Option parse stats: {json.dumps(dict(parse_stats), ensure_ascii=False)}
- Empty Options in catalog: {len(empty_opts)}
- Cases missing catalog id: pop={len(missing_pop)} grouped={len(missing_grp)}
- Share sum ≠ 1 cases: pop={n_pop_bad} grouped={n_grp_bad}
"""
    (qcsv / "question_catalog_README.md").write_text(catalog_readme, encoding="utf-8")

    created = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    readme = f"""# SimBench 本地整理包（twin2K 格式对齐）

创建时间：{created}

论文：[SimBench (arXiv:2510.17516)](https://arxiv.org/abs/2510.17516) · HF：`pitehu/SimBench`

## 这是什么

把官方 SimBench 两个分区整理成与 `data/twin2k` 相近的「题库 catalog + 人类回答」布局，方便后续数字孪生/问卷模拟代码复用同一套读取方式。

**注意：** SimBench 只有**群体回答分布**，没有 twin2K 那种 `pid` 级个体答卷；本包**不会**伪造个体。

## 两个并列分区

| 文件 | 来源 | 条数 | 含义 |
|------|------|------|------|
| `pop_cases.jsonl` / `pop_human_shares.csv` | SimBenchPop | 7167 | 20 个数据集上的总体人群分布 |
| `grouped_cases.jsonl` / `grouped_human_shares.csv` | SimBenchGrouped | 6343 | 5 个大型调查上的人口统计分组分布 |

二者是 sibling splits，可按实验需要自行选用或组合；本包**不**预置训练/测试或样本内外划分。

## 目录结构

```
data/simbench/
  manifest.json
  README.md
  raw/                         # parquet + stats 快照
  question_catalog_and_human_response_csv/
    question_catalog.json
    question_catalog_README.md
    pop_cases.jsonl
    grouped_cases.jsonl
    pop_human_shares.csv
    grouped_human_shares.csv
  by_dataset/<DatasetName>/
  by_task/<task_type>/
```

## 规模

- Catalog 唯一题：{len(catalog)}
- Pop / Grouped cases：{len(pop_cases)} / {len(grp_cases)}
- 任务类型（Pop cases）：{json.dumps({k: v['n_pop_cases'] for k,v in task_breakdown.items()}, ensure_ascii=False)}
- 数据集：{', '.join(datasets)}

## 质量检查

- 行数：pop={len(pop_cases)} grouped={len(grp_cases)}
- Catalog ↔ case 映射缺失：pop={len(missing_pop)} grouped={len(missing_grp)}
- 空 Options：{len(empty_opts)}
- 选项解析：{json.dumps(dict(parse_stats), ensure_ascii=False)}
- 每 case 的 share 求和≈1：异常 pop={n_pop_bad} grouped={n_grp_bad}
  - 样例 pop bad: {pop_bad[:3]}
  - 样例 grouped bad: {grp_bad[:3]}

## 使用提示

拼提示时用 case 内 `group_prompt_template` + `group_prompt_variable_map` + `input_template`；预测选项分布后与 `human_answer`（或 shares CSV）比对即可。
"""
    (DST / "README.md").write_text(readme, encoding="utf-8")

    def fsize(p: Path):
        return p.stat().st_size if p.exists() else 0

    files_meta = {}
    for p in [
        qcsv / "question_catalog.json",
        qcsv / "question_catalog_README.md",
        qcsv / "pop_cases.jsonl",
        qcsv / "grouped_cases.jsonl",
        qcsv / "pop_human_shares.csv",
        qcsv / "grouped_human_shares.csv",
        DST / "README.md",
    ]:
        files_meta[str(p.relative_to(DST))] = {"bytes": fsize(p)}

    manifest = {
        "repo_id": "pitehu/SimBench",
        "paper_arxiv": "2510.17516",
        "paper_url": "https://arxiv.org/abs/2510.17516",
        "created_at": created,
        "timezone": "Asia/Shanghai",
        "n_pop": len(pop_cases),
        "n_grouped": len(grp_cases),
        "n_catalog_questions": len(catalog),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "task_breakdown": task_breakdown,
        "dataset_breakdown": ds_breakdown,
        "layout_note": (
            "Mirrors data/twin2k catalog + human-response layout. "
            "SimBenchPop and SimBenchGrouped are sibling partitions (group-level distributions only); "
            "no synthetic individuals; no preset train/test or OOD protocol files."
        ),
        "human_answer_unit_in_jsonl": "percent_0_100",
        "share_unit_in_csv": "probability_0_1",
        "quality": {
            "parse_stats": dict(parse_stats),
            "empty_options_count": len(empty_opts),
            "missing_catalog_pop": len(missing_pop),
            "missing_catalog_grouped": len(missing_grp),
            "share_sum_bad_pop": n_pop_bad,
            "share_sum_bad_grouped": n_grp_bad,
        },
        "files": files_meta,
        "raw_link_modes": link_modes,
        "source_path": str(SRC),
        "twin2k_analog_path": str(ROOT / "data" / "twin2k"),
    }
    write_json(DST / "manifest.json", manifest)

    # Ensure no ood_splits leftover
    ood_path = DST / "ood_splits.json"
    if ood_path.exists():
        ood_path.unlink()

    print("\n=== DONE ===")
    print("DST:", DST)
    print("catalog:", len(catalog))
    print("pop/grouped:", len(pop_cases), len(grp_cases))
    print("parse_stats:", dict(parse_stats))
    print("empty_options:", len(empty_opts))
    print("missing catalog:", len(missing_pop), len(missing_grp))
    print("share bad:", n_pop_bad, n_grp_bad)
    print("task_breakdown pop:", {k: v["n_pop_cases"] for k, v in task_breakdown.items()})
    print("task_breakdown grouped:", {k: v["n_grouped_cases"] for k, v in task_breakdown.items()})
    print("datasets:", datasets)


if __name__ == "__main__":
    main()
