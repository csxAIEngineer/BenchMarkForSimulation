"""Ask Nemotron personas every closed-ended Twin-2K item.

The paper's "~500 questions" counts some multi-row tasks as one question and
includes open-ended items. This runner asks each scorable closed-ended answer
(matrix row, single choice, number, slider) and skips free text.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from popbench.interview import InterviewError, allocate_quotas
from popbench.metrics import jensen_shannon, total_variation_distance

FAMILY_ZH = {
    "demographics": "人口学",
    "personality": "人格与价值观",
    "cognitive": "认知",
    "economic": "经济偏好",
    "heuristics": "启发式",
    "experiments": "行为实验",
    "pricing": "购买",
}


def survey_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "interview" / "full"


def family_for(block: str) -> str:
    name = block.strip()
    if name == "Demographics":
        return "demographics"
    if name == "Personality":
        return "personality"
    if name == "Cognitive tests":
        return "cognitive"
    if name.startswith("Economic"):
        return "economic"
    if name.startswith("Product"):
        return "pricing"
    if name == "Non-experimental heuristics and biases":
        return "heuristics"
    return "experiments"


def is_open_entry(entry: dict[str, Any]) -> bool:
    if entry.get("is_descriptive") or entry.get("QuestionType") == "DB":
        return True
    settings = entry.get("Settings") or {}
    selector = settings.get("Selector")
    content = settings.get("ContentType")
    if entry.get("QuestionType") == "TE" and selector in {"ML", "FORM"}:
        return True
    if entry.get("QuestionType") == "TE" and selector == "SL" and content != "ValidNumber":
        return True
    return False


ASK_N = 500


def build_closed_items(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One scorable answer per row, in catalog order."""
    items: list[dict[str, Any]] = []
    for entry in catalog:
        if is_open_entry(entry):
            continue
        items.extend(_expand(entry))
    if len(items) < 400:
        raise InterviewError(f"expected several hundred closed items, found {len(items)}")
    for turn, item in enumerate(items, start=1):
        item["turn"] = turn
    return items


def cap_items(items: list[dict[str, Any]], n: int = ASK_N) -> list[dict[str, Any]]:
    """Keep the first n items, and finish the matrix question that contains item n."""
    if len(items) <= n:
        return items
    question = _question_key(items[n - 1]["item_id"])
    end = n
    while end < len(items) and _question_key(items[end]["item_id"]) == question:
        end += 1
    kept = items[:end]
    for turn, item in enumerate(kept, start=1):
        item["turn"] = turn
    return kept


def _question_key(item_id: str) -> str:
    head, sep, tail = item_id.rpartition("_")
    if sep and tail.isdigit():
        return head
    return item_id


def _expand(entry: dict[str, Any]) -> list[dict[str, Any]]:
    question_type = entry.get("QuestionType")
    selector = (entry.get("Settings") or {}).get("Selector")
    if question_type == "Matrix":
        return _matrix_items(entry)
    if question_type == "Slider":
        return _slider_items(entry)
    if selector in {"MAVR", "MAHR"}:
        return [_multi_as_choice(entry)]
    if question_type == "TE":
        return [_numeric_item(entry, entry["csv_columns"][0], _stem(entry))]
    return [_choice_item(entry, entry["csv_columns"][0], _stem(entry), list(entry.get("Options") or []))]


def _matrix_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = entry.get("Rows") or []
    columns = entry.get("csv_columns") or []
    if len(rows) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} rows and csv columns differ")
    selector = (entry.get("Settings") or {}).get("Selector")
    scale = list(entry.get("Columns") or [])
    items = []
    stem = _stem(entry)
    for statement, column in zip(rows, columns):
        if selector == "Bipolar":
            left, right = statement.split(":", 1)
            options = [left.strip(), right.strip()]
            prompt = f"{stem}\n\nChoose one.\nLeft: {options[0]}\nRight: {options[1]}"
        else:
            options = scale
            prompt = f"{stem}\n\n{statement}"
        items.append(_choice_item(entry, column, prompt, options))
    return items


def _slider_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    columns = entry.get("csv_columns") or []
    statements = entry.get("Statements") or []
    if statements and len(statements) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} slider statements and columns differ")
    bounds = entry.get("Range") or {}
    low = bounds.get("Min", 0)
    high = bounds.get("Max", 100)
    stem = _stem(entry)
    items = []
    for index, column in enumerate(columns):
        statement = statements[index].strip() if statements else ""
        extra = f"\n\n{statement}" if statement else ""
        prompt = f"{stem}{extra}\n\nReply with a number from {low} to {high}."
        items.append(_numeric_item(entry, column, prompt))
    return items


def _multi_as_choice(entry: dict[str, Any]) -> dict[str, Any]:
    options = list(entry.get("Options") or [])
    columns = list(entry.get("csv_columns") or [])
    if len(options) != len(columns):
        raise InterviewError(f"{entry['QuestionID']} options and columns differ")
    stem = _stem(entry) or "Which statement fits you best?"
    item = _choice_item(entry, columns[0], stem, options)
    item["item_id"] = entry["QuestionID"]
    item["encoding"] = "first_selected"
    item["columns"] = columns
    return item


def _choice_item(
    entry: dict[str, Any],
    column: str,
    prompt: str,
    options: list[str],
) -> dict[str, Any]:
    if len(options) < 2:
        raise InterviewError(f"{column} needs at least two options")
    return {
        "item_id": column,
        "column": column,
        "columns": [column],
        "encoding": "single",
        "kind": "choice",
        "family": family_for(entry.get("BlockName") or ""),
        "block": (entry.get("BlockName") or "").strip(),
        "prompt": prompt.strip(),
        "options": options,
    }


def _numeric_item(entry: dict[str, Any], column: str, prompt: str) -> dict[str, Any]:
    return {
        "item_id": column,
        "column": column,
        "columns": [column],
        "encoding": "number",
        "kind": "numeric",
        "family": family_for(entry.get("BlockName") or ""),
        "block": (entry.get("BlockName") or "").strip(),
        "prompt": prompt.strip(),
        "options": [],
    }


def _stem(entry: dict[str, Any]) -> str:
    text = (entry.get("QuestionText") or "").strip()
    if not text:
        return ""
    head = text.split("\n\n")[0].strip()
    if len(head) > 900:
        head = head[:900].rsplit(" ", 1)[0]
    return head


def question_message(item: dict[str, Any], n_items: int, previous: list[str]) -> str:
    lines = [f"Question {item['turn']} of {n_items}.", item["prompt"], ""]
    if item["kind"] == "choice":
        lines.append("Options:")
        for index, option in enumerate(item["options"], start=1):
            lines.append(f"{index}. {option}")
        lines.append("")
        lines.append(
            'Reply with JSON only: {"answer": "<one option, exactly as written>", '
            '"rationale": "<one short sentence>"}'
        )
    else:
        lines.append(
            'Reply with JSON only: {"answer": "<a number>", "rationale": "<one short sentence>"}'
        )
    if previous:
        lines.extend(["", "Answers you already gave:"])
        lines.extend(previous)
    return "\n".join(lines)


def human_baseline(numeric: pd.DataFrame, items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in items:
        if item["kind"] == "choice":
            codes = _choice_codes(numeric, item)
            observed = codes.dropna().astype(int)
            n_options = len(item["options"])
            shares = []
            for code in range(1, n_options + 1):
                shares.append(float((observed == code).mean()) if len(observed) else 0.0)
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "choice",
                    "family": item["family"],
                    "n": int(len(observed)),
                    "shares": shares,
                    "mean_code": float(observed.mean()) if len(observed) else None,
                }
            )
        else:
            values = pd.to_numeric(numeric[item["column"]], errors="coerce").dropna()
            rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "numeric",
                    "family": item["family"],
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else None,
                }
            )
    return {"n_respondents": int(len(numeric)), "items": rows}


def _choice_codes(frame: pd.DataFrame, item: dict[str, Any]) -> pd.Series:
    if item["encoding"] == "single":
        return pd.to_numeric(frame[item["column"]], errors="coerce")
    block = frame[item["columns"]].apply(pd.to_numeric, errors="coerce")
    selected = block.fillna(0).gt(0)
    any_selected = selected.any(axis=1)
    index = selected.to_numpy().argmax(axis=1) + 1
    codes = pd.Series(index, index=frame.index, dtype="float64")
    return codes.where(any_selected)


def sample_panel(data_dir: Path, items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    import numpy as np
    import pyarrow.parquet as pq

    from popbench.interview import NARRATIVE_FIELDS, _age_band, _nemotron_persona
    from popbench.personas import nemotron_shard_name

    path = Path(data_dir) / "nemotron_usa" / nemotron_shard_name(0)
    if not path.is_file():
        raise InterviewError(f"Nemotron shard is missing: {path}")
    columns = [
        "uuid",
        "age",
        "sex",
        "state",
        "education_level",
        "occupation",
        "marital_status",
        *NARRATIVE_FIELDS,
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    adults = frame.loc[frame["age"] >= 18].copy()
    adults["age_band"] = adults["age"].map(_age_band)
    adults = adults.loc[adults["age_band"].notna() & adults["sex"].isin(["Female", "Male"])]
    adults["stratum"] = adults["age_band"] + "|" + adults["sex"]
    quotas = allocate_quotas(adults["stratum"].value_counts().to_dict(), n)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum, count in quotas.items():
        pool = adults.index[adults["stratum"] == stratum].to_numpy()
        picked.extend(rng.choice(pool, size=count, replace=False).tolist())
    panel = adults.loc[picked].sort_values(["stratum", "uuid"])
    turns = [
        {
            "turn": item["turn"],
            "item_id": item["item_id"],
            "kind": item["kind"],
            "family": item["family"],
            "block": item["block"],
            "prompt": item["prompt"],
            "options": list(item["options"]),
        }
        for item in items
    ]
    records = []
    for row in panel.itertuples(index=False):
        persona_id = str(row.uuid)
        records.append(
            {
                "id": f"nemotron:{persona_id}",
                "persona": {
                    "id": persona_id,
                    "age_band": row.age_band,
                    "sex": row.sex,
                    "text": _nemotron_persona(row),
                },
                "turns": turns,
            }
        )
    return records


def score_full(responses: list[dict[str, Any]], baseline: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, list[Any]] = {item["item_id"]: [] for item in items}
    n_invalid = 0
    n_answers = 0
    for person in responses:
        answered = {row["item_id"]: row for row in person["answers"]}
        for item in items:
            n_answers += 1
            row = answered.get(item["item_id"])
            if row is None or not row.get("ok"):
                n_invalid += 1
                continue
            by_item[item["item_id"]].append(row["value"])
    human = {row["item_id"]: row for row in baseline["items"]}
    item_rows = []
    for item in items:
        values = by_item[item["item_id"]]
        gold = human[item["item_id"]]
        if item["kind"] == "choice" and values and gold["n"]:
            n_options = len(item["options"])
            shares = [values.count(code) / len(values) for code in range(1, n_options + 1)]
            tvd = total_variation_distance(shares, gold["shares"])
            js = jensen_shannon(shares, gold["shares"])
            mean_code = sum(values) / len(values)
            item_rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "choice",
                    "family": item["family"],
                    "block": item["block"],
                    "prompt": item["prompt"],
                    "n": len(values),
                    "human_n": gold["n"],
                    "tvd": tvd,
                    "js": js,
                    "simulated_mean": mean_code,
                    "human_mean": gold["mean_code"],
                }
            )
        elif item["kind"] == "numeric" and values and gold["mean"] is not None:
            simulated_mean = sum(values) / len(values)
            item_rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": "numeric",
                    "family": item["family"],
                    "block": item["block"],
                    "prompt": item["prompt"],
                    "n": len(values),
                    "human_n": gold["n"],
                    "mae": abs(simulated_mean - gold["mean"]),
                    "simulated_mean": simulated_mean,
                    "human_mean": gold["mean"],
                }
            )
        else:
            item_rows.append(
                {
                    "item_id": item["item_id"],
                    "kind": item["kind"],
                    "family": item["family"],
                    "block": item["block"],
                    "prompt": item["prompt"],
                    "n": len(values),
                    "human_n": gold["n"],
                }
            )
    return {
        "n_people": len(responses),
        "n_items": len(items),
        "n_answers": n_answers,
        "n_invalid": n_invalid,
        "items": item_rows,
        "families": _family_summary(item_rows),
    }


def _family_summary(item_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        grouped[row["family"]].append(row)
    summary = {}
    for family, rows in grouped.items():
        tvds = [row["tvd"] for row in rows if row.get("tvd") is not None]
        jss = [row["js"] for row in rows if row.get("js") is not None]
        maes = [row["mae"] for row in rows if row.get("mae") is not None]
        summary[family] = {
            "n_items": len(rows),
            "n_choice": sum(row["kind"] == "choice" for row in rows),
            "n_numeric": sum(row["kind"] == "numeric" for row in rows),
            "mean_tvd": sum(tvds) / len(tvds) if tvds else None,
            "mean_js": sum(jss) / len(jss) if jss else None,
            "mean_mae": sum(maes) / len(maes) if maes else None,
        }
    return summary


def render_full_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    choice = [row for row in summary["items"] if row.get("tvd") is not None]
    mean_tvd = sum(row["tvd"] for row in choice) / len(choice) if choice else None
    lines = [
        "# Nemotron 回答 Twin-2K 闭口题",
        "",
        f"- 模型：{meta.get('model', 'unknown')}",
        f"- 模拟人数：{summary['n_people']}",
        f"- 题目：{summary['n_items']} 道（闭口题共 {summary.get('n_closed', summary['n_items'])} 道，按问卷顺序答到第 {ASK_N} 道，并答完该题所在矩阵）",
        f"- 无效回答：{summary['n_invalid']} / {summary['n_answers']}",
        f"- 选择题平均 TVD：{_fmt(mean_tvd)}",
        "",
        "Nemotron 人物不是 Twin-2K 受访者。选择题比的是选项份额，数字题比的是均值。",
        "",
        "## 分块",
        "",
        "| 分块 | 题数 | 选择 / 数字 | 平均 TVD | 数字题平均绝对差 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family, row in summary["families"].items():
        lines.append(
            "| {name} | {n} | {choice} / {numeric} | {tvd} | {mae} |".format(
                name=FAMILY_ZH.get(family, family),
                n=row["n_items"],
                choice=row["n_choice"],
                numeric=row["n_numeric"],
                tvd=_fmt(row["mean_tvd"]),
                mae=_fmt(row["mean_mae"]),
            )
        )
    lines.extend(["", "## 差得最大的选择题", ""])
    ranked = sorted(choice, key=lambda row: row["tvd"], reverse=True)[:12]
    for row in ranked:
        prompt = " ".join(row["prompt"].split())
        if len(prompt) > 110:
            prompt = prompt[:110] + "…"
        lines.append(
            f"- TVD {row['tvd']:.3f} · {FAMILY_ZH.get(row['family'], row['family'])} · {row['item_id']} · {prompt}"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def run_full_survey(data_dir: Path, run_dir: Path, workers: int = 4, n_people: int = 50, seed: int = 0) -> Path:
    from popbench.simulate import chat, model_settings_from_env, parse_choice, response_cache_key
    from popbench.twin2k import load_twin2k

    try:
        settings = model_settings_from_env()
    except RuntimeError as exc:
        raise InterviewError(str(exc)) from exc
    data = load_twin2k(data_dir, download=False)
    closed = build_closed_items(data.catalog)
    items = cap_items(closed)
    print(f"asking {len(items)} of {len(closed)} closed items", flush=True)
    out = survey_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_path = out / "human_baseline.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
    else:
        baseline = human_baseline(data.wave1_3, items)
        baseline_path.write_text(json.dumps(baseline) + "\n")
    panel_path = out / "nemotron_panel.jsonl"
    if panel_path.is_file():
        records = [json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
        kept_ids = [item["item_id"] for item in items]
        kept = set(kept_ids)
        for record in records:
            selected = [turn for turn in record["turns"] if turn["item_id"] in kept]
            for turn_no, turn in enumerate(selected, start=1):
                turn["turn"] = turn_no
            record["turns"] = selected
    else:
        records = sample_panel(data_dir, items, n=n_people, seed=seed)
        _write_jsonl(panel_path, records)
    run_dir = Path(run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    finished = 0
    lock = threading.Lock()

    def _one(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal finished
        result = _answer_record(
            settings,
            record,
            cache_dir,
            chat,
            parse_choice,
            response_cache_key,
        )
        with lock:
            finished += 1
            print(f"answered {finished}/{len(records)} {record['id']}", flush=True)
        return result

    workers = max(1, min(workers, len(records)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(_one, records))
    _write_jsonl(run_dir / "responses.jsonl", responses)
    summary = score_full(responses, baseline, items)
    summary["model"] = settings.model
    summary["n_closed"] = len(closed)
    (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    report = render_full_markdown(summary, {"model": settings.model})
    (run_dir / "report.md").write_text(report)
    return run_dir


def _answer_record(settings, record, cache_dir, chat, parse_choice, response_cache_key) -> dict[str, Any]:
    from popbench.interview import system_message

    system = system_message(record["persona"]["text"])
    answers = []
    previous: list[str] = []
    n_items = len(record["turns"])
    for turn in record["turns"]:
        user = question_message(turn, n_items, previous)
        cache_path = cache_dir / f"{response_cache_key(settings.model, record['id'], turn['item_id'])}.json"
        if cache_path.is_file():
            raw = json.loads(cache_path.read_text())["raw"]
        else:
            raw = _chat_with_retry(settings, [{"role": "system", "content": system}, {"role": "user", "content": user}], chat)
            cache_path.write_text(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
        try:
            if turn["kind"] == "choice":
                choice = parse_choice(raw, tuple(turn["options"]))
                value = turn["options"].index(choice.answer) + 1
                shown = choice.answer
            else:
                value = _parse_number(raw, parse_choice)
                shown = str(value)
            answers.append({"item_id": turn["item_id"], "value": value, "raw": raw, "ok": True})
            if turn["turn"] % 50 == 0:
                print(f"  {record['id']} turn {turn['turn']}/{n_items}", flush=True)
            short = " ".join(turn["prompt"].split())
            if len(short) > 80:
                short = short[:80] + "…"
            previous.append(f"{turn['turn']}. {short} -> {shown}")
        except ValueError as exc:
            answers.append(
                {"item_id": turn["item_id"], "value": None, "raw": raw, "ok": False, "error": str(exc)}
            )
    return {"id": record["id"], "answers": answers}


def _parse_number(raw: str, parse_choice) -> float:
    choice = parse_choice(raw, ())
    match = re.search(r"-?\d+(?:\.\d+)?", choice.answer.replace(",", ""))
    if not match:
        raise ValueError(f"answer is not a number: {choice.answer}")
    return float(match.group(0))


def _chat_with_retry(settings, messages, chat) -> str:
    import httpx

    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return chat(settings, messages)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            code = exc.response.status_code
            if code != 429 and code < 500:
                raise InterviewError(f"model request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(min(60, 2**attempt))
    raise InterviewError(f"model request failed after retries: {last_error}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
