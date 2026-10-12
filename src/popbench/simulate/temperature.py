"""Temperature sweep on the 10.12 BFI-44 sample.

Temperature 0 reuses ``runs/simulation/responses.jsonl``. Each higher
temperature answers the same persona cards in its own directory, because the
response cache key does not include temperature. ``model_settings_from_env``
stays at temperature 0; this module passes ``replace(settings, temperature=value)``
into ``answer_survey``. The sweep does not attach recent text.

``run_t08_recent`` is separate. It answers temperature 0.8 again after one
undiluted event is appended to each card, and writes that run under
``runs/simulation/temperature/t0.8-recent``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from popbench.dao.io import read_jsonl
from popbench.simulate.client import model_settings_from_env
from popbench.simulate.recent import (
    FAR_ITEMS,
    FROZEN_ITEMS,
    NEAR_ITEMS,
    UNDILUTED_EVENT_KINDS,
    UNDILUTED_EVENT_LABEL,
    UNDILUTED_EVENT_SEED,
    attach_undiluted_event,
    event_places,
    score_frozen,
    undiluted_event_counts,
    undiluted_event_of_kind,
)
from popbench.simulate.survey import answer_survey

# Temperature 0 is the stored baseline. The other three are sampled once.
TEMPERATURES: tuple[float, ...] = (0.0, 0.4, 0.8, 1.2)
TIER_DIR: dict[float, str] = {0.4: "t0.4", 0.8: "t0.8", 1.2: "t1.2"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def simulation_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "runs" / "simulation"


def report_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "reports" / "10.12-report" / "temperature" / "summary.md"


def t08_recent_report_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "reports" / "10.12-report" / "temperature" / "t0.8-recent.md"


T08_RECENT_DIR = "t0.8-recent"
T08_TEMPERATURE = 0.8
_DELTA_KEYS: tuple[str, ...] = (
    "mean_tvd",
    "far_mean_tvd",
    "near_mean_tvd",
    "near_mean_abs",
    "extreme_share",
)


def tier_metrics(
    responses: Sequence[Mapping[str, Any]],
    human_by_item: Mapping[str, Mapping[str, Any]],
    item_ids: Sequence[str],
) -> dict[str, float | int]:
    """Average TVD, the frozen-item gaps, extreme-option share, and invalid answers."""
    missing = [item_id for item_id in FROZEN_ITEMS if item_id not in item_ids]
    if missing:
        raise ValueError(f"item list is missing frozen ids: {', '.join(missing)}")
    scores = score_frozen(responses, human_by_item, item_ids)
    n_invalid, n_answers = _invalid_answers(responses)
    return {
        "n_people": len(responses),
        "n_items": len(item_ids),
        "mean_tvd": _mean_tvd(scores, item_ids),
        "far_mean_tvd": _mean_tvd(scores, FAR_ITEMS),
        "near_mean_tvd": _mean_tvd(scores, NEAR_ITEMS),
        "near_mean_abs": _mean_abs(scores, NEAR_ITEMS),
        "extreme_share": extreme_share(responses, item_ids),
        "n_invalid": n_invalid,
        "n_answers": n_answers,
    }


def extreme_share(
    responses: Sequence[Mapping[str, Any]],
    item_ids: Sequence[str],
) -> float:
    """Mean, across items, of the share of usable answers coded 1 or 5."""
    rates = [_item_extreme(responses, item_id) for item_id in item_ids]
    return sum(rates) / len(rates)


def human_extreme_share(
    human_by_item: Mapping[str, Mapping[str, Any]],
    item_ids: Sequence[str],
) -> float:
    """Mean, across items, of the human share on options 1 and 5."""
    rates: list[float] = []
    for item_id in item_ids:
        shares = human_by_item[item_id]["shares"]
        if len(shares) != 5:
            raise ValueError(f"{item_id} human shares must have length 5")
        rates.append(float(shares[0]) + float(shares[4]))
    return sum(rates) / len(rates)


def render_summary(
    tiers: Sequence[Mapping[str, Any]],
    *,
    human_extreme: float,
) -> str:
    """Markdown table for the four temperatures."""
    lines = [
        "# 温度梯度",
        "",
        "题干、选项和人物卡与 10.12 的 200 人基线相同。温度 0 复用已有答案。"
        "0.4、0.8、1.2 用同一批人物卡各答一次，模型是 `deepseek-flash`。"
        "人类份额来自 Twin-2K 基线。最远 8 题和最近 8 题沿用 10.12 冻结的题号。",
        "",
        "| 温度 | 44 题平均 TVD | 最远 8 题平均 TVD | 最近 8 题平均 TVD | 最近 8 题均分绝对差 | 两端份额 | 无效回答 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in tiers:
        lines.append(
            f"| {_fmt_temp(float(tier['temperature']))} | {_fmt(float(tier['mean_tvd']))} | "
            f"{_fmt(float(tier['far_mean_tvd']))} | {_fmt(float(tier['near_mean_tvd']))} | "
            f"{_fmt(float(tier['near_mean_abs']))} | {_fmt(float(tier['extreme_share']))} | "
            f"{int(tier['n_invalid'])} / {int(tier['n_answers'])} |"
        )
    lines.extend(
        [
            "",
            "两端份额是 44 题上选项 1 或 5 的比例，再对题目取平均。"
            f"人类基线的两端份额是 {_fmt(human_extreme)}。"
            "均分绝对差是模拟均分与人类均分之差的绝对值，再对最近 8 题取平均。",
            "",
            "答案目录：",
            "",
            "- 温度 0：`runs/simulation/responses.jsonl`",
            "- 温度 0.4：`runs/simulation/temperature/t0.4`",
            "- 温度 0.8：`runs/simulation/temperature/t0.8`",
            "- 温度 1.2：`runs/simulation/temperature/t1.2`",
            "",
            "大于 0 的每一档只抽了一次。",
            "",
        ]
    )
    return "\n".join(lines)


def render_t08_recent(
    plain: Mapping[str, Any],
    recent: Mapping[str, Any],
    *,
    human_extreme: float,
    counts: Mapping[str, int],
) -> str:
    """Markdown for temperature 0.8 with one recent event, against 0.8 without it."""
    delta = _tier_delta(recent, plain)
    lines = [
        "# 温度 0.8 加上随机大事",
        "",
        "温度仍是 0.8，模型是 `deepseek-flash`。题干和选项与 10.12 的 200 人基线相同。"
        "对照是已经答完的、没有 recent 的 0.8 档。"
        "这一档在每人原人物卡末尾加一件已经发生、并且改变了现在做法的大事，标题是 `## Recent event`。"
        "人类份额来自 Twin-2K 基线。最远 8 题和最近 8 题沿用 10.12 冻结的题号。",
        "",
        "| 条件 | 44 题平均 TVD | 最远 8 题平均 TVD | 最近 8 题平均 TVD | 最近 8 题均分绝对差 | 两端份额 | 无效回答 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metric_row("0.8 无 recent", plain),
        _metric_row("0.8 加 recent", recent),
        (
            f"| 差值 | {_fmt(float(delta['mean_tvd']))} | {_fmt(float(delta['far_mean_tvd']))} | "
            f"{_fmt(float(delta['near_mean_tvd']))} | {_fmt(float(delta['near_mean_abs']))} | "
            f"{_fmt(float(delta['extreme_share']))} | — |"
        ),
        "",
        "两端份额是 44 题上选项 1 或 5 的比例，再对题目取平均。"
        f"人类基线的两端份额是 {_fmt(human_extreme)}。"
        "差值是「0.8 加 recent」减去「0.8 无 recent」。"
        "均分绝对差是模拟均分与人类均分之差的绝对值，再对最近 8 题取平均。",
        "",
        _draw_sentence(counts),
        "",
        "下面每段用 30-49 的地点。其他年龄段只换事发地点。",
        "",
    ]
    for kind in UNDILUTED_EVENT_KINDS:
        lines.append(f"### {UNDILUTED_EVENT_LABEL[kind]}")
        lines.append("")
        lines.append(undiluted_event_of_kind(kind, "30-49"))
        lines.append("")
    lines.extend(
        [
            "答案目录：",
            "",
            "- 0.8 无 recent：`runs/simulation/temperature/t0.8`",
            "- 0.8 加 recent：`runs/simulation/temperature/t0.8-recent`",
            "",
            "没有 recent 的 0.8 档这次不重跑。",
            "",
        ]
    )
    return "\n".join(lines)


def run_t08_recent(*, workers: int = 8, root: Path | None = None) -> dict[str, Any]:
    """Answer temperature 0.8 with one undiluted event, then compare with t0.8."""
    sim = simulation_dir(root)
    records = read_jsonl(sim / "records.jsonl")
    if len(records) != 200:
        raise SystemExit(f"expected 200 records, found {len(records)}")
    human_by_item, item_ids = _human_items(sim / "human_baseline.json")
    expected = {str(record["id"]) for record in records}
    plain_rows = _responses_by_id(sim / "temperature" / "t0.8" / "responses.jsonl", expected)
    plain = tier_metrics(plain_rows, human_by_item, item_ids)
    run_dir = sim / "temperature" / T08_RECENT_DIR
    saved = _stored_responses(run_dir / "responses.jsonl", expected)
    if saved is None:
        settings = model_settings_from_env()
        if settings.model != "deepseek-flash":
            raise SystemExit(f"expected deepseek-flash, found {settings.model}")
        attached = [attach_undiluted_event(record) for record in records]
        answer_survey(
            attached,
            run_dir,
            workers=workers,
            settings=replace(settings, temperature=T08_TEMPERATURE),
            panel="atus",
            seed=0,
            family="personality",
            resume=True,
        )
        saved = _stored_responses(run_dir / "responses.jsonl", expected)
        if saved is None:
            raise SystemExit(f"{run_dir} is missing responses")
    else:
        print(f"reuse {run_dir / 'responses.jsonl'}", flush=True)
    recent = tier_metrics(saved, human_by_item, item_ids)
    counts = undiluted_event_counts(records)
    human_extreme = human_extreme_share(human_by_item, item_ids)
    text = render_t08_recent(plain, recent, human_extreme=human_extreme, counts=counts)
    path = t08_recent_report_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _print_named("0.8 plain", plain)
    _print_named("0.8 recent", recent)
    _print_named("delta", _tier_delta(recent, plain))
    print(f"wrote {path}", flush=True)
    return {
        "temperature": T08_TEMPERATURE,
        "seed": UNDILUTED_EVENT_SEED,
        "human_extreme_share": human_extreme,
        "plain": plain,
        "recent": recent,
        "delta": _tier_delta(recent, plain),
        "counts": counts,
        "report": str(path),
    }


def run_sweep(*, workers: int = 8, root: Path | None = None) -> dict[str, Any]:
    """Answer temperatures 0.4, 0.8, and 1.2, then write the summary."""
    sim = simulation_dir(root)
    records = read_jsonl(sim / "records.jsonl")
    if len(records) != 200:
        raise SystemExit(f"expected 200 records, found {len(records)}")
    human_by_item, item_ids = _human_items(sim / "human_baseline.json")
    expected = {str(record["id"]) for record in records}
    baseline = _responses_by_id(sim / "responses.jsonl", expected)
    tiers: list[dict[str, Any]] = [
        {"temperature": 0.0, **tier_metrics(baseline, human_by_item, item_ids)}
    ]
    _print_tier(tiers[-1])
    settings = None
    for temperature in TEMPERATURES:
        if temperature == 0.0:
            continue
        run_dir = sim / "temperature" / TIER_DIR[temperature]
        saved = _stored_responses(run_dir / "responses.jsonl", expected)
        if saved is None:
            if settings is None:
                settings = model_settings_from_env()
                if settings.model != "deepseek-flash":
                    raise SystemExit(f"expected deepseek-flash, found {settings.model}")
            answer_survey(
                records,
                run_dir,
                workers=workers,
                settings=replace(settings, temperature=temperature),
                panel="atus",
                seed=0,
                family="personality",
                resume=True,
            )
            saved = _stored_responses(run_dir / "responses.jsonl", expected)
            if saved is None:
                raise SystemExit(f"{run_dir} is missing responses")
        else:
            print(f"reuse {run_dir / 'responses.jsonl'}", flush=True)
        tiers.append({"temperature": temperature, **tier_metrics(saved, human_by_item, item_ids)})
        _print_tier(tiers[-1])
    human_extreme = human_extreme_share(human_by_item, item_ids)
    text = render_summary(tiers, human_extreme=human_extreme)
    path = report_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return {"human_extreme_share": human_extreme, "tiers": tiers, "report": str(path)}


def _human_items(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    if len(items) != 44:
        raise SystemExit(f"expected 44 human items, found {len(items)}")
    by_id = {item["item_id"]: item for item in items}
    item_ids = [item["item_id"] for item in items]
    missing = [item_id for item_id in FROZEN_ITEMS if item_id not in by_id]
    if missing:
        raise SystemExit(f"human baseline is missing {', '.join(missing)}")
    return by_id, item_ids


def _stored_responses(path: Path, expected: set[str]) -> list[dict[str, Any]] | None:
    """Return one row per id when the file already has the full panel."""
    if not path.is_file():
        return None
    rows = _read_responses(path)
    by_id = {str(row["id"]): row for row in rows}
    if expected <= set(by_id):
        return [by_id[persona_id] for persona_id in expected]
    return None


def _responses_by_id(path: Path, expected: set[str]) -> list[dict[str, Any]]:
    saved = _stored_responses(path, expected)
    if saved is None:
        raise SystemExit(f"{path} is missing responses for the 200 persona cards")
    return saved


def _read_responses(path: Path) -> list[dict[str, Any]]:
    """Read a response file. A trailing partial line is ignored."""
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _invalid_answers(responses: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    total = 0
    invalid = 0
    for response in responses:
        for answer in response.get("answers") or []:
            total += 1
            if _usable_code(answer) is None:
                invalid += 1
    return invalid, total


def _item_codes(responses: Sequence[Mapping[str, Any]], item_id: str) -> list[int]:
    codes: list[int] = []
    for response in responses:
        for answer in response.get("answers") or []:
            if answer.get("item_id") != item_id:
                continue
            code = _usable_code(answer)
            if code is not None:
                codes.append(code)
    return codes


def _item_extreme(responses: Sequence[Mapping[str, Any]], item_id: str) -> float:
    codes = _item_codes(responses, item_id)
    if not codes:
        raise ValueError(f"no usable answers for {item_id}")
    return sum(code in (1, 5) for code in codes) / len(codes)


def _usable_code(answer: Mapping[str, Any]) -> int | None:
    """Match the codes ``score_frozen`` keeps: ok answers on the 1–5 scale."""
    if answer.get("ok") is False:
        return None
    value = answer.get("value")
    if value is None:
        return None
    code = int(value)
    if code < 1 or code > 5:
        return None
    return code


def _mean_tvd(scores: Mapping[str, Mapping[str, float]], item_ids: Sequence[str]) -> float:
    return sum(float(scores[item_id]["tvd"]) for item_id in item_ids) / len(item_ids)


def _mean_abs(scores: Mapping[str, Mapping[str, float]], item_ids: Sequence[str]) -> float:
    total = 0.0
    for item_id in item_ids:
        row = scores[item_id]
        total += abs(float(row["sim_mean"]) - float(row["human_mean"]))
    return total / len(item_ids)


def _tier_delta(recent: Mapping[str, Any], plain: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(recent[key]) - float(plain[key]) for key in _DELTA_KEYS}


def _metric_row(label: str, tier: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {_fmt(float(tier['mean_tvd']))} | {_fmt(float(tier['far_mean_tvd']))} | "
        f"{_fmt(float(tier['near_mean_tvd']))} | {_fmt(float(tier['near_mean_abs']))} | "
        f"{_fmt(float(tier['extreme_share']))} | "
        f"{int(tier['n_invalid'])} / {int(tier['n_answers'])} |"
    )


def _draw_sentence(counts: Mapping[str, int]) -> str:
    places = event_places()
    place_bits = "、".join(f"{band} 为 `{places[band]}`" for band in ("18-29", "30-49", "50-64", "65+"))
    drawn = "，".join(
        f"{UNDILUTED_EVENT_LABEL[kind]} {int(counts[kind])}" for kind in UNDILUTED_EVENT_KINDS
    )
    return (
        f"每人一件。抽取：{drawn}。种子是 `{UNDILUTED_EVENT_SEED}`。"
        f"同一人重跑是同一件事。年龄段只改事发地点：{place_bits}。"
    )


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_temp(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.1f}"


def _print_tier(tier: Mapping[str, Any]) -> None:
    print(
        f"temperature {_fmt_temp(float(tier['temperature']))} "
        f"tvd44={_fmt(float(tier['mean_tvd']))} "
        f"far={_fmt(float(tier['far_mean_tvd']))} "
        f"near={_fmt(float(tier['near_mean_tvd']))} "
        f"near_abs={_fmt(float(tier['near_mean_abs']))} "
        f"extreme={_fmt(float(tier['extreme_share']))} "
        f"invalid={int(tier['n_invalid'])}/{int(tier['n_answers'])}",
        flush=True,
    )


def _print_named(label: str, tier: Mapping[str, Any]) -> None:
    print(
        f"{label} "
        f"tvd44={_fmt(float(tier['mean_tvd']))} "
        f"far={_fmt(float(tier['far_mean_tvd']))} "
        f"near={_fmt(float(tier['near_mean_tvd']))} "
        f"near_abs={_fmt(float(tier['near_mean_abs']))} "
        f"extreme={_fmt(float(tier['extreme_share']))}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temperature sweep on the 10.12 BFI-44 sample")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--recent",
        action="store_true",
        help="answer temperature 0.8 with one recent event and compare it with t0.8",
    )
    args = parser.parse_args(argv)
    if args.recent:
        run_t08_recent(workers=args.workers)
    else:
        run_sweep(workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
