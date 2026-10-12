"""Run the recent-experience loop on the 10.12 BFI-44 sample.

Round 0 reuses ``runs/simulation/responses.jsonl``. Each later round appends
that round's recent text and answers in a fresh directory. Rounds 1–3 share
one paragraph per age band. From round 4, each person gets one event.
After an accepted round, that draw is asked of all 200 people.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from popbench.dao.io import read_jsonl
from popbench.simulate.client import model_settings_from_env
from popbench.simulate.recent import (
    AGE_BANDS,
    EVENT_KIND_LABEL,
    FAR_ITEMS,
    FROZEN_ITEMS,
    ITEM_LABEL,
    ITEM_MOVE,
    NEAR_ITEMS,
    accept_round,
    age_band_of,
    attach_recent,
    compare_rounds,
    event_kind,
    first_per_band,
    has_recent_round,
    recent_is_personal,
    recent_text,
    score_frozen,
)
from popbench.simulate.survey import answer_survey

MOVE_ZH = {
    "down": "下拉",
    "up": "上抬",
    "light": "轻触",
    "protect": "保护",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def simulation_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "runs" / "simulation"


def report_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "reports" / "10.12-report" / "recent-pilot"


def run_pilot(
    *,
    workers: int = 8,
    max_rounds: int = 3,
    start_round: int = 1,
    n_per_band: int = 8,
    root: Path | None = None,
) -> dict[str, Any]:
    """Score round 0, then answer recent rounds from ``start_round`` through ``max_rounds``."""
    if start_round < 1 or max_rounds < start_round:
        raise ValueError("start_round must be at least 1 and no greater than max_rounds")
    sim = simulation_dir(root)
    out = report_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    personas = read_jsonl(sim / "personas.jsonl")
    records = read_jsonl(sim / "records.jsonl")
    responses = read_jsonl(sim / "responses.jsonl")
    human = _human_items(sim / "human_baseline.json")
    chosen = first_per_band(personas, n_per_band=n_per_band)
    ids = [person["id"] for person in chosen]
    by_record = {record["id"]: record for record in records}
    missing = [persona_id for persona_id in ids if persona_id not in by_record]
    if missing:
        raise SystemExit(f"records.jsonl is missing {len(missing)} pilot ids")
    pilot_records = [by_record[persona_id] for persona_id in ids]
    pilot_responses = [row for row in responses if row["id"] in set(ids)]
    if len(pilot_responses) != len(ids):
        raise SystemExit(
            f"round 0 has {len(pilot_responses)} of {len(ids)} pilot responses"
        )
    baseline = score_frozen(pilot_responses, human)
    _write_round(
        out / "round-0.md",
        round_index=0,
        scores=baseline,
        baseline=baseline,
        decision=None,
        n_people=len(ids),
    )
    summary: dict[str, Any] = {
        "n_people": len(ids),
        "ids": ids,
        "accepted_round": None,
        "rounds": [{"round": 0, "accepted": None, **_public_scores(baseline)}],
    }
    summary["rounds"].extend(_prior_rows(out / "summary.json", start_round))
    accepted_round: int | None = None
    last_round = max(int(row["round"]) for row in summary["rounds"])
    for round_index in range(start_round, max_rounds + 1):
        if not has_recent_round(round_index):
            print(f"round {round_index} has no recent text; stop for a rewrite", flush=True)
            break
        run_dir = sim / "recent-pilot" / f"round-{round_index}"
        answered = _ensure_answers(pilot_records, run_dir, round_index, workers=workers)
        scores = score_frozen(answered, human)
        decision = compare_rounds(baseline, scores)
        accepted = accept_round(baseline, scores)
        if accepted != bool(decision["accepted"]):
            raise RuntimeError("accept_round disagreed with compare_rounds")
        _write_round(
            out / f"round-{round_index}.md",
            round_index=round_index,
            scores=scores,
            baseline=baseline,
            decision=decision,
            n_people=len(ids),
            people=chosen,
        )
        row: dict[str, Any] = {
            "round": round_index,
            "accepted": accepted,
            **_public_decision(decision),
        }
        if recent_is_personal(round_index):
            row["draw"] = _draw_counts(chosen, round_index)
        summary["rounds"].append(row)
        last_round = round_index
        print(
            f"round {round_index} accepted={accepted} "
            f"far_drop={float(decision['far_tvd_drop']):.4f} "
            f"near_tvd_rise={float(decision['near_tvd_rise']):.4f} "
            f"near_abs_rise={float(decision['near_abs_rise']):.4f}",
            flush=True,
        )
        if accepted:
            accepted_round = round_index
            break
    summary["accepted_round"] = accepted_round
    summary["last_round"] = last_round
    summary["stopped_for_rewrite"] = (
        accepted_round is None and last_round < max_rounds and not has_recent_round(last_round + 1)
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    return summary


def run_full(
    round_index: int,
    *,
    workers: int = 8,
    root: Path | None = None,
    reuse_pilot_cache: bool = True,
) -> dict[str, Any]:
    """Answer all 200 baseline personas with one round's recent text."""
    if not has_recent_round(round_index):
        raise SystemExit(f"round {round_index} has no recent text")
    sim = simulation_dir(root)
    out = report_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(sim / "records.jsonl")
    responses = read_jsonl(sim / "responses.jsonl")
    human = _human_items(sim / "human_baseline.json")
    if len(records) != 200:
        raise SystemExit(f"expected 200 records, found {len(records)}")
    _archive_n200(out, round_index)
    baseline = score_frozen(responses, human)
    run_dir = sim / "recent-pilot" / f"n200-round-{round_index}"
    if reuse_pilot_cache:
        _reuse_cache(sim / "recent-pilot" / f"round-{round_index}" / "cache", run_dir / "cache")
    answered = _ensure_answers(records, run_dir, round_index, workers=workers)
    scores = score_frozen(answered, human)
    decision = compare_rounds(baseline, scores)
    path = out / "n200.md"
    _write_round(
        path,
        round_index=round_index,
        scores=scores,
        baseline=baseline,
        decision=decision,
        n_people=len(records),
        people=records,
        title=f"# 200 人对照（第 {round_index} 轮 recent）",
        note=(
            "这 200 人是 10.12 基线的全部样本。第 0 轮是他们已有的答案，没有 recent。"
            "接受规则与 32 人试点相同，只用于对照，不回头改句子。"
        ),
    )
    payload = {
        "round": round_index,
        "n_people": len(records),
        "accepted": bool(decision["accepted"]),
        **_public_decision(decision),
    }
    (out / "n200.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"n200 round {round_index} accepted={payload['accepted']} "
        f"far_drop={payload['far_tvd_drop']:.4f} "
        f"near_tvd_rise={payload['near_tvd_rise']:.4f} "
        f"near_abs_rise={payload['near_abs_rise']:.4f}",
        flush=True,
    )
    return payload


def _ensure_answers(
    records: list[dict[str, Any]],
    run_dir: Path,
    round_index: int,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    prepared = [attach_recent(record, round_index) for record in records]
    response_path = run_dir / "responses.jsonl"
    expected = {record["id"] for record in prepared}
    if response_path.is_file():
        saved = read_jsonl(response_path)
        if expected <= {row["id"] for row in saved}:
            print(f"reuse {response_path}", flush=True)
            return [row for row in saved if row["id"] in expected]
    settings = model_settings_from_env()
    answer_survey(
        prepared,
        run_dir,
        workers=workers,
        settings=settings,
        panel="atus",
        seed=0,
        family="personality",
        resume=True,
    )
    (run_dir / "recent.json").write_text(
        json.dumps(_recent_payload(round_index, records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    saved = read_jsonl(response_path)
    found = {row["id"] for row in saved}
    if expected - found:
        raise SystemExit(f"{run_dir} is missing {len(expected - found)} responses")
    return [row for row in saved if row["id"] in expected]


def _reuse_cache(source: Path, dest: Path) -> None:
    if not source.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source.glob("*.json"):
        target = dest / path.name
        if target.exists():
            continue
        shutil.copyfile(path, target)
        copied += 1
    if copied:
        print(f"reused {copied} cached replies from {source}", flush=True)


def _human_items(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = {item["item_id"]: item for item in payload["items"]}
    missing = [item_id for item_id in FROZEN_ITEMS if item_id not in items]
    if missing:
        raise SystemExit(f"human baseline is missing {', '.join(missing)}")
    return items


def _write_round(
    path: Path,
    *,
    round_index: int,
    scores: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    decision: dict[str, float | bool] | None,
    n_people: int,
    people: Sequence[Mapping[str, Any]] | None = None,
    title: str | None = None,
    note: str | None = None,
) -> None:
    lines = [title or f"# Recent 经历第 {round_index} 轮", ""]
    lines.append(f"- 人数：{n_people}")
    if note:
        lines.extend(["", note, ""])
    if decision is None:
        lines.append("- 这一轮是第 0 轮，沿用已有答案，没有 recent。")
        lines.append(f"- 最远 8 题平均 TVD：{_fmt(float(_mean_tvd(baseline, FAR_ITEMS)))}")
        lines.append(f"- 最近 8 题平均 TVD：{_fmt(float(_mean_tvd(baseline, NEAR_ITEMS)))}")
        lines.append(f"- 最近 8 题均分绝对差：{_fmt(float(_mean_abs(baseline, NEAR_ITEMS)))}")
    else:
        lines.append(f"- 接受：{'是' if decision['accepted'] else '否'}")
        lines.append(
            "- 最远 8 题平均 TVD："
            f"{_fmt(float(decision['far_mean_tvd_baseline']))} → {_fmt(float(decision['far_mean_tvd']))}"
            f"（{_signed_change(-float(decision['far_tvd_drop']), fell='下降', rose='上升')}）"
        )
        lines.append(
            "- 最近 8 题平均 TVD："
            f"{_fmt(float(decision['near_mean_tvd_baseline']))} → {_fmt(float(decision['near_mean_tvd']))}"
            f"（{_signed_change(float(decision['near_tvd_rise']), fell='下降', rose='上升')}）"
        )
        lines.append(
            "- 最近 8 题均分绝对差："
            f"{_fmt(float(decision['near_mean_abs_baseline']))} → {_fmt(float(decision['near_mean_abs']))}"
            f"（{_signed_change(float(decision['near_abs_rise']), fell='下降', rose='上升')}）"
        )
    if round_index > 0 and people is not None and recent_is_personal(round_index):
        counts = _draw_counts(people, round_index)
        lines.append(
            "- 抽取："
            f"中彩票 {counts['windfall']}，被迫开口 {counts['speech']}，核爆炸 {counts['blast']}"
        )
    if round_index > 0:
        lines.extend(["", "## Recent 原文", ""])
        if recent_is_personal(round_index):
            if not people:
                raise ValueError("a per-person round needs the people who were answered")
            grouped: dict[str, list[str]] = {band: [] for band in AGE_BANDS}
            for person in people:
                band = age_band_of(person)
                if band not in grouped:
                    raise KeyError(f"unexpected age band {band}")
                grouped[band].append(str(person["id"]))
            for band in AGE_BANDS:
                lines.append(f"### {band}")
                lines.append("")
                for person_id in grouped[band]:
                    kind = event_kind(person_id, round_index)
                    lines.append(f"#### {person_id}（{EVENT_KIND_LABEL[kind]}）")
                    lines.append("")
                    lines.append(recent_text(band, round_index, person_id=person_id))
                    lines.append("")
        else:
            for band in AGE_BANDS:
                lines.append(f"### {band}")
                lines.append("")
                lines.append(recent_text(band, round_index))
                lines.append("")
    lines.extend(["", "## 16 题", ""])
    lines.append("| 题 | 短名 | 方向 | 第0轮 TVD | 本轮 TVD | TVD 变化 | 第0轮均分 | 本轮均分 | 人类均分 | 绝对差变化 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item_id in FROZEN_ITEMS:
        before = baseline[item_id]
        after = scores[item_id]
        before_gap = abs(before["sim_mean"] - before["human_mean"])
        after_gap = abs(after["sim_mean"] - after["human_mean"])
        lines.append(
            f"| `{item_id}` | {ITEM_LABEL[item_id]} | {MOVE_ZH[ITEM_MOVE[item_id]]} | "
            f"{_fmt(before['tvd'])} | {_fmt(after['tvd'])} | {_fmt(after['tvd'] - before['tvd'])} | "
            f"{_fmt(before['sim_mean'])} | {_fmt(after['sim_mean'])} | {_fmt(after['human_mean'])} | "
            f"{_fmt(after_gap - before_gap)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Recent 经历试点",
        "",
        f"- 人数：{summary['n_people']}",
        f"- 接受的轮次：{summary['accepted_round'] if summary['accepted_round'] is not None else '无'}",
        "",
        "| 轮次 | 接受 | 最远 TVD 下降 | 最近 TVD 上升 | 最近均分绝对差上升 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["rounds"]:
        if row["round"] == 0:
            lines.append("| 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {row['round']} | {'是' if row['accepted'] else '否'} | "
            f"{_fmt(row['far_tvd_drop'])} | {_fmt(row['near_tvd_rise'])} | {_fmt(row['near_abs_rise'])} |"
        )
    if summary.get("stopped_for_rewrite"):
        lines.extend(["", "下一轮还没有句子。改 `src/popbench/simulate/recent.py` 后再跑。"])
    lines.append("")
    return "\n".join(lines)


def _public_scores(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "far_mean_tvd": _mean_tvd(scores, FAR_ITEMS),
        "near_mean_tvd": _mean_tvd(scores, NEAR_ITEMS),
        "near_mean_abs": _mean_abs(scores, NEAR_ITEMS),
    }


def _public_decision(decision: dict[str, float | bool]) -> dict[str, float]:
    return {
        "far_tvd_drop": float(decision["far_tvd_drop"]),
        "near_tvd_rise": float(decision["near_tvd_rise"]),
        "near_abs_rise": float(decision["near_abs_rise"]),
        "far_mean_tvd": float(decision["far_mean_tvd"]),
        "near_mean_tvd": float(decision["near_mean_tvd"]),
        "near_mean_abs": float(decision["near_mean_abs"]),
    }


def _mean_tvd(scores: dict[str, dict[str, float]], item_ids: tuple[str, ...]) -> float:
    return sum(scores[item_id]["tvd"] for item_id in item_ids) / len(item_ids)


def _mean_abs(scores: dict[str, dict[str, float]], item_ids: tuple[str, ...]) -> float:
    total = 0.0
    for item_id in item_ids:
        row = scores[item_id]
        total += abs(row["sim_mean"] - row["human_mean"])
    return total / len(item_ids)


def _archive_n200(out: Path, round_index: int) -> None:
    """Keep an earlier 200-person write-up when this round replaces it."""
    path = out / "n200.md"
    if not path.is_file():
        return
    first = path.read_text(encoding="utf-8").splitlines()[0]
    prefix = "# 200 人对照（第 "
    suffix = " 轮 recent）"
    if not (first.startswith(prefix) and first.endswith(suffix)):
        return
    old = first[len(prefix) : -len(suffix)]
    if old == str(round_index):
        return
    dest = out / f"n200-round-{old}.md"
    if not dest.exists():
        shutil.copyfile(path, dest)
    json_path = out / "n200.json"
    json_dest = out / f"n200-round-{old}.json"
    if json_path.is_file() and not json_dest.exists():
        shutil.copyfile(json_path, json_dest)


def _prior_rows(path: Path, start_round: int) -> list[dict[str, Any]]:
    """Keep already scored rounds when a later series starts above round 1."""
    if start_round <= 1 or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("rounds", []) if 0 < int(row["round"]) < start_round]
    rows.sort(key=lambda row: int(row["round"]))
    return rows


def _draw_counts(people: Sequence[Mapping[str, Any]], round_index: int) -> dict[str, int]:
    counts = {"windfall": 0, "speech": 0, "blast": 0}
    for person in people:
        kind = event_kind(str(person["id"]), round_index)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _recent_payload(round_index: int, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if recent_is_personal(round_index):
        text = {
            str(record["id"]): recent_text(
                age_band_of(record),
                round_index,
                person_id=str(record["id"]),
            )
            for record in records
        }
        return {"round": round_index, "by": "person", "text": text}
    return {
        "round": round_index,
        "by": "age_band",
        "text": {band: recent_text(band, round_index) for band in AGE_BANDS},
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _signed_change(value: float, *, fell: str, rose: str) -> str:
    """`value` is a rise. Negative means the metric fell."""
    if value >= 0:
        return f"{rose} {_fmt(value)}"
    return f"{fell} {_fmt(-value)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recent-experience loop on the 10.12 BFI-44 sample")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--full-only", action="store_true", help="skip the 32-person loop")
    parser.add_argument("--skip-full", action="store_true", help="do not run the 200-person follow-up")
    parser.add_argument("--round", type=int, default=None, dest="full_round")
    args = parser.parse_args(argv)
    summary: dict[str, Any] | None = None
    if not args.full_only:
        summary = run_pilot(
            workers=args.workers,
            max_rounds=args.max_rounds,
            start_round=args.start_round,
        )
        if summary.get("stopped_for_rewrite"):
            return 2
    if args.skip_full:
        return 0
    round_index = args.full_round
    if round_index is None and summary is not None:
        round_index = summary.get("accepted_round")
    if round_index is None and args.full_only:
        saved = report_dir() / "summary.json"
        if not saved.is_file():
            raise SystemExit("no accepted round; pass --round")
        payload = json.loads(saved.read_text(encoding="utf-8"))
        round_index = payload.get("accepted_round")
    if not round_index:
        print("no accepted round; skip the 200-person run", flush=True)
        return 0
    run_full(int(round_index), workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
