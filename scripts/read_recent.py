#!/usr/bin/env python3
"""Print Jin10 flashes stored for recent paragraphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from popbench.simulate.recent_store import flash_by_id, latest_flashes, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print Jin10 flashes stored for recent paragraphs")
    parser.add_argument("--limit", type=int, default=8, help="how many latest flashes to print")
    parser.add_argument("--id", help="print this flash instead of the latest list")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    stats = summarize()
    print(f"条数: {stats['count']}")
    if stats["time_min"] is None:
        print("时间范围: 无")
    else:
        print(f"时间范围: {stats['time_min']} ~ {stats['time_max']}")
    print(f"免费正文: {stats['free_count']}")
    print()

    if args.id:
        row = flash_by_id(args.id)
        if row is None:
            print(f"没有 id {args.id}", file=sys.stderr)
            return 1
        _print_detail(row)
        return 0

    for row in latest_flashes(args.limit):
        _print_summary(row)
        print()
    return 0


def _topics(row: dict) -> str:
    names = row["topics"] or []
    if not names:
        return "无"
    return "、".join(str(name) for name in names)


def _eligible(row: dict) -> bool:
    return row["vip_locked"] == 0 and bool(row["plain"])


def _clip(text: str, limit: int = 80) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def _print_summary(row: dict) -> None:
    mark = "" if _eligible(row) else "  VIP"
    body = row["plain"] or row["vip_title"]
    print(f"{row['time']}  id={row['id']}  主题: {_topics(row)}{mark}")
    print(f"  {_clip(body)}")


def _print_detail(row: dict) -> None:
    print(f"id: {row['id']}")
    print(f"time: {row['time']}")
    print(f"主题: {_topics(row)}")
    print(f"site_tags: {'、'.join(str(tag) for tag in row['site_tags']) or '无'}")
    print(f"vip_locked: {row['vip_locked']}")
    if row["vip_title"]:
        print(f"vip_title: {row['vip_title']}")
    print(f"用于段落: {'是' if _eligible(row) else '否'}")
    print(f"正文: {row['plain']}")
    if row["content"] != row["plain"]:
        print(f"原文: {row['content']}")


if __name__ == "__main__":
    raise SystemExit(main())
