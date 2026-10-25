#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jin10_fetch.py —— 金十数据财经快讯抓取工具

实测结论（2026-10-19）：
  * 接口 https://flash-api.jin10.com/get_flash_list 可访问
  * 必须带 Referer + Origin + x-app-id + x-version，否则返回 502
  * 免登录可读免费快讯正文；VIP 快讯仅返回标题（lock=true）
  * 单次返回最近约 2 小时的 21 条；分页游标需登录态，未登录时服务端会忽略
    max_time 参数并始终返回最新一批——因此本工具用「定时轮询 + 按 id 去重累积」策略

用法:
    python3 scripts/jin10_fetch.py once
    python3 scripts/jin10_fetch.py collect --minutes 120
    python3 scripts/jin10_fetch.py check

``once`` 和 ``collect`` 写入 SQLite（默认 data/recent/jin10.sqlite，可用
环境变量 POPBENCH_RECENT_DB 或 --db 覆盖）。同一 id 再次抓到会更新。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from popbench.simulate.recent_store import db_path, summarize, topic_names, upsert_flashes

FLASH_API = "https://flash-api.jin10.com/get_flash_list"

# 这几个头缺一就会 502。x-app-id / x-version 取自站点公开前端代码
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.jin10.com/",
    "Origin": "https://www.jin10.com",
    "x-app-id": "bVBF4FyRTn5NJF5n",
    "x-version": "1.0.0",
}

# 金十 kinds.id 为站内内部编码，官方未公开完整字典（实测仅能确认少量含义），
# 故仅映射已确认项，其余保留原始 ID，并叠加基于正文的关键词分类。
KIND_TAG_CN = {
    "1": "要闻",
    "2": "外汇商品",
    "29": "地缘政治",
    "17952": "机构观点",
    "18458": "宏观数据",
}

# 正文关键词 -> 主题标签（用于补充站内内部编码无法翻译的部分）
TOPIC_RULES = [
    ("地缘冲突", ["战争", "袭击", "空袭", "导弹", "无人机", "交火", "停火", "军演",
                  "制裁", "冲突", "爆炸", "轰炸", "开火", "伤亡"]),
    ("外交活动", ["外交部", "会谈", "峰会", "外交", "访问", "签署", "协定",
                  "斡旋", "大使", "联合国"]),
    ("央行与利率", ["央行", "利率", "降息", "加息", "货币政策", "通胀", "CPI",
                    "PPI", "美联储", "欧央行", "存款准备金"]),
    ("经济数据", ["GDP", "PMI", "失业率", "非农", "零售", "工业产出", "贸易",
                  "进出口", "消费者信心"]),
    ("大宗商品", ["原油", "黄金", "白银", "铜价", "天然气", "铁矿石", "煤炭",
                  "粮食", "原油库存", "EIA"]),
    ("外汇市场", ["美元", "欧元", "日元", "人民币汇率", "离岸人民币", "外汇",
                  "汇率", "美元指数"]),
    ("股指市场", ["股指", "标普", "纳斯达克", "道琼斯", "A股", "港股", "恒生",
                  "日经", "涨停", "跌停"]),
    ("企业与公司", ["公司", "财报", "营收", "净利润", "并购", "IPO", "上市",
                    "股价", "央行", "裁员", "工厂"]),
    ("国内政策", ["国务院", "发改委", "央行", "财政部", "工信部", "政策",
                  "规划", "会议", "部署"]),
]


def classify(content):
    """按正文关键词打主题标签。"""
    if not content:
        return []
    hits = []
    for tag, kws in TOPIC_RULES:
        if any(k in content for k in kws):
            hits.append(tag)
    return hits


def fetch_once(timeout=20):
    """抓取最新一批快讯，返回规范化后的记录列表。"""
    params = urllib.parse.urlencode({"channel": "-8200", "vip": "1", "max_time": ""})
    request = urllib.request.Request(f"{FLASH_API}?{params}", headers=HEADERS, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}（检查是否缺少 Referer/Origin 头）") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"金十接口不可达: {exc.reason}") from exc
    if status != 200:
        raise RuntimeError(f"HTTP {status}（检查是否缺少 Referer/Origin 头）")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("金十接口返回的不是 JSON") from exc
    return rows_from_payload(payload)


def rows_from_payload(payload):
    """把接口 JSON 收成写入库的记录。"""
    items = payload.get("data") or []
    if not isinstance(items, list):
        raise RuntimeError("金十接口没有返回快讯列表")
    out = []
    for it in items:
        d = it.get("data") or {}
        locked = bool(d.get("lock"))
        kind_ids = [str(k.get("id")) for k in (it.get("kinds") or []) if k.get("id")]
        content = "" if locked else (d.get("content") or "")
        # 站内编码（已确认含义的翻译）+ 正文关键词主题标签
        site_tags = [KIND_TAG_CN.get(k, f"编码{k}") for k in kind_ids]
        topic_tags = classify(content)
        out.append({
            "id": it.get("id", ""),
            "time": it.get("time", ""),
            "channel": it.get("channel", []),
            "important": it.get("important", 0),
            "type": it.get("type", 0),
            "content": content,
            "vip_locked": locked,
            "vip_title": d.get("vip_title") or "",
            "vip_level": d.get("vip_level") or 0,
            "site_tags": site_tags,
            "topics": topic_tags,
        })
    return out


def _clock(stamp):
    text = str(stamp or "")
    if len(text) >= 16:
        return text[11:16]
    return text


def _print_batch(rows):
    free = [row for row in rows if not row["vip_locked"]]
    print(f"[金十] 抓取 {len(rows)} 条（免费正文 {len(free)} 条，VIP 锁定 {len(rows) - len(free)} 条）")
    if rows:
        print(f"  时间范围: {min(row['time'] for row in rows)} ~ {max(row['time'] for row in rows)}")
    shown = 0
    for row in free:
        if not row["content"]:
            continue
        print(f"  [{_clock(row['time'])}] {row['content'][:64]}")
        shown += 1
        if shown >= 8:
            break


def _store(rows, path):
    """Write flashes that have a topic. Rows with 主题无 are left out."""
    skipped = sum(1 for row in rows if not topic_names(row.get("topics")))
    if skipped:
        print(f"  主题为空，不收纳 {skipped} 条")
    return upsert_flashes(rows, path=path)


def _save(rows, path):
    before = summarize(path=path)["count"]
    written = _store(rows, path)
    stats = summarize(path=path)
    added = stats["count"] - before
    print(
        f"[已保存] {db_path(path)}（累计 {stats['count']} 条，"
        f"本次收纳 {written} 条，净增 {added} 条，免费正文 {stats['free_count']} 条）"
    )
    return stats


def cmd_once(args):
    rows = fetch_once()
    _print_batch(rows)
    _save(rows, args.db)


def cmd_collect(args):
    """轮询累积。服务端分页需登录态，故靠定时拉取 + id 去重来沉淀历史。"""
    start_n = summarize(path=args.db)["count"]
    print(f"[金十] 开始轮询 {args.minutes} 分钟，间隔 {args.interval}s，已有 {start_n} 条")
    deadline = time.time() + args.minutes * 60
    polls = 0
    try:
        while time.time() < deadline:
            polls += 1
            try:
                rows = fetch_once()
                before = summarize(path=args.db)["count"]
                _store(rows, args.db)
                new = summarize(path=args.db)["count"] - before
                if new > 0:
                    print(f"  [{polls}] +{new} 条，累计 {before + new}")
            except Exception as exc:
                print(f"  [{polls}] 抓取失败: {str(exc)[:60]}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已中断，正在保存…")
    stats = summarize(path=args.db)
    print(
        f"[完成] 轮询 {polls} 次，共 {stats['count']} 条"
        f"（新增 {stats['count'] - start_n}）→ {db_path(args.db)}"
    )
    print(f"  其中免费正文 {stats['free_count']} 条")


def cmd_check(_args):
    print("金十数据接口连通性检测")
    print("-" * 46)
    try:
        rows = fetch_once()
        free = [row for row in rows if not row["vip_locked"] and row["content"]]
        print(f"[OK] 接口可访问，返回 {len(rows)} 条（免费 {len(free)}）")
        for row in free[:3]:
            print(f"  - [{_clock(row['time'])}] {row['content'][:60]}")
    except Exception as exc:
        print(f"[FAIL] {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="金十数据财经快讯抓取工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    once = sub.add_parser("once", help="抓取最新一批快讯并写入 SQLite")
    once.add_argument("--db", default=None, help="SQLite 路径，默认 POPBENCH_RECENT_DB 或 data/recent/jin10.sqlite")
    once.set_defaults(func=cmd_once)

    collect = sub.add_parser("collect", help="定时轮询累积（补足服务端无分页的缺口）")
    collect.add_argument("--minutes", type=int, default=60, help="轮询时长（分钟）")
    collect.add_argument("--interval", type=int, default=60, help="轮询间隔（秒）")
    collect.add_argument("--db", default=None, help="SQLite 路径，默认 POPBENCH_RECENT_DB 或 data/recent/jin10.sqlite")
    collect.set_defaults(func=cmd_collect)

    check = sub.add_parser("check", help="连通性检测")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
