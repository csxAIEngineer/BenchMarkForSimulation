"""SQLite store of Jin10 flashes used as recent-event text.

The database file is ``data/recent/jin10.sqlite`` unless ``POPBENCH_RECENT_DB``
is set. Paragraphs use rows with ``vip_locked`` 0, non-empty text, and at least
one topic, newest first. A flash with no topic is not stored. Markup is removed
when a paragraph is built. The stored content is left as it was fetched.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterator

ENV_DB = "POPBENCH_RECENT_DB"
EMPTY_STORE_MESSAGE = "no free Jin10 flash with content; run scripts/jin10_fetch.py once"

_TAG = re.compile(r"<[^>]+>")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS flashes (
    id TEXT PRIMARY KEY,
    time TEXT NOT NULL,
    channel TEXT NOT NULL,
    important INTEGER NOT NULL,
    type INTEGER NOT NULL,
    content TEXT NOT NULL,
    vip_locked INTEGER NOT NULL,
    vip_title TEXT NOT NULL,
    vip_level INTEGER NOT NULL,
    site_tags TEXT NOT NULL,
    topics TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""
_UPSERT = """
INSERT INTO flashes (
    id, time, channel, important, type, content,
    vip_locked, vip_title, vip_level, site_tags, topics, fetched_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    time = excluded.time,
    channel = excluded.channel,
    important = excluded.important,
    type = excluded.type,
    content = excluded.content,
    vip_locked = excluded.vip_locked,
    vip_title = excluded.vip_title,
    vip_level = excluded.vip_level,
    site_tags = excluded.site_tags,
    topics = excluded.topics,
    fetched_at = excluded.fetched_at
"""
_SELECT = """
SELECT id, time, channel, important, type, content, vip_locked,
       vip_title, vip_level, site_tags, topics, fetched_at
FROM flashes
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def db_path(path: Path | str | None = None) -> Path:
    """Store path. An explicit path wins, then the env var, then the repo file."""
    if path is not None and str(path).strip():
        return Path(path)
    override = os.environ.get(ENV_DB)
    if override:
        return Path(override)
    return repo_root() / "data" / "recent" / "jin10.sqlite"


def strip_markup(text: str) -> str:
    """Drop simple HTML tags and collapse whitespace."""
    raw = unescape(text or "")
    raw = _TAG.sub(" ", raw)
    return " ".join(raw.split())


def topic_names(value: Any) -> list[str]:
    """Topic labels. An empty result is what the reader shows as 主题无."""
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        text = value.strip()
        if not text or text == "无":
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            items = parsed if isinstance(parsed, list) else [parsed]
        else:
            items = [text]
    else:
        items = list(value)
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name != "无":
            names.append(name)
    return names


def upsert_flashes(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Path | str | None = None,
    fetched_at: str | None = None,
) -> int:
    """Insert flashes that have a topic. The same id with no topic is removed."""
    stamp = fetched_at or _utc_now()
    kept: list[Mapping[str, Any]] = []
    drop_ids: list[str] = []
    for row in rows:
        flash_id = str(row.get("id") or "").strip()
        if not flash_id:
            raise ValueError("flash id is empty")
        if topic_names(row.get("topics")):
            kept.append(row)
        else:
            drop_ids.append(flash_id)
    with _connection(db_path(path), write=True) as conn:
        if kept:
            conn.executemany(_UPSERT, [_bind(row, stamp) for row in kept])
        if drop_ids:
            conn.executemany("DELETE FROM flashes WHERE id = ?", [(flash_id,) for flash_id in drop_ids])
        _delete_untopiced(conn)
    return len(kept)


def free_flashes(*, path: Path | str | None = None) -> list[dict[str, Any]]:
    """Unlocked flashes with text and a topic, newest ``time`` then ``id`` first."""
    rows = _load(
        path,
        where="vip_locked = 0 AND TRIM(content) != ''",
        suffix="ORDER BY time DESC, id DESC",
    )
    return [row for row in rows if row["plain"] and topic_names(row["topics"])]


def plain_for_index(index: int, *, path: Path | str | None = None) -> str:
    """Plain text of the free flash at this index, cycling when the store is short."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("flash index must be a non-negative integer")
    rows = free_flashes(path=path)
    if not rows:
        raise RuntimeError(EMPTY_STORE_MESSAGE)
    return str(rows[index % len(rows)]["plain"])


def summarize(*, path: Path | str | None = None) -> dict[str, Any]:
    """Row count, usable free-text count, and the oldest and newest times."""
    target = db_path(path)
    if not target.is_file():
        return {"count": 0, "free_count": 0, "time_min": None, "time_max": None}
    with _connection(target, write=False) as conn:
        count, time_min, time_max = conn.execute(
            "SELECT COUNT(*), MIN(time), MAX(time) FROM flashes"
        ).fetchone()
    return {
        "count": int(count),
        "free_count": len(free_flashes(path=path)),
        "time_min": time_min,
        "time_max": time_max,
    }


def latest_flashes(limit: int, *, path: Path | str | None = None) -> list[dict[str, Any]]:
    """Newest flashes, including locked rows."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return []
    return _load(path, suffix="ORDER BY time DESC, id DESC LIMIT ?", params=(limit,))


def flash_by_id(flash_id: str, *, path: Path | str | None = None) -> dict[str, Any] | None:
    """One stored flash, or None when that id is not in the store."""
    rows = _load(path, where="id = ?", params=(str(flash_id),))
    if not rows:
        return None
    return rows[0]


def _delete_untopiced(conn: sqlite3.Connection) -> None:
    found = conn.execute("SELECT id, topics FROM flashes").fetchall()
    drop = [(str(row["id"]),) for row in found if not topic_names(row["topics"])]
    if drop:
        conn.executemany("DELETE FROM flashes WHERE id = ?", drop)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bind(row: Mapping[str, Any], fetched_at: str) -> tuple[Any, ...]:
    flash_id = str(row.get("id") or "").strip()
    if not flash_id:
        raise ValueError("flash id is empty")
    content = row.get("content")
    if content is None:
        content = ""
    else:
        content = str(content)
    return (
        flash_id,
        str(row.get("time") or ""),
        _json_list(row.get("channel")),
        _as_int(row.get("important")),
        _as_int(row.get("type")),
        content,
        _as_flag(row.get("vip_locked")),
        str(row.get("vip_title") or ""),
        _as_int(row.get("vip_level")),
        _json_list(row.get("site_tags")),
        _json_list(row.get("topics")),
        fetched_at,
    )


def _as_flag(value: Any) -> int:
    if value is True or value == 1:
        return 1
    return 0


def _as_int(value: Any) -> int:
    if value is None or value is False:
        return 0
    if value is True:
        return 1
    return int(value)


def _json_list(value: Any) -> str:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    return json.dumps(items, ensure_ascii=False)


def _public(row: sqlite3.Row) -> dict[str, Any]:
    content = str(row["content"])
    return {
        "id": str(row["id"]),
        "time": str(row["time"]),
        "channel": json.loads(row["channel"]),
        "important": int(row["important"]),
        "type": int(row["type"]),
        "content": content,
        "plain": strip_markup(content),
        "vip_locked": int(row["vip_locked"]),
        "vip_title": str(row["vip_title"]),
        "vip_level": int(row["vip_level"]),
        "site_tags": json.loads(row["site_tags"]),
        "topics": json.loads(row["topics"]),
        "fetched_at": str(row["fetched_at"]),
    }


def _load(
    path: Path | str | None,
    *,
    where: str = "",
    params: tuple[Any, ...] = (),
    suffix: str = "",
) -> list[dict[str, Any]]:
    target = db_path(path)
    if not target.is_file():
        return []
    sql = _SELECT
    if where:
        sql += " WHERE " + where
    if suffix:
        sql += " " + suffix
    with _connection(target, write=False) as conn:
        found = conn.execute(sql, params).fetchall()
    return [_public(row) for row in found]


@contextmanager
def _connection(path: Path, *, write: bool) -> Iterator[sqlite3.Connection]:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        if write:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        yield conn
        if write:
            conn.commit()
    finally:
        conn.close()
