"""Jin10 flash store: id updates, and locked or empty rows stay out of paragraphs."""

from pathlib import Path

import pytest

from popbench.simulate.recent import education_recent_text, recent_text, undiluted_event_text
from popbench.simulate.recent_store import (
    db_path,
    flash_by_id,
    free_flashes,
    plain_for_index,
    strip_markup,
    summarize,
    upsert_flashes,
)


def _row(
    flash_id: str,
    when: str,
    content: str,
    *,
    vip_locked: bool = False,
    vip_title: str = "",
    topics: list[str] | None = None,
    site_tags: list[str] | None = None,
    channel: list[int] | None = None,
) -> dict:
    return {
        "id": flash_id,
        "time": when,
        "channel": [1, 3] if channel is None else channel,
        "important": 0,
        "type": 0,
        "content": content,
        "vip_locked": vip_locked,
        "vip_title": vip_title,
        "vip_level": 1 if vip_locked else 0,
        "site_tags": ["要闻"] if site_tags is None else site_tags,
        "topics": ["地缘冲突"] if topics is None else topics,
    }


def test_default_path_is_the_repo_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POPBENCH_RECENT_DB", raising=False)
    path = db_path()
    assert path.as_posix().endswith("data/recent/jin10.sqlite")


def test_lookup_uses_the_env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "other.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    upsert_flashes([_row("env", "2026-10-19 01:00:00", "From env.")])
    assert plain_for_index(0) == "From env."
    assert path.is_file()


def test_upsert_updates_the_same_id(tmp_path: Path):
    path = tmp_path / "dedup.sqlite"
    upsert_flashes(
        [_row("1", "2026-10-19 01:00:00", "first", topics=["地缘冲突"])],
        path=path,
        fetched_at="2026-10-19T01:00:00Z",
    )
    upsert_flashes(
        [
            _row("1", "2026-10-19 02:00:00", "second", topics=["外交活动"], site_tags=["地缘政治"]),
            _row("2", "2026-10-19 03:00:00", "other"),
        ],
        path=path,
        fetched_at="2026-10-19T02:00:00Z",
    )
    stats = summarize(path=path)
    assert stats["count"] == 2
    assert stats["free_count"] == 2
    assert stats["time_min"] == "2026-10-19 02:00:00"
    assert stats["time_max"] == "2026-10-19 03:00:00"
    row = flash_by_id("1", path=path)
    assert row is not None
    assert row["content"] == "second"
    assert row["time"] == "2026-10-19 02:00:00"
    assert row["fetched_at"] == "2026-10-19T02:00:00Z"
    assert row["topics"] == ["外交活动"]
    assert row["site_tags"] == ["地缘政治"]
    assert row["channel"] == [1, 3]
    assert plain_for_index(0, path=path) == "other"
    assert plain_for_index(1, path=path) == "second"


def test_same_time_breaks_ties_by_id_descending(tmp_path: Path):
    path = tmp_path / "tie.sqlite"
    upsert_flashes(
        [
            _row("a", "2026-10-19 04:00:00", "lower id"),
            _row("b", "2026-10-19 04:00:00", "higher id"),
        ],
        path=path,
    )
    assert plain_for_index(0, path=path) == "higher id"
    assert plain_for_index(1, path=path) == "lower id"


def test_strip_markup_drops_tags_and_keeps_text():
    assert strip_markup("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"
    assert strip_markup("   ") == ""
    assert strip_markup("<br/>") == ""


def test_markup_stays_in_the_row_and_leaves_the_paragraph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "html.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    raw = "<p>Hello&nbsp;<b>world</b></p>"
    upsert_flashes([_row("html", "2026-10-19 12:00:00", raw, topics=["股指市场"])])
    stored = flash_by_id("html")
    assert stored is not None
    assert stored["content"] == raw
    assert stored["plain"] == "Hello world"
    text = recent_text("18-29", 1)
    assert text == "Hello world"
    assert "<b>" not in text
    assert "<p>" not in text


def test_locked_or_empty_content_is_not_a_paragraph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "skip.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    upsert_flashes(
        [
            _row("locked", "2026-10-19 03:00:00", "SECRET headline", vip_locked=True, vip_title="VIP title"),
            _row("blank", "2026-10-19 02:00:00", "   "),
            _row("tags", "2026-10-19 01:30:00", "<br/>"),
            _row("free", "2026-10-19 01:00:00", "Public headline."),
        ]
    )
    assert [row["id"] for row in free_flashes()] == ["free"]
    assert plain_for_index(0) == "Public headline."
    assert plain_for_index(4) == "Public headline."
    text = education_recent_text("65+", "visitor-1")
    assert text == "Public headline."
    assert "SECRET" not in text
    stored = flash_by_id("locked")
    assert stored is not None
    assert stored["content"] == "SECRET headline"
    assert stored["vip_locked"] == 1


def test_empty_topics_are_not_stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "topics.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    upsert_flashes(
        [
            _row("kept", "2026-10-19 01:00:00", "Kept headline.", topics=["地缘冲突"]),
            _row("bare", "2026-10-19 03:00:00", "No topic headline.", topics=[]),
            _row("blank-label", "2026-10-19 04:00:00", "Labeled none.", topics=["无"]),
        ]
    )
    assert summarize()["count"] == 1
    assert [row["id"] for row in free_flashes()] == ["kept"]
    assert flash_by_id("bare") is None
    assert flash_by_id("blank-label") is None
    assert "No topic" not in recent_text("18-29", 1)
    upsert_flashes([_row("kept", "2026-10-19 05:00:00", "Kept headline.", topics=[])])
    assert flash_by_id("kept") is None
    assert summarize()["count"] == 0


def test_no_free_flash_names_the_fetch_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "empty.sqlite"
    monkeypatch.setenv("POPBENCH_RECENT_DB", str(path))
    with pytest.raises(RuntimeError, match=r"scripts/jin10_fetch\.py once"):
        recent_text("18-29", 1)
    upsert_flashes([_row("locked", "2026-10-19 03:00:00", "SECRET", vip_locked=True)])
    with pytest.raises(RuntimeError, match=r"scripts/jin10_fetch\.py once"):
        undiluted_event_text("30-49", "visitor-1")
    with pytest.raises(RuntimeError, match=r"scripts/jin10_fetch\.py once"):
        education_recent_text("50-64", "visitor-1")
