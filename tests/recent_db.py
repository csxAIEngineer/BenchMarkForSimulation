"""Temporary Jin10 flashes for tests that build recent paragraphs."""

from pathlib import Path

from popbench.simulate.recent_store import upsert_flashes

# Newest first. Kind i in a pool uses FLASHES[i], cycling when the pool is longer.
FLASHES: tuple[str, ...] = (
    "Alpha markets moved.",
    "Beta oil stock changed.",
    "Gamma talks opened.",
    "Delta shares were set aside.",
    "Epsilon rates were posted.",
)
_TOPICS: tuple[str, ...] = (
    "股指市场",
    "大宗商品",
    "外交活动",
    "企业与公司",
    "央行与利率",
)


def seed_flashes(path: Path | None = None) -> None:
    rows = []
    for index, body in enumerate(FLASHES):
        rows.append(
            {
                "id": f"flash-{index:02d}",
                "time": f"2026-10-19 {5 - index:02d}:00:00",
                "channel": [1],
                "important": 0,
                "type": 0,
                "content": body,
                "vip_locked": False,
                "vip_title": "",
                "vip_level": 0,
                "site_tags": ["要闻"],
                "topics": [_TOPICS[index]],
            }
        )
    upsert_flashes(rows, path=path)
