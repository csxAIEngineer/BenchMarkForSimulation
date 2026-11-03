"""One latent tendency card per person.

The draw uses only the person id, so it stays fixed across that person's
items and across runs. Different ids get different cards.
"""

from __future__ import annotations

import hashlib
import random

_AXES = (
    ("问卷认真度", ("低", "中", "高")),
    ("价格敏感", ("低", "中", "高")),
    ("冲动", ("低", "中", "高")),
    ("对新奇零食 / 电子小件", ("排斥", "无所谓", "想试试")),
    ("对「看起来贵的日用品」", ("死抠", "一般", "不在乎")),
    ("未知品类默认", ("偏向不买", "五五开", "偏向买")),
)


def tendency_card(person_id: str) -> str:
    rng = _rng(person_id)
    lines = ["【本轮个人倾向｜整场调查固定不变】"]
    for label, options in _AXES:
        lines.append(f"- {label}：{options[rng.randrange(len(options))]}")
    lines.append("用这些倾向回答所有题；不要每题重新发明一套价值观。")
    return "\n".join(lines)


def _rng(person_id: str) -> random.Random:
    digest = hashlib.sha256(f"tendency:{person_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))
