"""Persona-conditioned tendency card.

An earlier card drew every axis from a hash of ``person_id``. That spread
people apart, but the spread did not follow income, education, age, or stated
preferences, so cross-person rank (Fisher-z Pearson) fell while the SD ratio
rose. Levels here are a deterministic function of those fields. At most one
axis that no field locked moves by one step, from ``seed`` and ``person_id``,
so identical cards are not clones and uncorrelated noise cannot reshuffle the
ranking. Outer Bernoulli / categorical sampling stays the main stochasticity.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Same adapter the persona card uses, so BDM phrases and outcome columns stay out.
from popbench.models.atus_expanded.expand import _adapt_demographics

_PRICE = "价格敏感"
_NOVELTY = "对新奇零食 / 电子小件"
_PRICEY = "对看起来贵的日用品"
_UNKNOWN = "未知品类"
_ATTITUDE = "态度题"

_LEVELS: dict[str, tuple[str, str, str]] = {
    _PRICE: ("低", "中", "高"),
    _NOVELTY: ("排斥", "无所谓", "想试试"),
    _PRICEY: ("死抠", "一般", "不在乎"),
    _UNKNOWN: ("偏向不买", "五五开", "偏向买"),
    _ATTITUDE: ("偏保守", "中间", "偏自由"),
}

_ANCHOR_LABEL = {
    "income": "收入",
    "income_flag": "收入是否超过5万美元",
    "education": "学历",
    "age": "年龄",
    "politics": "政治倾向",
    "price": "对价格的在意",
    "novelty": "尝新意愿",
    "employment": "就业",
    "marital": "婚姻",
    "household": "家庭规模",
    "religion": "宗教",
    "work": "每周工时",
}
_ANCHOR_ORDER = tuple(_ANCHOR_LABEL)
_MAX_ANCHORS = 8

_KEY_KIND = {
    "income": "income",
    "household_income": "income",
    "qid21": "income",
    "education": "education",
    "age": "age",
    "political": "politics",
    "ideology": "politics",
    "price_importance": "price",
    "try_novel_food_technology": "novelty",
    "marital_status": "marital",
    "household_size": "household",
    "work_hours": "work",
    "employment": "employment",
    "employment_status": "employment",
    "religion": "religion",
}
_SKIP_KEYS = {"education_bucket", "income_over_50k"}
_AGE_BAND = re.compile(
    r"(?<!\d)(?:18\s*[-–]\s*29|30\s*[-–]\s*49|50\s*[-–]\s*64|65\s*\+)",
    re.I,
)
_EMPLOYMENT = (
    "full-time",
    "part-time",
    "self-employed",
    "unemployed",
    "retired",
    "student",
    "home-maker",
    "homemaker",
)
_MARITAL = (
    "never been married",
    "never married",
    "living with a partner",
    "divorced",
    "separated",
    "widowed",
    "married",
)


@dataclass(frozen=True)
class TendencyAxis:
    label: str
    level: str
    pinned: bool


@dataclass(frozen=True)
class Tendency:
    """Field-locked axes, before the optional one-step nudge."""

    anchors: tuple[str, ...]
    axes: tuple[TendencyAxis, ...]

    def levels(self) -> dict[str, str]:
        return {axis.label: axis.level for axis in self.axes}


def persona_tendency(
    demographics: Mapping[str, Any] | None = None,
    *,
    demographic_text: str | None = None,
) -> Tendency:
    """Map card fields to axes. Ignores person id and seed."""
    found = _collect(demographics or {}, demographic_text)
    return Tendency(_anchors(found), _axes(found))


def tendency_card(
    person_id: str,
    demographics: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
    demographic_text: str | None = None,
) -> str:
    """Render the card the model sees. Same inputs always return the same text."""
    base = persona_tendency(demographics, demographic_text=demographic_text)
    axes, nudge = _nudge(base.axes, person_id, seed)
    return _render(base.anchors, axes, nudge)


def _collect(demographics: Mapping[str, Any], demographic_text: str | None) -> dict[str, str]:
    adapted = _adapt_demographics(demographics)
    found: dict[str, str] = {}
    _put_flag(found, adapted)
    keyed = [key for key in adapted if key not in _SKIP_KEYS and _kind_for_key(key)]
    rest = [key for key in adapted if key not in _SKIP_KEYS and key not in keyed]
    for key in (*keyed, *rest):
        kind = _kind_for_key(key) or _kind_for_value(str(adapted[key]))
        if kind:
            _put(found, kind, str(adapted[key]))
    for line in _text_lines(demographic_text):
        parsed = _from_text_line(line)
        if parsed is not None:
            _put(found, parsed[0], parsed[1])
    return found


def _put_flag(found: dict[str, str], adapted: Mapping[str, Any]) -> None:
    flag = adapted.get("income_over_50k")
    if flag is not None and not adapted.get("income"):
        _put(found, "income_flag", str(flag))


def _put(found: dict[str, str], kind: str, value: str) -> None:
    text = " ".join(value.split())
    if kind not in found and text:
        found[kind] = text[:180]


def _anchors(found: Mapping[str, str]) -> tuple[str, ...]:
    rows: list[str] = []
    for kind in _ANCHOR_ORDER:
        if kind not in found:
            continue
        value = found[kind]
        if kind == "income_flag":
            value = "是" if _truthy(value) else "否"
        rows.append(f"{_ANCHOR_LABEL[kind]}：{value}")
    return tuple(rows[:_MAX_ANCHORS])


def _axes(found: Mapping[str, str]) -> tuple[TendencyAxis, ...]:
    income = _income_tier(found.get("income", ""), found.get("income_flag"))
    education = _education_tier(found.get("education", ""))
    age = _age_band(found.get("age", ""))
    politics = _politics(found.get("politics", ""))
    stated_price = _stated_level(found.get("price", ""))
    stated_novelty = _stated_level(found.get("novelty", ""))
    price = _price_axis(income, stated_price)
    novelty = _novelty_axis(age, education, stated_novelty)
    pricey = _pricey_axis(income)
    unknown = _unknown_axis(price, novelty)
    axes = [price, novelty, pricey, unknown]
    attitude = _attitude_axis(politics)
    if attitude is not None:
        axes.append(attitude)
    return tuple(axes)


def _price_axis(income: str | None, stated: str | None) -> TendencyAxis:
    score = 1 if income is None else {"high": 0, "mid": 1, "low": 2}[income]
    if stated == "high":
        score += 1
    elif stated == "low":
        score -= 1
    return TendencyAxis(_PRICE, _LEVELS[_PRICE][_clamp(score)], income is not None or stated is not None)


def _novelty_axis(age: str | None, education: str | None, stated: str | None) -> TendencyAxis:
    score = 1
    if age == "young":
        score += 1
    elif age == "older":
        score -= 1
    if education == "high":
        score += 1
    elif education == "low":
        score -= 1
    if stated == "high":
        score += 1
    elif stated == "low":
        score -= 1
    pinned = age is not None or education is not None or stated is not None
    return TendencyAxis(_NOVELTY, _LEVELS[_NOVELTY][_clamp(score)], pinned)


def _pricey_axis(income: str | None) -> TendencyAxis:
    if income == "high":
        level = "不在乎"
    elif income == "low":
        level = "死抠"
    else:
        level = "一般"
    return TendencyAxis(_PRICEY, level, income is not None)


def _unknown_axis(price: TendencyAxis, novelty: TendencyAxis) -> TendencyAxis:
    if not price.pinned and not novelty.pinned:
        return TendencyAxis(_UNKNOWN, "五五开", False)
    price_index = _LEVELS[_PRICE].index(price.level)
    novelty_index = _LEVELS[_NOVELTY].index(novelty.level)
    if price_index >= 2 and novelty_index <= 1:
        level = "偏向不买"
    elif price_index <= 0 and novelty_index >= 1:
        level = "偏向买"
    else:
        level = "五五开"
    return TendencyAxis(_UNKNOWN, level, True)


def _attitude_axis(politics: str | None) -> TendencyAxis | None:
    if politics == "liberal":
        return TendencyAxis(_ATTITUDE, "偏自由", True)
    if politics == "conservative":
        return TendencyAxis(_ATTITUDE, "偏保守", True)
    if politics == "mid":
        return TendencyAxis(_ATTITUDE, "中间", True)
    return None


def _nudge(
    axes: tuple[TendencyAxis, ...],
    person_id: str,
    seed: int,
) -> tuple[tuple[TendencyAxis, ...], tuple[str, str, str] | None]:
    free = [index for index, axis in enumerate(axes) if not axis.pinned]
    if not free:
        return axes, None
    rng = _rng(seed, person_id)
    index = free[rng.randrange(len(free))]
    axis = axes[index]
    options = _LEVELS[axis.label]
    current = options.index(axis.level)
    step = -1 if rng.randrange(2) == 0 else 1
    if not 0 <= current + step < len(options):
        step = -step
    updated = TendencyAxis(axis.label, options[current + step], False)
    nudged = tuple(updated if i == index else item for i, item in enumerate(axes))
    return nudged, (axis.label, axis.level, updated.level)


def _render(
    anchors: tuple[str, ...],
    axes: tuple[TendencyAxis, ...],
    nudge: tuple[str, str, str] | None,
) -> str:
    lines = ["【个人倾向｜由人物卡字段锁定，整场不变】"]
    if anchors:
        lines.append("锚点（用来把这个人跟别人分开）：")
        lines.extend(f"- {item}" for item in anchors)
    else:
        lines.append("锚点：没有收入、学历、年龄或写明的偏好。不要改用全体共用的默认。")
    lines.append("作答用：")
    lines.extend(f"- {axis.label}：{axis.level}" for axis in axes)
    if nudge is None:
        lines.append("未锁定轴的一档抖动：无。已有字段的轴保持不动。")
    else:
        label, old, new = nudge
        lines.append(f"未锁定轴的一档抖动：{label} {old}→{new}。已由字段锁定的轴不动。")
    lines.append("用这些锚点回答所有题；不要每题重新发明一套价值观。")
    return "\n".join(lines)


def _rng(seed: int, person_id: str) -> random.Random:
    digest = hashlib.sha256(f"tendency-nudge:{seed}:{person_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _clamp(score: int) -> int:
    return max(0, min(2, score))


def _kind_for_key(key: str) -> str | None:
    folded = key.casefold().replace("-", "_")
    if folded in _KEY_KIND:
        return _KEY_KIND[folded]
    if "income" in folded:
        return "income"
    if "educat" in folded:
        return "education"
    if folded == "age" or folded.endswith("_age"):
        return "age"
    if "politic" in folded or "ideolog" in folded:
        return "politics"
    if "price" in folded:
        return "price"
    if "novel" in folded:
        return "novelty"
    if "marital" in folded:
        return "marital"
    if "household" in folded and "size" in folded:
        return "household"
    if "employ" in folded:
        return "employment"
    if "relig" in folded:
        return "religion"
    if "work_hour" in folded:
        return "work"
    return None


def _kind_for_value(value: str) -> str | None:
    if _looks_like_income(value):
        return "income"
    if _education_tier(value):
        return "education"
    if _AGE_BAND.search(value):
        return "age"
    if _politics(value):
        return "politics"
    if _employment(value):
        return "employment"
    lowered = value.casefold()
    if any(token in lowered for token in _MARITAL):
        return "marital"
    return None


def _from_text_line(line: str) -> tuple[str, str] | None:
    """Read a labeled field or an Answer line. Option bullets are not answers."""
    if line.startswith("- "):
        return None
    label, value = _split_label(line)
    if label.casefold() == "answer":
        kind = _kind_for_value(value)
        return None if kind is None else (kind, value)
    kind = _kind_for_key(label) if label else None
    if kind is None or not value:
        return None
    return kind, value


def _split_label(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", line.strip()
    label, value = line.split(":", 1)
    return label.strip(), value.strip()


def _text_lines(demographic_text: str | None) -> tuple[str, ...]:
    if not demographic_text:
        return ()
    return tuple(line.strip() for line in demographic_text.splitlines() if line.strip())


def _looks_like_income(value: str) -> bool:
    if not re.search(r"\d", value):
        return False
    folded = value.casefold()
    if "$" in value or "cad" in folded:
        return True
    return bool(re.search(r"\d{2,3},\d{3}", value)) and any(
        token in folded for token in ("less than", "or more", "or less", "-")
    )


def _income_tier(text: str, flag: str | None) -> str | None:
    folded = text.casefold()
    money = [int(piece.replace(",", "")) for piece in re.findall(r"\d[\d,]*", text)]
    money = [number for number in money if number >= 1_000]
    if money:
        return _tier_from_money(folded, money[0])
    if flag is None:
        return None
    return "mid" if _truthy(flag) else "low"


def _tier_from_money(folded: str, low_end: int) -> str:
    if any(token in folded for token in ("or less", "or under", "≤")) and low_end <= 50_000:
        return "low"
    if low_end >= 75_000:
        return "high"
    if low_end >= 50_000:
        return "mid"
    return "low"


def _education_tier(text: str) -> str | None:
    if not text.strip():
        return None
    folded = text.casefold().replace("_", " ")
    if "less than high" in folded or "no formal" in folded:
        return "low"
    if "high school" in folded and not any(token in folded for token in ("college", "bachelor", "university")):
        return "low"
    if "some college" in folded or "some university" in folded:
        return "mid"
    if "associate" in folded:
        return "mid"
    high_tokens = ("postgrad", "bachelor", "university", "professional", "college graduate", "master", "doctoral")
    if any(token in folded for token in high_tokens) or folded == "bachelor or higher":
        return "high"
    return None


def _age_band(text: str) -> str | None:
    if not text.strip():
        return None
    if _AGE_BAND.search(text):
        folded = text.casefold()
        if "18" in folded and "29" in folded:
            return "young"
        if "65" in folded:
            return "older"
        return "mid"
    numbers = [int(piece) for piece in re.findall(r"\d+", text)]
    if len(numbers) != 1 or not 15 <= numbers[0] <= 100:
        return None
    age = numbers[0]
    if age < 35:
        return "young"
    if age >= 65:
        return "older"
    return "mid"


def _politics(text: str) -> str | None:
    folded = text.casefold()
    if not folded.strip():
        return None
    liberal = any(token in folded for token in ("liberal", "democrat"))
    conservative = any(token in folded for token in ("conservative", "republican"))
    if liberal and conservative:
        return "mid"
    if liberal:
        return "liberal"
    if conservative:
        return "conservative"
    if any(token in folded for token in ("moderate", "independent")):
        return "mid"
    return None


def _stated_level(text: str) -> str | None:
    folded = text.casefold()
    if not folded.strip():
        return None
    if any(token in folded for token in ("not important", "unimportant", "unlikely", "not likely")):
        return "low"
    if any(token in folded for token in ("somewhat", "moderate", "medium")):
        return "mid"
    if any(token in folded for token in ("very", "extremely", "important", "likely", "high")):
        return "high"
    if folded in {"low", "mid", "high"}:
        return folded
    return None


def _employment(value: str) -> bool:
    folded = value.casefold()
    return any(token in folded for token in _EMPLOYMENT)


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}
