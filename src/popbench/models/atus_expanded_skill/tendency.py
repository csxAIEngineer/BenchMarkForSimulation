"""Card-anchored tendency for cross-person rank.

An id-hash axis draw spreads people apart without tracking the persona, so
Fisher-z Pearson falls. Binning income, education, and age into behavior
levels can reverse ranks the card already implies. This card echoes stated
preferences and prior answers verbatim, keeps those three fields as soft
cues, and allows a tiny residual only when the card has no anchor.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from popbench.models.atus_expanded.expand import _adapt_demographics

_MAX_CLEAR = 8
_TILTS = ("略低", "居中", "略高")
_SOFT_LABEL = (("income", "收入"), ("education", "学历"), ("age", "年龄"))
_CLEAR_FIELDS = (
    ("price_importance", "对价格的在意"),
    ("try_novel_food_technology", "尝新意愿"),
    ("political", "政治倾向"),
    ("ideology", "政治倾向"),
    ("food_resource_constraints", "食物资源约束"),
    ("locally_grown_importance", "本地种植的在意"),
    ("pesticide_free_importance", "无农药的在意"),
    ("brand_importance", "品牌的在意"),
    ("primary_grocery_shopper", "谁买菜"),
    ("primary_cook", "谁做饭"),
    ("grows_own_food", "是否自己种菜"),
    ("food_variety_concern", "对食品种类的担心"),
    ("leafy_green_meals", "叶菜餐次"),
    ("meals_not_prepared_at_home", "外食餐次"),
    ("heard_of_vertical_farming", "是否听说垂直农业"),
    ("heard_of_greenhouse_growing", "是否听说温室种植"),
)
_SOFT_FIELDS = (
    ("income", "income"),
    ("household_income", "income"),
    ("education", "education"),
    ("age", "age"),
)
_SKIP = {
    "education_bucket",
    "sex",
    "race",
    "state",
    "state_name",
    "marital_status",
    "employment",
    "employment_status",
    "work_hours",
    "household_size",
    "religion",
    "income_over_50k",
}
_LINE_LABELS = (
    ("household income:", "income"),
    ("personal income:", "income"),
    ("income:", "income"),
    ("education:", "education"),
    ("age:", "age"),
    ("importance of price", "price"),
    ("political leaning:", "politics"),
    ("likelihood of trying", "novelty"),
)
_QID = re.compile(r"^(?:qid|q)\d+", re.I)
_AGE_BAND = re.compile(
    r"(?<!\d)(?:18\s*[-–]\s*29|30\s*[-–]\s*49|50\s*[-–]\s*64|65\s*\+)",
    re.I,
)
_EDUCATION = (
    "high school",
    "less than high",
    "no formal",
    "some college",
    "some university",
    "associate",
    "bachelor",
    "university",
    "professional",
    "college",
    "postgrad",
    "master",
    "doctoral",
    "phd",
)
_POLITICS = ("liberal", "conservative", "democrat", "republican", "moderate", "independent")
_IDENTITY = {
    "female",
    "male",
    "man",
    "woman",
    "non-binary",
    "nonbinary",
    "married",
    "never married",
    "divorced",
    "separated",
    "widowed",
    "white",
    "black",
    "asian",
    "hispanic",
}
_STRUCTURAL = ("sex", "gender", "race", "ethnic", "marital")


@dataclass(frozen=True)
class Tendency:
    """Verbatim rank anchors. Soft cues are not behavior bins."""

    clear: tuple[tuple[str, str], ...]
    soft: tuple[tuple[str, str], ...]

    @property
    def has_anchor(self) -> bool:
        return bool(self.clear or self.soft)


def persona_tendency(
    demographics: Mapping[str, Any] | None = None,
    *,
    demographic_text: str | None = None,
) -> Tendency:
    """Read card fields. Ignores person id and seed."""
    clear: list[tuple[str, str]] = []
    soft: dict[str, str] = {}
    _take_fields(_adapt_demographics(demographics or {}), clear, soft)
    _take_text(demographic_text, clear, soft)
    ordered = tuple((label, soft[kind]) for kind, label in _SOFT_LABEL if kind in soft)
    return Tendency(tuple(clear[:_MAX_CLEAR]), ordered)


def tendency_card(
    person_id: str,
    demographics: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
    demographic_text: str | None = None,
) -> str:
    """Render the card the model sees. Anchors do not depend on person id."""
    base = persona_tendency(demographics, demographic_text=demographic_text)
    return _render(base, _tilt(person_id, seed) if not base.has_anchor else None)


def _take_fields(
    adapted: Mapping[str, Any],
    clear: list[tuple[str, str]],
    soft: dict[str, str],
) -> None:
    handled = _named_fields(adapted, clear, soft)
    _income_flag(adapted, soft)
    for key, raw in adapted.items():
        if key in handled or key in _SKIP:
            continue
        value = _clean(raw)
        loose = None if not value else _loose_key(key, value)
        if loose is not None:
            _store(loose, value, clear, soft)


def _named_fields(
    adapted: Mapping[str, Any],
    clear: list[tuple[str, str]],
    soft: dict[str, str],
) -> set[str]:
    handled: set[str] = set()
    for key, label in _CLEAR_FIELDS:
        value = _clean(adapted.get(key))
        if value:
            _add_clear(clear, label, value)
            handled.add(key)
    for key, kind in _SOFT_FIELDS:
        value = _clean(adapted.get(key))
        if value:
            _add_soft(soft, kind, value)
            handled.add(key)
    return handled


def _income_flag(adapted: Mapping[str, Any], soft: dict[str, str]) -> None:
    if "income" in soft or "income_over_50k" not in adapted:
        return
    text = "超过 5 万美元" if _truthy(adapted.get("income_over_50k")) else "不超过 5 万美元"
    _add_soft(soft, "income", text)


def _take_text(
    demographic_text: str | None,
    clear: list[tuple[str, str]],
    soft: dict[str, str],
) -> None:
    for key, value in _text_items(demographic_text):
        if not value:
            continue
        _store(_text_target(key, value), value, clear, soft)


def _store(
    target: tuple[str, str] | None,
    value: str,
    clear: list[tuple[str, str]],
    soft: dict[str, str],
) -> None:
    if target is None:
        return
    kind, label = target
    if kind == "soft":
        _add_soft(soft, label, value)
        return
    _add_clear(clear, label, value)


def _text_target(key: str, value: str) -> tuple[str, str] | None:
    if key in {"income", "education", "age"}:
        return ("soft", key)
    if key == "price":
        return ("clear", "对价格的在意")
    if key == "novelty":
        return ("clear", "尝新意愿")
    if key == "politics":
        return ("clear", "政治倾向")
    route = _question_route(key)
    if route in {"income", "education", "age"}:
        return ("soft", route)
    if route == "price":
        return ("clear", "对价格的在意")
    if route == "novelty":
        return ("clear", "尝新意愿")
    if route == "politics" or _is_politics(value):
        return ("clear", "政治倾向")
    if route == "skip" or _is_identity(value):
        return None
    shaped = _shape(value)
    if shaped is not None:
        return shaped
    snippet = " ".join(key.split())[:42]
    label = f"既往回答：{snippet}" if snippet else "既往回答"
    return ("clear", label)


def _loose_key(key: str, value: str) -> tuple[str, str] | None:
    folded = key.casefold().replace("-", "_")
    if "price" in folded and "income" not in folded:
        return ("clear", "对价格的在意")
    if "novel" in folded:
        return ("clear", "尝新意愿")
    if "politic" in folded or "ideolog" in folded:
        return ("clear", "政治倾向")
    if "income" in folded:
        return ("soft", "income")
    if "educat" in folded:
        return ("soft", "education")
    if folded == "age" or folded.endswith("_age"):
        return ("soft", "age")
    shaped = _shape(value)
    if shaped is not None:
        return shaped
    if _QID.match(folded) and not _is_identity(value):
        return ("clear", f"既往回答 {key}")
    return None


def _shape(value: str) -> tuple[str, str] | None:
    if _is_income(value):
        return ("soft", "income")
    if _is_education(value):
        return ("soft", "education")
    if _is_age(value):
        return ("soft", "age")
    if _is_politics(value):
        return ("clear", "政治倾向")
    return None


def _text_items(demographic_text: str | None) -> tuple[tuple[str, str], ...]:
    if not demographic_text:
        return ()
    question: list[str] = []
    found: list[tuple[str, str]] = []
    for line in _lines(demographic_text):
        labeled = _labeled_line(line)
        if labeled is not None:
            found.append(labeled)
            question = []
            continue
        if line.casefold().startswith("answer:"):
            found.append((" ".join(question), line.split(":", 1)[1].strip()))
            question = []
            continue
        question.append(line)
    return tuple(found)


def _labeled_line(line: str) -> tuple[str, str] | None:
    folded = line.casefold()
    if folded.startswith("answer:"):
        return None
    for prefix, kind in _LINE_LABELS:
        if not folded.startswith(prefix) or ":" not in line:
            continue
        value = line.split(":", 1)[1].strip()
        return (kind, value) if value else None
    return None


def _lines(text: str) -> tuple[str, ...]:
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("- ") or line.casefold().startswith("options"):
            continue
        kept.append(line)
    return tuple(kept)


def _question_route(question: str) -> str | None:
    folded = question.casefold()
    if any(token in folded for token in _STRUCTURAL):
        return "skip"
    if "income" in folded or "收入" in folded:
        return "income"
    if "educat" in folded or "学历" in folded or "school" in folded:
        return "education"
    if "年龄" in folded or "how old" in folded or re.search(r"\bage\b", folded):
        return "age"
    if "price" in folded:
        return "price"
    if "novel" in folded:
        return "novelty"
    if "politic" in folded or "ideolog" in folded:
        return "politics"
    return None


def _render(base: Tendency, tilt: str | None) -> str:
    lines = ["【个人倾向｜跟人物卡排序，整场不变】", _clear_block(base.clear), _soft_block(base.soft)]
    if tilt is None:
        lines.append("残差：关闭。已有卡上锚点，人物 id 不再抽取倾向轴。")
    else:
        lines.append(f"残差（没有卡上锚点时才用，偏移不超过 0.04）：{tilt}")
    lines.append("用这些锚点回答所有题；不要答成平均受访者，也不要每题重新发明一套价值观。")
    return "\n".join(lines)


def _clear_block(clear: tuple[tuple[str, str], ...]) -> str:
    if not clear:
        return "明确线索：无。不要改用全体共用的默认。"
    rows = "\n".join(f"- {label}：{value}" for label, value in clear)
    return "明确线索（跨人高低以这些为准，随机倾向不得覆盖）：\n" + rows


def _soft_block(soft: tuple[tuple[str, str], ...]) -> str:
    if not soft:
        return "软线索：无。"
    rows = "\n".join(f"- {label}：{value}" for label, value in soft)
    header = "软线索（只在题目直接相关时保持原文上的相对位置，禁止收成三档行为锁）："
    return header + "\n" + rows


def _add_clear(clear: list[tuple[str, str]], label: str, value: str) -> None:
    if any(existing_label == label or existing == value for existing_label, existing in clear):
        return
    clear.append((label, value))


def _add_soft(soft: dict[str, str], kind: str, value: str) -> None:
    if kind not in soft and value:
        soft[kind] = value


def _tilt(person_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"tendency-residual:{seed}:{person_id}".encode()).hexdigest()
    return _TILTS[int(digest[:8], 16) % 3]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _is_income(value: str) -> bool:
    if not re.search(r"\d", value):
        return False
    folded = value.casefold()
    if "$" in value or "cad" in folded:
        return True
    return bool(re.search(r"\d{2,3},\d{3}", value)) and any(
        token in folded for token in ("less than", "or more", "or less", "-")
    )


def _is_education(value: str) -> bool:
    folded = value.casefold().replace("_", " ")
    return any(token in folded for token in _EDUCATION)


def _is_age(value: str) -> bool:
    """Band text only. A bare 15–100 is a count or a score, not an age."""
    if _is_income(value):
        return False
    return _AGE_BAND.search(value) is not None


def _is_politics(value: str) -> bool:
    folded = value.casefold()
    return any(token in folded for token in _POLITICS)


def _is_identity(value: str) -> bool:
    return value.casefold().strip() in _IDENTITY


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}
