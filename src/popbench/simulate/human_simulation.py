"""Human-like survey prompting for Twin-2K persona twins.

Follows ``skill/humanSimulation/SKILL.md``: survey framing, a stable
person-level tendency card, and (for binary purchase items) ``buy_prob``
plus outer Bernoulli sampling so temperature 0 does not collapse every
card onto the same yes/no rule.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

BUY_OPTION = "Yes, I would purchase the product"
SKIP_OPTION = "No, I would not purchase the product"
BUY_PROB_MIN = 0.15
BUY_PROB_MAX = 0.85

_CARE = ("low", "mid", "high")
_PRICE = ("low", "mid", "high")
_IMPULSE = ("low", "mid", "high")
_NOVELTY = ("avoid", "indifferent", "curious")
_EXPENSIVE = ("pinch", "normal", "ignore")
_UNKNOWN = ("lean skip", "fifty-fifty", "lean buy")

_SYSTEM = (
    "You are filling out a market survey, not giving consumer-advice. "
    "You ARE the person described below. Answer quickly; you may be inconsistent; go by feel. "
    "Do not write long reasoning. Do not judge whether a price is fair. "
    "Do not invent needs the card never mentioned. "
    "For preferences the card does not state, use your personal leanings below—"
    "not a default rule shared by everyone.\n\n"
    "{tendency}\n\n"
    "{persona}"
)

_TENDENCY = (
    "【Personal leanings｜fixed for this whole survey】\n"
    "- Survey care: {care} (low = often click casually)\n"
    "- Price sensitivity: {price}\n"
    "- Impulse: {impulse}\n"
    "- Toward novel snacks / small gadgets: {novelty}\n"
    "- Toward everyday goods that look expensive: {expensive}\n"
    "- Default when the category is unknown: {unknown}\n"
    "Use these leanings on every question; do not invent a new value system per item."
)


def tendency_fields(person_id: str, seed: int) -> dict[str, str]:
    """Stable leanings for one twin. Same person id and seed → same card."""
    rng = _rng(seed, person_id, "tendency")
    return {
        "care": rng.choice(_CARE),
        "price": rng.choice(_PRICE),
        "impulse": rng.choice(_IMPULSE),
        "novelty": rng.choice(_NOVELTY),
        "expensive": rng.choice(_EXPENSIVE),
        "unknown": rng.choice(_UNKNOWN),
    }


def tendency_card(person_id: str, seed: int) -> str:
    """Rendered tendency block pasted into the system message."""
    return _TENDENCY.format(**tendency_fields(person_id, seed))


def human_system_message(persona_text: str, *, person_id: str, seed: int) -> str:
    """Survey system role plus the person card and that twin's leanings."""
    return _SYSTEM.format(
        tendency=tendency_card(person_id, seed),
        persona=str(persona_text).strip(),
    )


def is_pricing_purchase(turn: Mapping[str, Any]) -> bool:
    """True for the Twin-2K QID9 yes/no purchase stem."""
    if str(turn.get("family") or "") == "pricing":
        return True
    options = [str(option) for option in turn.get("options") or []]
    return options == [BUY_OPTION, SKIP_OPTION]


def pricing_reply_instruction(n: int) -> str:
    """Ask for buy probabilities; the outer loop samples yes/no."""
    if n == 1:
        return (
            "Only from the persona card and the personal leanings above, estimate "
            'how likely you are to tap "purchase" on this survey item.\n'
            "Rules:\n"
            "1) One short gut line (≤15 words). No fair-price analysis.\n"
            f"2) buy_prob must sit in {BUY_PROB_MIN:.2f}–{BUY_PROB_MAX:.2f} "
            "(survey answers are rarely certain).\n"
            "3) Do not output yes/no.\n"
            'Reply with JSON only: {"gut":"...","buy_prob":0.xx}'
        )
    return (
        "Only from the persona card and the personal leanings above, estimate "
        'how likely you are to tap "purchase" on each survey item.\n'
        "Rules:\n"
        "1) One short gut line per item (≤15 words). No fair-price analysis.\n"
        f"2) Each buy_prob must sit in {BUY_PROB_MIN:.2f}–{BUY_PROB_MAX:.2f}.\n"
        "3) Do not output yes/no.\n"
        'Reply with JSON only: {"answers":[{"gut":"...","buy_prob":0.xx}, ...]} '
        f"with exactly {n} objects, one per question, in the order below."
    )


def clamp_buy_prob(value: float) -> float:
    """Keep survey probabilities away from 0/1 collapse."""
    return min(BUY_PROB_MAX, max(BUY_PROB_MIN, float(value)))


def sample_purchase(
    buy_prob: float,
    *,
    person_id: str,
    item_id: str,
    seed: int,
) -> bool:
    """Bernoulli draw. Same inputs always yield the same yes/no."""
    p = clamp_buy_prob(buy_prob)
    return _rng(seed, person_id, "buy", item_id).random() < p


def parse_buy_probs(raw: str, turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read gut + buy_prob for each pricing turn from one model string."""
    payload = _json_payload(raw)
    rows = _buy_rows(payload, len(turns))
    out: list[dict[str, Any]] = []
    for turn, row in zip(turns, rows, strict=True):
        gut = str(row.get("gut") or "").strip()
        if "buy_prob" not in row:
            raise ValueError(f"missing buy_prob for {turn['item_id']}")
        prob = clamp_buy_prob(float(row["buy_prob"]))
        out.append({"item_id": turn["item_id"], "gut": gut, "buy_prob": prob})
    return out


def purchase_answer(
    turn: Mapping[str, Any],
    *,
    buy_prob: float,
    gut: str,
    person_id: str,
    seed: int,
    raw: str | None = None,
) -> dict[str, Any]:
    """Map a sampled buy/skip onto the Twin-2K option index (1 = yes, 2 = no)."""
    del raw  # Batch text is not cached; each item stores its own buy_prob payload.
    options = [str(option) for option in turn["options"]]
    if options != [BUY_OPTION, SKIP_OPTION]:
        raise ValueError(f"{turn['item_id']} is not a yes/no purchase item")
    prob = clamp_buy_prob(buy_prob)
    bought = sample_purchase(
        prob,
        person_id=person_id,
        item_id=str(turn["item_id"]),
        seed=seed,
    )
    answer = BUY_OPTION if bought else SKIP_OPTION
    value = options.index(answer) + 1
    model_payload = {"gut": gut, "buy_prob": prob}
    stored = {
        "answer": answer,
        "gut": gut,
        "buy_prob": prob,
        "sampled": bought,
    }
    return {
        "item_id": turn["item_id"],
        "value": value,
        "raw": json.dumps(stored, ensure_ascii=False),
        "ok": True,
        "model_raw": json.dumps(model_payload, ensure_ascii=False),
    }


def soft_choice_reply_instruction(n: int, option_count: int) -> str:
    """Soft option weights for non-pricing items under temperature 0."""
    slot = "[" + ", ".join(["0.xx"] * option_count) + "]"
    if n == 1:
        return (
            "Answer as a hasty survey taker using the persona and personal leanings.\n"
            "Rules:\n"
            "1) One short gut line (≤15 words). No long fairness or optimality analysis.\n"
            "2) Give non-negative probs over the listed options in order; they should "
            "sum to about 1 (we renormalize). Avoid putting nearly all mass on one option "
            "unless the card forces it.\n"
            "3) Do not name the final option.\n"
            f'Reply with JSON only: {{"gut":"...","probs":{slot}}}'
        )
    return (
        "Answer as a hasty survey taker using the persona and personal leanings.\n"
        "Rules:\n"
        "1) One short gut line per item (≤15 words).\n"
        "2) For each item give non-negative probs over its options in listed order; "
        "we renormalize. Avoid one-option collapse unless the card forces it.\n"
        "3) Do not name the final option.\n"
        f'Reply with JSON only: {{"answers":[{{"gut":"...","probs":{slot}}}, ...]}} '
        f"with exactly {n} objects, one per question, in the order below."
    )


def parse_soft_probs(raw: str, turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read gut + option weights for each soft-choice turn."""
    payload = _json_payload(raw)
    rows = _buy_rows(payload, len(turns))
    out: list[dict[str, Any]] = []
    for turn, row in zip(turns, rows, strict=True):
        options = list(turn["options"])
        if "probs" not in row:
            raise ValueError(f"missing probs for {turn['item_id']}")
        probs = _normalize_probs(row["probs"], len(options))
        out.append(
            {
                "item_id": turn["item_id"],
                "gut": str(row.get("gut") or "").strip(),
                "probs": probs,
            }
        )
    return out


def sample_soft_choice(
    probs: Sequence[float],
    *,
    person_id: str,
    item_id: str,
    seed: int,
) -> int:
    """Categorical draw → 0-based option index. Deterministic for fixed inputs."""
    weights = _normalize_probs(probs, len(probs))
    draw = _rng(seed, person_id, "choice", item_id).random()
    total = 0.0
    for index, weight in enumerate(weights):
        total += weight
        if draw < total:
            return index
    return len(weights) - 1


def soft_choice_answer(
    turn: Mapping[str, Any],
    *,
    probs: Sequence[float],
    gut: str,
    person_id: str,
    seed: int,
    raw: str | None = None,
) -> dict[str, Any]:
    """Map sampled option weights onto a Twin-2K choice row."""
    del raw
    options = [str(option) for option in turn["options"]]
    weights = _normalize_probs(probs, len(options))
    index = sample_soft_choice(
        weights,
        person_id=person_id,
        item_id=str(turn["item_id"]),
        seed=seed,
    )
    answer = options[index]
    model_payload = {"gut": gut, "probs": weights}
    stored = {
        "answer": answer,
        "gut": gut,
        "probs": weights,
        "sampled_index": index,
    }
    return {
        "item_id": turn["item_id"],
        "value": index + 1,
        "raw": json.dumps(stored, ensure_ascii=False),
        "ok": True,
        "model_raw": json.dumps(model_payload, ensure_ascii=False),
    }


def _normalize_probs(values: Any, n: int) -> list[float]:
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f"expected {n} probs, got {values!r}")
    nums = [max(0.0, float(value)) for value in values]
    total = sum(nums)
    if total <= 0:
        return [1.0 / n] * n
    return [value / total for value in nums]


def _buy_rows(payload: Any, n: int) -> list[dict[str, Any]]:
    if n == 1 and isinstance(payload, dict) and "buy_prob" in payload:
        return [payload]
    if n == 1 and isinstance(payload, dict) and "probs" in payload:
        return [payload]
    if isinstance(payload, dict) and "answers" in payload:
        values = payload["answers"]
    elif isinstance(payload, list):
        values = payload
    else:
        raise ValueError("model response has no answers")
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f"expected {n} answers, got {values!r}")
    rows: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("each answer must be an object")
        rows.append(item)
    return rows


def _json_payload(raw: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            array_at = text.find("[")
            array_end = text.rfind("]")
            if array_at >= 0 and array_end > array_at:
                return json.loads(text[array_at : array_end + 1])
            raise ValueError("model response is not JSON") from None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not JSON") from exc


def _rng(seed: int, person_id: str, *parts: str) -> random.Random:
    material = ":".join([str(seed), str(person_id), *parts]).encode()
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
