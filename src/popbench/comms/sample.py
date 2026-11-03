"""Outer sampling for the survey-twin skill.

Pricing uses Bernoulli on a clamped buy_prob. Choice questions use a
categorical draw on probs. Numeric and other items keep the model's answer.
Seeds are ``seed:person_id:buy:item_id`` and ``seed:person_id:choice:item_id``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from popbench.schema import PromptPackage

PROB_LOW = 0.15
PROB_HIGH = 0.85
_BUY_OPTIONS = {"buy", "skip"}


def question_contract(package: PromptPackage) -> str:
    """pricing, choice, or direct. Gold is not consulted."""
    options = tuple(package.options)
    folded = {option.strip().casefold() for option in options if option.strip()}
    if folded and folded <= _BUY_OPTIONS and "buy" in folded:
        return "pricing"
    if package.kind == "numeric":
        return "direct"
    if package.kind in {"multiple_choice", "likert"} and len(options) >= 2:
        return "choice"
    return "direct"


def apply_outer_sample(
    parsed: dict[str, Any],
    package: PromptPackage,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    kind = question_contract(package)
    if kind == "pricing":
        return _price_sample(parsed, package, seed)
    if kind == "choice":
        return _choice_sample(parsed, package, seed)
    if "answer" not in parsed:
        return None, f"{package.question_id} is missing answer"
    return parsed, None


def clamp_prob(value: float) -> float:
    return min(PROB_HIGH, max(PROB_LOW, float(value)))


def sample_rng(seed: int, person_id: str, kind: str, item_id: str) -> random.Random:
    material = f"{seed}:{person_id}:{kind}:{item_id}"
    digest = hashlib.sha256(material.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _price_sample(
    parsed: dict[str, Any],
    package: PromptPackage,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if "buy_prob" not in parsed:
        return None, f"{package.question_id} is missing buy_prob"
    try:
        probability = clamp_prob(float(parsed["buy_prob"]))
    except (TypeError, ValueError):
        return None, f"{package.question_id} buy_prob is not a number"
    rng = sample_rng(seed, package.subject_id, "buy", package.question_id)
    decision = "buy" if rng.random() < probability else "skip"
    updated = dict(parsed)
    updated["buy_prob"] = probability
    updated["answer"] = decision
    return updated, None


def _choice_sample(
    parsed: dict[str, Any],
    package: PromptPackage,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    probs = parsed.get("probs")
    labels = list(package.options)
    weights = _weights(probs, labels)
    if weights is None:
        return None, f"{package.question_id} probs must match its options"
    rng = sample_rng(seed, package.subject_id, "choice", package.question_id)
    updated = dict(parsed)
    updated["answer"] = _categorical(rng, labels, weights)
    return updated, None


def _weights(probs: Any, labels: list[str]) -> list[float] | None:
    if isinstance(probs, dict):
        try:
            values = [max(0.0, float(probs.get(label, 0.0))) for label in labels]
        except (TypeError, ValueError):
            return None
        return values
    if isinstance(probs, list) and len(probs) == len(labels):
        try:
            return [max(0.0, float(value)) for value in probs]
        except (TypeError, ValueError):
            return None
    return None


def _categorical(rng: random.Random, labels: list[str], weights: list[float]) -> str:
    total = sum(weights)
    if total <= 0:
        weights = [1.0 for _ in labels]
        total = float(len(labels))
    draw = rng.random() * total
    cursor = 0.0
    for label, weight in zip(labels, weights, strict=True):
        cursor += weight
        if draw < cursor:
            return label
    return labels[-1]
