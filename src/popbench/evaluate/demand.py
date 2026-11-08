"""Free-price endpoint and between-subject Twin2K demand curves.

Offers are joined on ``subject_id`` and ``question_id``. A curve here is the
purchase rate in observed subject-price cells. It is not an individual's
counterfactual demand, and a non-increasing pattern is not strict decrease:
a later tier may tie an earlier one. An upward step is a rate that rises as
the comparable price tier rises. Cells with no mapped answers are absent,
not zero demand.

Person rates and these offer rates both use the official 0/1 option codes.
Unmapped replies, parse errors, and conflicting duplicates stay unknown.
They are not coded as No. Mapped rates use the mapped denominator. Bounds
put every unknown at No (lower) or Yes (upper) over all assigned offers.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from popbench.evaluate.options import binary_codebook, code_purchase
from popbench.evaluate.stimulus import is_free_price, tier_index

FREE = Decimal("0.00")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _collapse(rows: Sequence[Mapping[str, object]]) -> tuple[str, str | None]:
    if not rows:
        return "missing", None
    if any(row.get("parse_error") for row in rows):
        if any(not row.get("parse_error") for row in rows):
            return "conflict", None
        return "parse_error", None
    texts = [str(row.get("answer") or "").strip() for row in rows]
    if len(set(texts)) != 1:
        return "conflict", None
    return "coded_text", texts[0]


def build_offers(
    outcome_rows: Sequence[Mapping[str, object]],
    response_rows: Sequence[Mapping[str, object]],
    sample_ids: Sequence[str],
    stimuli: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, Any]]:
    """One offer per sample subject and pricing question.

    ``stimuli`` maps ``(subject_id, question_id)`` to a parsed stimulus
    (``product``, ``price_usd``). Missing parses are kept and marked.
    """
    sample = [str(subject_id) for subject_id in sample_ids]
    sample_set = set(sample)
    human: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    options_by_question: dict[str, set[str]] = defaultdict(set)
    for row in outcome_rows:
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in sample_set:
            continue
        items = row.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id") or "")
            answer = str(item.get("answer") or "").strip()
            human[(subject_id, question_id)].append({"answer": answer, "parse_error": None})
            if answer:
                options_by_question[question_id].add(answer)
    codebooks = {
        question_id: binary_codebook(sorted(options))
        for question_id, options in options_by_question.items()
    }
    model: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in response_rows:
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in sample_set:
            continue
        question_id = str(row.get("question_id") or "")
        model[(subject_id, question_id)].append(row)

    question_ids = sorted({question_id for _subject, question_id in human})
    offers: list[dict[str, Any]] = []
    for subject_id in sample:
        for question_id in question_ids:
            if (subject_id, question_id) not in human and (subject_id, question_id) not in stimuli:
                continue
            stimulus = stimuli.get((subject_id, question_id))
            human_status, human_text = _collapse(human.get((subject_id, question_id), []))
            model_status, model_text = _collapse(model.get((subject_id, question_id), []))
            codebook = codebooks.get(question_id)
            human_code = None
            if human_status == "coded_text" and human_text is not None and codebook is not None:
                human_code = code_purchase(human_text, codebook)
                human_status = "mapped" if human_code is not None else "unmapped"
            model_code = None
            if model_status == "coded_text" and model_text is not None and codebook is not None:
                model_code = code_purchase(model_text, codebook)
                model_status = "mapped" if model_code is not None else "unmapped"
            price = None if stimulus is None else stimulus.get("price_usd")
            product = None if stimulus is None else str(stimulus.get("product") or "")
            offers.append(
                {
                    "human_code": human_code,
                    "human_status": human_status,
                    "model_code": model_code,
                    "model_status": model_status,
                    "price_usd": price,
                    "product": product,
                    "question_id": question_id,
                    "subject_id": subject_id,
                    "tier": None,
                }
            )
    maxima: dict[str, Decimal] = {}
    for offer in offers:
        price = offer["price_usd"]
        product = str(offer["product"] or "")
        if isinstance(price, Decimal) and price > 0 and product:
            maxima[product] = max(maxima.get(product, price), price)
    for offer in offers:
        price = offer["price_usd"]
        product = str(offer["product"] or "")
        if isinstance(price, Decimal) and product in maxima:
            offer["tier"] = tier_index(price, maxima[product])
        offer["product_max_usd"] = str(maxima[product]) if product in maxima else None
    return offers


def _status_counts(offers: Sequence[Mapping[str, Any]], side: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for offer in offers:
        counts[str(offer[f"{side}_status"])] += 1
    return dict(counts)


def _rate_block(codes: Sequence[int | None], *, assigned: int) -> dict[str, Any]:
    """Mapped conditional rate plus bounds that keep unknowns out of the Yes count."""
    known = [code for code in codes if code is not None]
    yes = sum(known)
    unknown = assigned - len(known)
    mapped = len(known)
    return {
        "assigned": assigned,
        "mapped": mapped,
        "rate_mapped": (yes / mapped) if mapped else None,
        "rate_lower_unknown_as_no": (yes / assigned) if assigned else None,
        "rate_upper_unknown_as_yes": ((yes + unknown) / assigned) if assigned else None,
        "unknown": unknown,
        "yes": yes,
    }


def free_endpoint(offers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Purchase at parsed ``$0.00`` offers. Unknowns stay out of the mapped rate."""
    free = [
        offer
        for offer in offers
        if isinstance(offer.get("price_usd"), Decimal) and is_free_price(offer["price_usd"])
    ]
    both = [
        offer
        for offer in free
        if offer.get("human_code") is not None and offer.get("model_code") is not None
    ]
    human_codes = [offer["human_code"] if isinstance(offer.get("human_code"), int) else None for offer in free]
    model_codes = [offer["model_code"] if isinstance(offer.get("model_code"), int) else None for offer in free]
    paired_human = [int(offer["human_code"]) for offer in both]
    paired_model = [int(offer["model_code"]) for offer in both]
    agreements = sum(left == right for left, right in zip(paired_human, paired_model))
    human_rate = _mean([float(code) for code in paired_human])
    model_rate = _mean([float(code) for code in paired_model])
    gap = None if human_rate is None or model_rate is None else (model_rate - human_rate) * 100
    return {
        "assigned_offers": len(free),
        "both_mapped_offers": len(both),
        "human": _rate_block(human_codes, assigned=len(free)),
        "human_status": _status_counts(free, "human"),
        "model": _rate_block(model_codes, assigned=len(free)),
        "model_status": _status_counts(free, "model"),
        "paired_accuracy": (agreements / len(both)) if both else None,
        "paired_gap_percentage_points": gap,
        "paired_human_rate": human_rate,
        "paired_model_rate": model_rate,
        "products": len({str(offer["product"]) for offer in free if offer.get("product")}),
        "questions": len({str(offer["question_id"]) for offer in free}),
        "subjects": len({str(offer["subject_id"]) for offer in free}),
    }


def _group_rates(offers: Sequence[Mapping[str, Any]], side: str) -> dict[tuple[str, int], dict[str, float]]:
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for offer in offers:
        tier = offer.get("tier")
        product = offer.get("product")
        code = offer.get(f"{side}_code")
        if not isinstance(tier, int) or not product or not isinstance(code, int):
            continue
        buckets[(str(product), tier)].append(code)
    return {
        key: {"n": float(len(codes)), "rate": sum(codes) / len(codes)}
        for key, codes in buckets.items()
    }


def product_curves(offers: Sequence[Mapping[str, Any]], side: str) -> list[dict[str, Any]]:
    """Within-product mapped purchase rates by tier. Empty tiers are omitted."""
    grouped = _group_rates(offers, side)
    rows = []
    for (product, tier), stats in sorted(grouped.items()):
        rows.append(
            {
                "n_mapped": int(stats["n"]),
                "product": product,
                "purchase_rate": stats["rate"],
                "tier": tier,
                "tier_fraction": tier / 10,
            }
        )
    return rows


def equal_product_curve(product_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Unweighted mean of product rates at each tier. A product with no mapped offers is absent."""
    by_tier: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for row in product_rows:
        tier = int(row["tier"])
        by_tier[tier].append(float(row["purchase_rate"]))
        counts[tier] += int(row["n_mapped"])
    curve = []
    for tier in sorted(by_tier):
        rates = by_tier[tier]
        curve.append(
            {
                "n_mapped": counts[tier],
                "n_products": len(rates),
                "purchase_rate": sum(rates) / len(rates),
                "tier": tier,
                "tier_fraction": tier / 10,
            }
        )
    return curve


def offer_weighted_curve(offers: Sequence[Mapping[str, Any]], side: str) -> list[dict[str, Any]]:
    """Each mapped offer has equal weight. Mixes products that have different cell sizes."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for offer in offers:
        tier = offer.get("tier")
        code = offer.get(f"{side}_code")
        if isinstance(tier, int) and isinstance(code, int):
            buckets[tier].append(code)
    return [
        {
            "n_mapped": len(codes),
            "n_products": None,
            "purchase_rate": sum(codes) / len(codes),
            "tier": tier,
            "tier_fraction": tier / 10,
        }
        for tier, codes in sorted(buckets.items())
    ]


def adjacent_upward_jumps(curve: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adjacent observed tiers whose purchase rate rises. Ties are not jumps."""
    ordered = [point for point in curve if point.get("purchase_rate") is not None]
    ordered.sort(key=lambda point: int(point["tier"]))
    jumps = []
    for left, right in zip(ordered, ordered[1:]):
        if int(right["tier"]) != int(left["tier"]) + 1:
            continue
        increase = float(right["purchase_rate"]) - float(left["purchase_rate"])
        if increase > 0:
            jumps.append(
                {
                    "from_rate": float(left["purchase_rate"]),
                    "from_tier": int(left["tier"]),
                    "increase": increase,
                    "to_rate": float(right["purchase_rate"]),
                    "to_tier": int(right["tier"]),
                }
            )
    return jumps


def free_to_first_positive(product_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean of (first positive tier rate - free rate) over products observed at both."""
    by_product: dict[str, dict[int, float]] = defaultdict(dict)
    for row in product_rows:
        by_product[str(row["product"])][int(row["tier"])] = float(row["purchase_rate"])
    jumps = []
    for rates in by_product.values():
        if 0 not in rates:
            continue
        positive = sorted(tier for tier in rates if tier > 0)
        if not positive:
            continue
        first = positive[0]
        jumps.append(rates[first] - rates[0])
    highest = []
    for rates in by_product.values():
        if 0 not in rates:
            continue
        positive = [tier for tier in rates if tier > 0]
        if not positive:
            continue
        highest.append(rates[max(positive)] - rates[0])
    return {
        "max_positive_minus_free": _mean(highest),
        "n_products_free_and_positive": len(jumps),
        "rate_increase_to_first_positive": _mean(jumps),
    }


def _subjects(offers: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(offer["subject_id"]) for offer in offers})


def _resample_mean_gap(offers: Sequence[Mapping[str, Any]], rng: random.Random) -> float | None:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for offer in offers:
        by_subject[str(offer["subject_id"])].append(offer)
    subjects = sorted(by_subject)
    if not subjects:
        return None
    drawn = [subjects[rng.randrange(len(subjects))] for _ in subjects]
    human_sum = 0
    model_sum = 0
    total = 0
    for subject_id in drawn:
        for offer in by_subject[subject_id]:
            human_sum += int(offer["human_code"])
            model_sum += int(offer["model_code"])
            total += 1
    if total == 0:
        return None
    return (model_sum / total - human_sum / total) * 100


def _resample_jump(offers: Sequence[Mapping[str, Any]], side: str, rng: random.Random) -> float | None:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for offer in offers:
        by_subject[str(offer["subject_id"])].append(offer)
    subjects = sorted(by_subject)
    if not subjects:
        return None
    drawn = [subjects[rng.randrange(len(subjects))] for _ in subjects]
    resampled: list[Mapping[str, Any]] = []
    for subject_id in drawn:
        resampled.extend(by_subject[subject_id])
    summary = free_to_first_positive(product_curves(resampled, side))
    value = summary["rate_increase_to_first_positive"]
    return None if value is None else float(value)


def _interval(samples: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "ci_high": _percentile(ordered, 0.975),
        "ci_low": _percentile(ordered, 0.025),
        "n_valid": len(ordered),
    }


def bootstrap_free_gap(
    offers: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Subject-cluster percentile interval for the paired free-price gap, in percentage points."""
    paired = [
        offer
        for offer in offers
        if isinstance(offer.get("price_usd"), Decimal)
        and is_free_price(offer["price_usd"])
        and isinstance(offer.get("human_code"), int)
        and isinstance(offer.get("model_code"), int)
    ]
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        gap = _resample_mean_gap(paired, rng)
        if gap is not None:
            samples.append(gap)
    return {"replicates": replicates, "seed": seed, **_interval(samples)}


def bootstrap_free_jump(
    offers: Sequence[Mapping[str, Any]],
    side: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Subject-cluster interval for the product-equal-weight free-to-first-positive increase."""
    usable = [
        offer
        for offer in offers
        if isinstance(offer.get("tier"), int) and isinstance(offer.get(f"{side}_code"), int)
    ]
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        jump = _resample_jump(usable, side, rng)
        if jump is not None:
            samples.append(jump)
    return {"replicates": replicates, "seed": seed, **_interval(samples)}


def classify_free_endpoint(jump: float | None, ci_low: float | None, ci_high: float | None) -> str:
    """Describe the free-to-next-tier step without a causal or paper-replication claim.

    ``confirmed_in_this_sample`` means the point estimate rises as the price
    leaves zero and the subject-cluster percentile interval stays positive.
    ``not_confirmed`` means the estimate does not rise and the interval stays
    non-positive. Every other pattern, including an interval that covers zero,
    is ``inconclusive``.
    """
    if jump is None or ci_low is None or ci_high is None:
        return "inconclusive"
    if jump > 0 and ci_low > 0:
        return "confirmed_in_this_sample"
    if jump <= 0 and ci_high <= 0:
        return "not_confirmed"
    return "inconclusive"
