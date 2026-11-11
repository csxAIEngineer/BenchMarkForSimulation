"""Arm A checks and the extra three-option metrics.

Official six-metric and free-tier numbers still come from ``score_run`` and
the demand helpers. This module only builds the locked triplets, the
conditional curve, and the section 10 checks.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from popbench.evaluate.stimulus import parse_stimulus, tier_index
from popbench.simulate.three_option import PERMS, to_semantic

HUMAN_NEED_REJECTION = Decimal("0.191")
SHARE_OF_SIX = 1 / 6
PERM_TOLERANCE = 0.02


def content_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash the logical triplet rows, independent of parquet metadata."""
    lines = []
    ordered = sorted(rows, key=lambda row: (str(row["pid"]), str(row["question_id"])))
    for row in ordered:
        lines.append(
            json.dumps(
                {
                    "pid": str(row["pid"]),
                    "price": str(row["price"]),
                    "product_id": str(row["product_id"]),
                    "question_id": str(row["question_id"]),
                    "tier": int(row["tier"]),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_maxima(outcome_rows: Sequence[Mapping[str, object]], sample_ids: set[str]) -> dict[str, Decimal]:
    prices: dict[str, list[Decimal]] = defaultdict(list)
    for row in outcome_rows:
        subject_id = str(row.get("subject_id") or "")
        if subject_id not in sample_ids:
            continue
        items = row.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            stimulus = parse_stimulus(str(item.get("question_text") or ""))
            if stimulus is None:
                continue
            price = stimulus["price_usd"]
            if isinstance(price, Decimal) and price > 0:
                prices[str(stimulus["product"])].append(price)
    return {product: max(values) for product, values in prices.items()}


def triplets_from_responses(
    responses: Sequence[Mapping[str, object]],
    outcomes_by_subject: Mapping[str, Sequence[Mapping[str, object]]],
    product_max: Mapping[str, Decimal],
) -> list[dict[str, object]]:
    """Join baseline response keys to the locked stimulus. Does not resample prices."""
    found: list[dict[str, object]] = []
    for row in responses:
        subject_id = str(row.get("subject_id") or row.get("pid") or "")
        question_id = str(row.get("question_id") or "")
        items = outcomes_by_subject.get(subject_id)
        if items is None:
            raise ValueError(f"subject {subject_id} is missing from outcomes")
        item = next((candidate for candidate in items if str(candidate.get("question_id")) == question_id), None)
        if item is None:
            raise ValueError(f"{subject_id} {question_id} is missing from outcomes")
        stimulus = parse_stimulus(str(item.get("question_text") or ""))
        if stimulus is None:
            raise ValueError(f"{subject_id} {question_id} stimulus does not parse")
        product_id = str(stimulus["product"])
        price = stimulus["price_usd"]
        if not isinstance(price, Decimal):
            raise ValueError(f"{subject_id} {question_id} price is not a decimal")
        maximum = product_max.get(product_id)
        if maximum is None:
            raise ValueError(f"{product_id} has no positive price")
        tier = tier_index(price, maximum)
        if tier is None:
            raise ValueError(f"{product_id} price {price} is off the grid")
        found.append(
            {
                "pid": subject_id,
                "price": f"{price:.2f}",
                "product_id": product_id,
                "question_id": question_id,
                "tier": tier,
            }
        )
    return found


def write_triplets(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    """Write a deterministic parquet file and return the sha256 of its bytes."""
    ordered = sorted(rows, key=lambda row: (str(row["pid"]), str(row["question_id"])))
    table = pa.table(
        {
            "pid": [str(row["pid"]) for row in ordered],
            "product_id": [str(row["product_id"]) for row in ordered],
            "question_id": [str(row["question_id"]) for row in ordered],
            "price": [str(row["price"]) for row in ordered],
            "tier": [int(row["tier"]) for row in ordered],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return file_sha256(path)


def read_triplets(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    return [
        {
            "pid": str(row["pid"]),
            "price": str(row["price"]),
            "product_id": str(row["product_id"]),
            "question_id": str(row["question_id"]),
            "tier": int(row["tier"]),
        }
        for row in rows
    ]


def permutation_balance(option_orders: Sequence[Sequence[str]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, ...]] = Counter(tuple(order) for order in option_orders)
    total = sum(counts.values())
    rows = []
    for perm in PERMS:
        count = counts.get(perm, 0)
        share = (count / total) if total else 0.0
        rows.append(
            {
                "count": count,
                "option_order": ",".join(perm),
                "pass": abs(share - SHARE_OF_SIX) <= PERM_TOLERANCE,
                "share": share,
            }
        )
    return rows


def class_shares(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Full-sample shares. Unmapped rows are not assigned to a class."""
    total = len(rows)
    counts = Counter(row.get("semantic") for row in rows)
    if total == 0:
        return {"P_buy": 0.0, "P_need": 0.0, "P_price": 0.0, "P_unmapped": 0.0}
    return {
        "P_buy": counts.get("buy", 0) / total,
        "P_need": counts.get("no_need", 0) / total,
        "P_price": counts.get("too_expensive", 0) / total,
        "P_unmapped": counts.get(None, 0) / total,
    }


def product_equal_rates(rows: Sequence[Mapping[str, object]], *, conditional: bool) -> dict[int, float]:
    """Mean of within-product rates at each tier. A product with an empty cell is absent."""
    buckets: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        semantic = row.get("semantic")
        if semantic not in {"buy", "no_need", "too_expensive"}:
            continue
        tier = row.get("tier")
        product = row.get("product_id")
        if not isinstance(tier, int) or not product:
            continue
        buckets[(str(product), tier)][str(semantic)] += 1
    by_tier: dict[int, list[float]] = defaultdict(list)
    for (_product, tier), counts in buckets.items():
        if conditional:
            denominator = counts["buy"] + counts["too_expensive"]
        else:
            denominator = counts["buy"] + counts["no_need"] + counts["too_expensive"]
        if denominator == 0:
            continue
        by_tier[tier].append(counts["buy"] / denominator)
    return {tier: sum(rates) / len(rates) for tier, rates in by_tier.items() if rates}


def interpolate_p50(rates: Mapping[int, float]) -> dict[str, Any]:
    """Interpolate the 50 percent point on tier*20 (0 to 200).

    An upward step is recorded and interpolation is skipped. A curve that
    never falls through 50 percent is reported as above 200 percent.
    """
    upward = []
    for tier in range(10):
        if tier not in rates or (tier + 1) not in rates:
            continue
        if rates[tier + 1] > rates[tier]:
            upward.append(
                {
                    "from_rate": rates[tier],
                    "from_tier": tier,
                    "to_rate": rates[tier + 1],
                    "to_tier": tier + 1,
                }
            )
    if upward:
        return {"p50": None, "reason": "upward tier; interpolation not forced", "upward_tiers": upward}
    for tier in range(10):
        left = rates.get(tier)
        right = rates.get(tier + 1)
        if left is None or right is None:
            continue
        if left >= 0.5 >= right:
            if left == right:
                value = float(tier * 20)
            else:
                weight = (left - 0.5) / (left - right)
                value = (tier * 20) + weight * 20
            return {"p50": value, "reason": None, "upward_tiers": []}
    return {"p50": None, "reason": "P50 > 200%", "upward_tiers": []}


def semantic_spot_checks(rows: Sequence[Mapping[str, object]], limit: int = 5) -> list[dict[str, object]]:
    """Recompute semantics for replies whose display order is not A,B,C."""
    checks = []
    for row in rows:
        order = row.get("option_order")
        letter = row.get("parsed_letter")
        if not isinstance(order, list) or tuple(order) == ("A", "B", "C"):
            continue
        if not isinstance(letter, str):
            continue
        expected = to_semantic(letter, order)
        checks.append(
            {
                "expected_semantic": expected,
                "option_order": list(order),
                "parsed_letter": letter,
                "pass": expected == row.get("semantic"),
                "pid": row.get("pid"),
                "product_id": row.get("product_id"),
                "stored_semantic": row.get("semantic"),
            }
        )
        if len(checks) == limit:
            break
    return checks
